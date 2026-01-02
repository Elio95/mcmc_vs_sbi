"""
Stochastic Volatility Model 1 (SV1) with two parameters: sigma (step_size) and nu (degrees of freedom)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, Exponential, constraints
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, t as student_t
from statsmodels.tsa.stattools import pacf
from sbi.inference import SNPE_C, SNLE
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer.hmc import hmc
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect
import time
import os


# ============================================================================
# Data Loading
# ============================================================================

def load_sp500_data(data_path="SP500.csv"):
    """Load S&P 500 data and compute log-returns."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"S&P 500 data not found at {data_path}. "
        )
    
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    df["change"] = np.log(df["Close"]).diff()
    df.dropna(inplace=True)
    return df


# ============================================================================
# Prior Distribution (Log-Space for Flow)
# ============================================================================

class LogSpaceExponentialSVPrior(Distribution):
    """
    Prior for (sigma, nu) ~ Exp(rate_step) × Exp(rate_nu)
    but parameterized in log-space: z = (log_sigma, log_nu)
    """
    support = constraints.real
    
    def __init__(self, rate_step=50.0, rate_nu=0.1, validate_args=False):
        super().__init__(validate_args=validate_args)
        self.rate_step = rate_step
        self.rate_nu = rate_nu
        self.exp_step = Exponential(rate_step)
        self.exp_nu = Exponential(rate_nu)

    def sample(self, sample_shape=torch.Size()):
        """Sample from exponential, return log-space."""
        step_samples = self.exp_step.sample(sample_shape)
        nu_samples = self.exp_nu.sample(sample_shape)
        return torch.stack([step_samples.log(), nu_samples.log()], dim=-1)

    def log_prob(self, z):
        """Log probability with Jacobian correction."""
        z_sigma = z[..., 0]
        z_nu = z[..., 1]
        sigma = z_sigma.exp()
        nu = z_nu.exp()
        
        log_p_sigma = (torch.log(torch.tensor(self.rate_step, device=z.device))
                      - self.rate_step * sigma + z_sigma)
        log_p_nu = (torch.log(torch.tensor(self.rate_nu, device=z.device))
                   - self.rate_nu * nu + z_nu)
        
        return log_p_sigma + log_p_nu


# ============================================================================
# Simulator
# ============================================================================

def simulate_sv(ztheta, n_samples, rng=None):
    """
    Simulate SV model in log-space.
    ztheta: (log_sigma, log_nu)
    Returns: simulated log-returns
    """
    log_sigma = ztheta[0].item()
    log_nu = ztheta[1].item()
    sigma = np.exp(log_sigma)
    nu = np.exp(log_nu)
    
    if rng is None:
        rng = np.random.default_rng()
    
    h = np.zeros(n_samples)
    h[0] = 0.0
    
    for t in range(1, n_samples):
        eta = rng.normal(0, sigma)
        h[t] = h[t-1] + eta
    
    epsilon = rng.standard_t(df=nu, size=n_samples)
    y = np.exp(h / 2.0) * epsilon
    
    return torch.from_numpy(y).float()


# ============================================================================
# Summary Statistics
# ============================================================================

def calculate_summary_statistics(y):
    """
    Compute 9-dimensional summary statistics.
    [mean, log_var, ac1, ac2, ac3, pac1, pac2, skew, kurtosis]
    """
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    
    if len(y_np) < 5:
        return torch.zeros(9, dtype=torch.float32)
    
    mean_ = np.mean(y_np)
    var_ = np.var(y_np) + 1e-6
    var_log = np.log(var_)
    
    def autocorr(x, lag):
        if lag >= len(x):
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    
    ac1 = autocorr(y_np, 1)
    ac2 = autocorr(y_np, 2)
    ac3 = autocorr(y_np, 3)
    acs = np.nan_to_num([ac1, ac2, ac3], nan=0.0, posinf=0.0, neginf=0.0)
    
    pac = pacf(y_np, nlags=2, method="yw")
    pac_lag1 = pac[1] if len(pac) > 1 else 0.0
    pac_lag2 = pac[2] if len(pac) > 2 else 0.0
    pacs = np.nan_to_num([pac_lag1, pac_lag2], nan=0.0, posinf=0.0, neginf=0.0)
    
    sk_ = skew(y_np)
    kt_ = kurtosis(y_np, fisher=True)
    
    arr = np.array([
        mean_, var_log,
        acs[0], acs[1], acs[2],
        pacs[0], pacs[1],
        sk_, kt_
    ], dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr)


def gen_stats_mean_std(sim_fun, prior, num_sim=2000, n_data=2000):
    """Generate normalization statistics."""
    stats_all = []
    for _ in range(num_sim):
        ztheta = prior.sample((1,)).squeeze(0)
        y_ = sim_fun(ztheta, n_data)
        if torch.isnan(y_).any():
            continue
        s_ = calculate_summary_statistics(y_)
        stats_all.append(s_.numpy())
    
    arr = np.array(stats_all)
    m = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd[sd == 0] = 1e-6
    return m, sd


def normalize_summary_stats(s, m_s, s_s):
    """Normalize summary statistics."""
    return (s - m_s) / s_s


# ============================================================================
# MCMC with NumPyro
# ============================================================================

def sv_model(returns):
    T = returns.shape[0]
    step_size = numpyro.sample("step_size", dist.Exponential(50.0))
    s = numpyro.sample("s", dist.GaussianRandomWalk(scale=step_size, num_steps=T))
    nu = numpyro.sample("nu", dist.Exponential(0.1))
    numpyro.sample("r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs=returns)


def run_mcmc_nuts(returns, num_warmup=2000, num_samples=5000, rng_seed=42):
    """Run NUTS MCMC."""
    numpyro.set_platform("cpu")
    init_rng_key, sample_rng_key = random.split(random.PRNGKey(rng_seed))
    
    model_info = initialize_model(init_rng_key, sv_model, model_args=(returns,))
    init_kernel, sample_kernel = hmc(model_info.potential_fn, algo="NUTS")
    hmc_state = init_kernel(model_info.param_info, num_warmup, rng_key=sample_rng_key)
    
    samples = fori_collect(
        num_warmup,
        num_warmup + num_samples,
        sample_kernel,
        hmc_state,
        transform=lambda state: model_info.postprocess_fn(state.z),
        progbar=True
    )
    return samples


class SummaryStatsEmbedding(nn.Module):
    """
    Embedding network for summary statistics.
    """
    def __init__(self, input_dim=9, hidden_dim=250):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return self.fc(x)


class ParameterEmbedding(nn.Module):
    """
    Embedding network for parameters.
    """
    def __init__(self, input_dim=2, hidden_dim=250):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, theta):
        return self.fc(theta)


def build_simulator_wrapped(prior, mean_s, std_s, T):
    """Build normalized simulator."""
    def simulator_batch(ztheta_batch):
        out = []
        rng_ = np.random.default_rng()
        for i in range(ztheta_batch.shape[0]):
            ret_ = simulate_sv(ztheta_batch[i], T, rng=rng_)
            if torch.isnan(ret_).any():
                out.append(torch.zeros(len(mean_s)))
            else:
                s_ = calculate_summary_statistics(ret_)
                normed = (s_ - torch.from_numpy(mean_s)) / torch.from_numpy(std_s)
                out.append(normed)
        return torch.stack(out, dim=0)
    return simulator_batch


def train_snpe(prior, simulator_fn, s_obs_norm, num_rounds=5, sims_per_round=5000):
    embedding_net = SummaryStatsEmbedding(input_dim=9, hidden_dim=250)
    
    density_estimator = posterior_nn(
        model="maf",
        hidden_features=64,
        num_transforms=10,
        embedding_net=embedding_net,
        z_score_theta="independent",
        z_score_x="independent",
    )
    
    snpe = SNPE_C(prior=prior, density_estimator=density_estimator)
    proposal = prior
    
    start_time = time.time()
    
    for rd in range(num_rounds):
        print(f"[SNPE-C Round {rd+1}/{num_rounds}]")
        ztheta_sims = proposal.sample((sims_per_round,))
        x_sims = simulator_fn(ztheta_sims)
        snpe.append_simulations(ztheta_sims, x_sims, proposal=proposal)
        
        density_est = snpe.train(
            training_batch_size=256,
            learning_rate=1e-4,
            max_num_epochs=1000,
            validation_fraction=0.1,
            stop_after_epochs=20,
            show_train_summary=True,
        )
        
        posterior = snpe.build_posterior(density_est).set_default_x(s_obs_norm)
        proposal = posterior
    
    runtime = time.time() - start_time
    ztheta_samples = posterior.sample((10000,), x=s_obs_norm).cpu()
    
    step_samples = torch.exp(ztheta_samples[:, 0]).numpy()
    nu_samples = torch.exp(ztheta_samples[:, 1]).numpy()
    
    return step_samples, nu_samples, runtime


def train_snle(prior, simulator_fn, s_obs_norm, num_rounds=5, sims_per_round=5000):
    embedding_net = ParameterEmbedding(input_dim=2, hidden_dim=250)
    
    density_estimator = likelihood_nn(
        model="maf",
        hidden_features=64,
        num_transforms=10,
        embedding_net=embedding_net,
        z_score_theta="independent",
        z_score_x="independent",
    )
    
    snle = SNLE(prior=prior, density_estimator=density_estimator)
    proposal = prior
    
    start_time = time.time()
    
    for rd in range(num_rounds):
        print(f"[SNLE Round {rd+1}/{num_rounds}]")
        ztheta_sims = proposal.sample((sims_per_round,))
        x_sims = simulator_fn(ztheta_sims)
        snle.append_simulations(ztheta_sims, x_sims)
        
        snle.train(
            training_batch_size=256,
            learning_rate=1e-4,
            max_num_epochs=1000,
            validation_fraction=0.1,
            stop_after_epochs=20,
            show_train_summary=True,
        )
        
        posterior = snle.build_posterior(
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized"
        ).set_default_x(s_obs_norm)
        proposal = posterior
    
    runtime = time.time() - start_time
    ztheta_samples = posterior.sample((10000,), x=s_obs_norm).cpu()
    
    step_samples = torch.exp(ztheta_samples[:, 0]).numpy()
    nu_samples = torch.exp(ztheta_samples[:, 1]).numpy()
    
    return step_samples, nu_samples, runtime


# ============================================================================
# Visualization
# ============================================================================

def plot_density_comparison(samples_mcmc, samples_snpe, samples_snle, 
                            param_names, save_path=None):
    """Plot overlay density plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'MCMC': 'blue', 'SNPE-C': 'green', 'SNLE': 'red'}
    
    for j, (param, ax) in enumerate(zip(param_names, axes)):
        sns.kdeplot(samples_mcmc[:, j], fill=True, alpha=0.4, 
                   color=colors['MCMC'], label='MCMC', ax=ax)
        sns.kdeplot(samples_snpe[:, j], fill=True, alpha=0.4,
                   color=colors['SNPE-C'], label='SNPE-C', ax=ax)
        sns.kdeplot(samples_snle[:, j], fill=True, alpha=0.4,
                   color=colors['SNLE'], label='SNLE', ax=ax)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pairplot(samples, param_names, filename):
    """Create pairplot for posterior samples."""
    df = pd.DataFrame(samples, columns=param_names)
    g = sns.pairplot(df, diag_kind='kde', corner=True, 
                     plot_kws={'s': 5, 'alpha': 0.5})
    g.fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    """Run SV1 experiment."""
    print("="*70)
    
    os.makedirs('sv1_results', exist_ok=True)
    
    df_sp = load_sp500_data("SP500.csv")
    returns_obs = df_sp["change"].values[:2000]
    returns_jax = jnp.array(returns_obs)
    
    print(f"Observations: {len(returns_obs)}")
    
    prior = LogSpaceExponentialSVPrior(rate_step=50.0, rate_nu=0.1)
    
    mean_s, std_s = gen_stats_mean_std(
        simulate_sv, prior, num_sim=2000, n_data=2000
    )
    
    s_obs = calculate_summary_statistics(torch.from_numpy(returns_obs).float())
    s_obs_norm = normalize_summary_stats(s_obs, mean_s, std_s)
    
    simulator = build_simulator_wrapped(prior, mean_s, std_s, len(returns_obs))
    
    print("\nRunning MCMC (NUTS)")
    start_mcmc = time.time()
    samples_mcmc_dict = run_mcmc_nuts(returns_jax, num_warmup=2000, num_samples=5000)
    mcmc_time = time.time() - start_mcmc
    
    step_mcmc = np.array(samples_mcmc_dict["step_size"])
    nu_mcmc = np.array(samples_mcmc_dict["nu"])
    samples_mcmc = np.column_stack([step_mcmc, nu_mcmc])
    
    print(f"MCMC time: {mcmc_time:.2f}s")
    
    print("\nRunning SNPE-C")
    step_snpe, nu_snpe, snpe_time = train_snpe(
        prior, simulator, s_obs_norm, num_rounds=5, sims_per_round=5000
    )
    samples_snpe = np.column_stack([step_snpe, nu_snpe])
    print(f"SNPE-C time: {snpe_time:.2f}s")
    
    print("\nRunning SNLE")
    step_snle, nu_snle, snle_time = train_snle(
        prior, simulator, s_obs_norm, num_rounds=5, sims_per_round=5000
    )
    samples_snle = np.column_stack([step_snle, nu_snle])
    print(f"SNLE time: {snle_time:.2f}s")
    
    param_names = [r'$\sigma$ (step size)', r'$\nu$ (df)']
    
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"{'Parameter':<20} {'Method':<12} {'Mean':>12} {'Std':>12}")
    print("-"*70)
    
    for i, param in enumerate(param_names):
        print(f"{param:<20} {'MCMC':<12} {samples_mcmc[:, i].mean():>12.4f} "
              f"{samples_mcmc[:, i].std():>12.4f}")
        print(f"{'':20} {'SNPE-C':<12} {samples_snpe[:, i].mean():>12.4f} "
              f"{samples_snpe[:, i].std():>12.4f}")
        print(f"{'':20} {'SNLE':<12} {samples_snle[:, i].mean():>12.4f} "
              f"{samples_snle[:, i].std():>12.4f}")
        print("-"*70)
    
    print(f"\nRuntime Comparison:")
    print(f"  MCMC:   {mcmc_time:.1f}s")
    print(f"  SNPE-C: {snpe_time:.1f}s ({snpe_time/mcmc_time:.2f}× MCMC)")
    print(f"  SNLE:   {snle_time:.1f}s ({snle_time/mcmc_time:.2f}× MCMC)")
    
    plot_density_comparison(samples_mcmc, samples_snpe, samples_snle, param_names,
                           'sv1_results/posterior_comparison.png')
    
    plot_pairplot(samples_mcmc, param_names, 'sv1_results/pairplot_mcmc.png')
    plot_pairplot(samples_snpe, param_names, 'sv1_results/pairplot_snpe.png')
    plot_pairplot(samples_snle, param_names, 'sv1_results/pairplot_snle.png')
    
    results_df = pd.DataFrame({
        'Parameter': param_names * 3,
        'Method': ['MCMC']*2 + ['SNPE-C']*2 + ['SNLE']*2,
        'Mean': np.concatenate([
            samples_mcmc.mean(axis=0),
            samples_snpe.mean(axis=0),
            samples_snle.mean(axis=0)
        ]),
        'Std': np.concatenate([
            samples_mcmc.std(axis=0),
            samples_snpe.std(axis=0),
            samples_snle.std(axis=0)
        ])
    })
    results_df.to_csv('sv1_results/parameter_estimates.csv', index=False)
    
    runtime_df = pd.DataFrame({
        'Method': ['MCMC', 'SNPE-C', 'SNLE'],
        'Runtime (s)': [mcmc_time, snpe_time, snle_time],
        'Speedup': [1.0, mcmc_time/snpe_time, mcmc_time/snle_time]
    })
    runtime_df.to_csv('sv1_results/runtime_comparison.csv', index=False)


if __name__ == "__main__":
    main()

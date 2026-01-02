"""
Pharmacokinetics Model
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
from sbi.utils.torchutils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from sbi.inference import SNPE_C, SNLE
import sys
sys.path.append('.')
from utils import compute_metrics, save_results_table

np.random.seed(2025)
torch.manual_seed(2025)
sns.set(style="ticks", font_scale=1.1)

N_SUBJ = 200
T_POINTS = 10
DOSE = 100.0
TIMES = np.linspace(0.5, 24.0, T_POINTS)
SIGMA_TRUE = 0.5

THETA_TRUE = {'theta_ka': 1.0, 'theta_ke': 0.1, 'theta_V': 20.0}
OMEGA_TRUE = {'omega_ka': 0.2, 'omega_ke': 0.1, 'omega_V': 0.1}


def simulate_pk_dataset(theta):
    """Generate PK dataset with 1-compartment model."""
    th_ka, th_ke, th_V, om_ka, om_ke, om_V = [float(x) for x in theta]
    rows = []
    
    for i in range(N_SUBJ):
        eta_ka = np.random.normal(0, om_ka)
        eta_ke = np.random.normal(0, om_ke)
        eta_V = np.random.normal(0, om_V)
        
        ka = th_ka * np.exp(eta_ka)
        ke = th_ke * np.exp(eta_ke)
        V = th_V * np.exp(eta_V)
        
        if abs(ka - ke) < 1e-8:
            ke = max(1e-8, ke * 0.999)
        
        A = DOSE * ka / (V * (ka - ke))
        conc = A * (np.exp(-ke * TIMES) - np.exp(-ka * TIMES))
        conc += np.random.normal(0.0, SIGMA_TRUE, size=T_POINTS)
        
        for t, c in zip(TIMES, conc):
            rows.append((i, t, c))
    
    return pd.DataFrame(rows, columns=["individual", "time", "concentration"])


def calculate_summary_statistics(df):
    """10-dimensional population-level summary statistics."""
    overall_mean = df.concentration.mean()
    overall_std = df.concentration.std()
    
    g = df.groupby("individual")
    mean_i = g.concentration.mean()
    std_i = g.concentration.std().fillna(0.0)
    
    idx_max = g.concentration.idxmax()
    cmax_i = df.loc[idx_max, "concentration"].values
    tmax_i = df.loc[idx_max, "time"].values
    
    vec = torch.tensor([
        overall_mean, overall_std,
        mean_i.mean(), mean_i.std(),
        std_i.mean(), std_i.std(),
        np.mean(cmax_i), np.std(cmax_i),
        np.mean(tmax_i), np.std(tmax_i)
    ], dtype=torch.float32)
    
    return vec


def run_mcmc_pk(df):
    """Run PyMC MCMC for PK model."""
    with pm.Model() as model:
        theta_ka = pm.Uniform('theta_ka', lower=0, upper=5)
        theta_ke = pm.Uniform('theta_ke', lower=0, upper=1)
        theta_V = pm.Uniform('theta_V', lower=10, upper=50)
        
        omega_ka = pm.Uniform('omega_ka', lower=0, upper=1)
        omega_ke = pm.Uniform('omega_ke', lower=0, upper=1)
        omega_V = pm.Uniform('omega_V', lower=0, upper=1)
        
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        ind = df['individual'].values.astype(int)
        t = df['time'].values.astype(float)
        y = df['concentration'].values.astype(float)
        
        eta_ka = pm.Normal('eta_ka', mu=0, sigma=1, shape=N_SUBJ)
        eta_ke = pm.Normal('eta_ke', mu=0, sigma=1, shape=N_SUBJ)
        eta_V = pm.Normal('eta_V', mu=0, sigma=1, shape=N_SUBJ)
        
        ka = theta_ka * pm.math.exp(omega_ka * eta_ka)
        ke = theta_ke * pm.math.exp(omega_ke * eta_ke)
        V = theta_V * pm.math.exp(omega_V * eta_V)
        
        ke_safe = pm.math.maximum(ke, 1e-8)
        ka_safe = pm.math.maximum(ka, 1e-8)
        denom = pm.math.maximum(ka_safe - ke_safe, 1e-8)
        
        A = DOSE * ka_safe[ind] / (V[ind] * denom[ind])
        mu = A * (pm.math.exp(-ke_safe[ind] * t) - pm.math.exp(-ka_safe[ind] * t))
        
        y_like = pm.Normal("y_like", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(10000, tune=5000, target_accept=0.95, progressbar=False)
    
    post = az.extract(trace, var_names=["theta_ka", "theta_ke", "theta_V",
                                        "omega_ka", "omega_ke", "omega_V"])
    samples = np.column_stack([
        post["theta_ka"].values, post["theta_ke"].values, post["theta_V"].values,
        post["omega_ka"].values, post["omega_ke"].values, post["omega_V"].values
    ])
    
    return samples


class XEmbedSNPE(nn.Module):
    def __init__(self, d_in=10, d_hid=256, p=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.LayerNorm(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class ThetaEmbedSNLE(nn.Module):
    def __init__(self, d_in=6, d_hid=256, p=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, θ):
        return self.net(θ)


def pk_simulator(theta_batch):
    """Simulator returning summary statistics."""
    outs = []
    for row in theta_batch:
        df = simulate_pk_dataset(row.cpu().numpy())
        outs.append(calculate_summary_statistics(df))
    return torch.stack(outs)


def run_snpe_pk(prior, simulator, x_obs, num_rounds=10, sims_per_round=2000):
    """Run SNPE-C inference."""
    snpe = SNPE_C(
        prior=prior,
        density_estimator=posterior_nn(
            model="maf",
            num_transforms=10,
            hidden_features=256,
            embedding_net=XEmbedSNPE(),
            z_score_theta="independent",
            z_score_x="independent",
        )
    )
    
    train_kw = {
        'learning_rate': 1e-3,
        'validation_fraction': 0.10,
        'training_batch_size': 1024,
        'stop_after_epochs': 50,
        'max_num_epochs': 500,
        'show_train_summary': False
    }
    
    proposal = prior
    for r in range(num_rounds):
        θr = proposal.sample((sims_per_round,))
        Xr = simulator(θr)
        snpe.append_simulations(θr, Xr, proposal=proposal)
        snpe.train(**train_kw)
        posterior = snpe.build_posterior()
        proposal = posterior.set_default_x(x_obs)
    
    return posterior


def run_snle_pk(prior, simulator, x_obs, num_rounds=10, sims_per_round=2000):
    """Run SNLE inference."""
    snle = SNLE(
        prior=prior,
        density_estimator=likelihood_nn(
            model="maf",
            num_transforms=10,
            hidden_features=256,
            embedding_net=ThetaEmbedSNLE(),
            z_score_theta="independent",
            z_score_x="independent",
        )
    )
    
    train_kw = {
        'validation_fraction': 0.10,
        'training_batch_size': 1024,
        'stop_after_epochs': 50,
        'max_num_epochs': 500,
        'show_train_summary': False
    }
    
    proposal = prior
    for r in range(num_rounds):
        θr = proposal.sample((sims_per_round,))
        Xr = simulator(θr)
        snle.append_simulations(θr, Xr)
        snle.train(**train_kw)
        posterior = snle.build_posterior(
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized"
        )
        proposal = posterior.set_default_x(x_obs)
    
    return posterior


def standard_inference_experiment(n_realizations=50):
    """Train from scratch for each dataset."""
    print("\n" + "="*70)
    print("STANDARD INFERENCE (Train from scratch per dataset)")
    print("="*70)
    
    true_params = np.array([
        THETA_TRUE['theta_ka'], THETA_TRUE['theta_ke'], THETA_TRUE['theta_V'],
        OMEGA_TRUE['omega_ka'], OMEGA_TRUE['omega_ke'], OMEGA_TRUE['omega_V']
    ])
    
    prior_min = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    prior_max = torch.tensor([5.0, 1.0, 50.0, 1.0, 1.0, 1.0])
    prior = BoxUniform(low=prior_min, high=prior_max)
    
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        df = simulate_pk_dataset(true_params)
        x_obs = calculate_summary_statistics(df)
        
        ref_sims = pk_simulator(prior.sample((5000,)))
        mu_s = ref_sims.mean(0)
        sigma_s = ref_sims.std(0).clamp_min(1e-8)
        x_obs_norm = (x_obs - mu_s) / sigma_s
        
        print("  Running MCMC")
        start_time = time.time()
        samples_mcmc = run_mcmc_pk(df)
        mcmc_time = time.time() - start_time
        
        print("  Running SNPE-C")
        start_time = time.time()
        posterior_snpe = run_snpe_pk(prior, pk_simulator, x_obs_norm)
        samples_snpe = posterior_snpe.sample((10000,), x=x_obs_norm).numpy()
        snpe_time = time.time() - start_time
        
        print("  Running SNLE")
        start_time = time.time()
        posterior_snle = run_snle_pk(prior, pk_simulator, x_obs_norm)
        samples_snle = posterior_snle.sample((5000,), x=x_obs_norm).numpy()
        snle_time = time.time() - start_time
        
        result = {
            'MCMC': {
                'samples': samples_mcmc,
                'time': mcmc_time,
                **compute_metrics(samples_mcmc, true_params)
            },
            'SNPE_C': {
                'samples': samples_snpe,
                'time': snpe_time,
                **compute_metrics(samples_snpe, true_params)
            },
            'SNLE': {
                'samples': samples_snle,
                'time': snle_time,
                **compute_metrics(samples_snle, true_params)
            }
        }
        
        results_list.append(result)
    
    os.makedirs('pk_results', exist_ok=True)
    save_results_table(results_list, 'pk_results/standard_inference_summary.csv')
    
    print("\nStandard inference complete!")
    return results_list


def amortized_inference_experiment(n_realizations=50):
    """Train once, apply to multiple datasets."""
    print("\n" + "="*70)
    print("AMORTIZED INFERENCE (Train once, apply to many datasets)")
    print("="*70)
    
    true_params = np.array([
        THETA_TRUE['theta_ka'], THETA_TRUE['theta_ke'], THETA_TRUE['theta_V'],
        OMEGA_TRUE['omega_ka'], OMEGA_TRUE['omega_ke'], OMEGA_TRUE['omega_V']
    ])
    
    prior_min = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    prior_max = torch.tensor([5.0, 1.0, 50.0, 1.0, 1.0, 1.0])
    prior = BoxUniform(low=prior_min, high=prior_max)
    
    df_dummy = simulate_pk_dataset(true_params)
    x_dummy = calculate_summary_statistics(df_dummy)
    
    ref_sims = pk_simulator(prior.sample((5000,)))
    mu_s = ref_sims.mean(0)
    sigma_s = ref_sims.std(0).clamp_min(1e-8)
    x_dummy_norm = (x_dummy - mu_s) / sigma_s
    
    start_train_snpe = time.time()
    posterior_snpe = run_snpe_pk(prior, pk_simulator, x_dummy_norm, 
                                 num_rounds=10, sims_per_round=2000)
    snpe_train_time = time.time() - start_train_snpe
    print(f"SNPE-C training time: {snpe_train_time:.2f}s")
    
    start_train_snle = time.time()
    posterior_snle = run_snle_pk(prior, pk_simulator, x_dummy_norm,
                                 num_rounds=10, sims_per_round=2000)
    snle_train_time = time.time() - start_train_snle
    print(f"SNLE training time: {snle_train_time:.2f}s")
    
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        df = simulate_pk_dataset(true_params)
        x_obs = calculate_summary_statistics(df)
        x_obs_norm = (x_obs - mu_s) / sigma_s
        
        start_time = time.time()
        samples_snpe = posterior_snpe.sample((10000,), x=x_obs_norm).numpy()
        snpe_time = time.time() - start_time
        
        start_time = time.time()
        samples_snle = posterior_snle.sample((5000,), x=x_obs_norm).numpy()
        snle_time = time.time() - start_time
        
        result = {
            'SNPE_C': {
                'samples': samples_snpe,
                'time': snpe_time,
                **compute_metrics(samples_snpe, true_params)
            },
            'SNLE': {
                'samples': samples_snle,
                'time': snle_time,
                **compute_metrics(samples_snle, true_params)
            }
        }
        
        results_list.append(result)
    
    os.makedirs('pk_results', exist_ok=True)
    save_results_table(results_list, 'pk_results/amortized_inference_summary.csv',
                      methods=['SNPE_C', 'SNLE'])
    
    print("\nAmortized inference complete!")
    print(f"Training time - SNPE-C: {snpe_train_time:.2f}s, SNLE: {snle_train_time:.2f}s")
    
    return results_list


def plot_single_realization(samples_mcmc, samples_snpe, samples_snle, true_params):
    """Plot results for a single realization."""
    param_names = [r"$\theta_{ka}$", r"$\theta_{ke}$", r"$\theta_V$",
                   r"$\omega_{ka}$", r"$\omega_{ke}$", r"$\omega_V$"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {'MCMC': 'blue', 'SNPE-C': 'green', 'SNLE': 'red'}
    
    for j, (param, true_val, ax) in enumerate(zip(param_names, true_params, axes)):
        sns.kdeplot(samples_mcmc[:, j], fill=True, alpha=0.4, color=colors['MCMC'],
                   label='MCMC', ax=ax)
        sns.kdeplot(samples_snpe[:, j], fill=True, alpha=0.4, color=colors['SNPE-C'],
                   label='SNPE-C', ax=ax)
        sns.kdeplot(samples_snle[:, j], fill=True, alpha=0.4, color=colors['SNLE'],
                   label='SNLE', ax=ax)
        
        ax.axvline(true_val, color='black', linestyle='--', linewidth=2, label='True')
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        if j == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('pk_results/posterior_comparison.png', dpi=300)
    plt.close()


def plot_pairplot(samples, param_names, true_vals, filename):
    """Create pairplot for posterior samples."""
    df = pd.DataFrame(samples, columns=param_names)
    g = sns.pairplot(df, diag_kind='kde', corner=True, plot_kws={'s': 5, 'alpha': 0.5})
    axes = g.axes
    
    for i in range(len(param_names)):
        for j in range(i):
            axes[i, j].plot(true_vals[j], true_vals[i], marker='+', 
                          mec='red', mew=2, ms=12)
        axes[i, i].axvline(true_vals[i], color='red', ls='--')
    
    g.fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_ppc(samples, true_params, filename):
    """Posterior predictive check."""
    n_draws = 100
    indices = np.random.choice(len(samples), n_draws, replace=False)
    
    plt.figure(figsize=(10, 6))
    
    df_true = simulate_pk_dataset(true_params)
    plt.hist(df_true.concentration, bins=30, density=True, alpha=0.3, 
            color='blue', label='Observed')
    
    for idx in indices:
        df_sim = simulate_pk_dataset(samples[idx])
        plt.hist(df_sim.concentration, bins=30, density=True, alpha=0.02,
                color='gray')
    
    plt.xlabel('Concentration (mg/L)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("Pharmacokinetics Model")
    print("="*70)
    
    results_standard = standard_inference_experiment(n_realizations=50)
    results_amortized = amortized_inference_experiment(n_realizations=50)
    
    first_result = results_standard[0]
    true_params = np.array([
        THETA_TRUE['theta_ka'], THETA_TRUE['theta_ke'], THETA_TRUE['theta_V'],
        OMEGA_TRUE['omega_ka'], OMEGA_TRUE['omega_ke'], OMEGA_TRUE['omega_V']
    ])
    
    plot_single_realization(
        first_result['MCMC']['samples'],
        first_result['SNPE_C']['samples'],
        first_result['SNLE']['samples'],
        true_params
    )
    
    param_names = [r"$\theta_{ka}$", r"$\theta_{ke}$", r"$\theta_V$",
                   r"$\omega_{ka}$", r"$\omega_{ke}$", r"$\omega_V$"]
    
    plot_pairplot(first_result['MCMC']['samples'], param_names, true_params,
                 'pk_results/pairplot_mcmc.png')
    plot_pairplot(first_result['SNPE_C']['samples'], param_names, true_params,
                 'pk_results/pairplot_snpe.png')
    plot_pairplot(first_result['SNLE']['samples'], param_names, true_params,
                 'pk_results/pairplot_snle.png')
    
    plot_ppc(first_result['MCMC']['samples'], true_params, 
            'pk_results/ppc_mcmc.png')
    plot_ppc(first_result['SNPE_C']['samples'], true_params,
            'pk_results/ppc_snpe.png')
    plot_ppc(first_result['SNLE']['samples'], true_params,
            'pk_results/ppc_snle.png')


if __name__ == "__main__":
    main()

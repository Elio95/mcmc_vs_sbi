"""
Radon Hierarchical Model
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
from sbi.inference import SNPE_C, SNLE, simulate_for_sbi

np.random.seed(2025)
torch.manual_seed(2025)
sns.set(style="ticks", font_scale=1.1)


def load_radon_data():
    """Load and prepare Minnesota radon data."""
    try:
        srrs2 = pd.read_csv(os.path.join("..", "data", "srrs2.dat"))
    except FileNotFoundError:
        srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
    
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()
    
    try:
        cty = pd.read_csv(os.path.join("..", "data", "cty.dat"))
    except FileNotFoundError:
        cty = pd.read_csv(pm.get_data("cty.dat"))
    
    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips
    
    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    
    srrs_mn.county = srrs_mn.county.map(str.strip)
    county, mn_counties = srrs_mn.county.factorize()
    srrs_mn["county_code"] = county
    
    radon = srrs_mn.activity
    log_radon = np.log(radon + 0.1).values
    floor_measure = srrs_mn.floor.values
    
    return log_radon, county, floor_measure, mn_counties


def calculate_summary_statistics_enhanced(y_sim, county_idx, floor_measure):
    """14-dimensional summary statistics for variance components."""
    y_sim = torch.as_tensor(y_sim, dtype=torch.float32)
    county_idx = torch.as_tensor(county_idx, dtype=torch.long)
    floor_measure = torch.as_tensor(floor_measure, dtype=torch.float32)
    
    mu = y_sim.mean()
    sd = y_sim.std()
    
    J = int(county_idx.max().item() + 1)
    alpha_hat = torch.empty(J)
    beta_hat = torch.empty(J)
    resid_var = torch.empty(J)
    county_means = torch.empty(J)
    
    for j in range(J):
        msk = county_idx == j
        x = floor_measure[msk]
        y = y_sim[msk]
        
        if x.numel() == 0:
            alpha_hat[j] = beta_hat[j] = resid_var[j] = county_means[j] = 0.
            continue
        
        x_bar, y_bar = x.mean(), y.mean()
        S_xx = ((x - x_bar) ** 2).sum()
        S_xy = ((x - x_bar) * (y - y_bar)).sum()
        
        beta_j = S_xy / (S_xx + 1e-12)
        alpha_j = y_bar - beta_j * x_bar
        
        y_pred = alpha_j + beta_j * x
        resid_j = y - y_pred
        
        alpha_hat[j] = alpha_j
        beta_hat[j] = beta_j
        resid_var[j] = (resid_j ** 2).mean()
        county_means[j] = y_bar
    
    var_alpha = alpha_hat.var(unbiased=False)
    var_beta = beta_hat.var(unbiased=False)
    cov_ab = ((alpha_hat - alpha_hat.mean()) * 
              (beta_hat - beta_hat.mean())).mean()
    
    median_abs_beta = beta_hat.abs().median()
    iqr_alpha = torch.quantile(alpha_hat, 0.75) - torch.quantile(alpha_hat, 0.25)
    iqr_beta = torch.quantile(beta_hat, 0.75) - torch.quantile(beta_hat, 0.25)
    
    mean_resid_var = resid_var.mean()
    std_resid_var = resid_var.std(unbiased=False)
    
    skewness = ((y_sim - mu) ** 3).mean() / (sd ** 3 + 1e-12)
    kurtosis = ((y_sim - mu) ** 4).mean() / (sd ** 4 + 1e-12)
    
    mean_county_means = county_means.mean()
    std_county_means = county_means.std(unbiased=False)
    
    return torch.tensor([
        mu, sd,
        var_alpha, var_beta, cov_ab,
        median_abs_beta, iqr_alpha, iqr_beta,
        mean_resid_var, std_resid_var,
        skewness, kurtosis,
        mean_county_means, std_county_means
    ], dtype=torch.float32)


def radon_simulator(theta, county_idx, floor_measure, num_counties):
    """Simulate radon data from hierarchical model."""
    num_simulations = theta.shape[0]
    num_observations = len(floor_measure)
    
    y_sim_list = []
    
    for i in range(num_simulations):
        gamma_0 = theta[i, 0]
        gamma_1 = theta[i, 1]
        tau_alpha = theta[i, 2]
        tau_beta = theta[i, 3]
        sigma = theta[i, 4]
        
        z_alpha = torch.randn(num_counties)
        z_beta = torch.randn(num_counties)
        
        u_alpha = z_alpha * tau_alpha
        u_beta = z_beta * tau_beta
        
        alpha_j = gamma_0 + u_alpha
        beta_j = gamma_1 + u_beta
        
        alpha_obs = alpha_j[county_idx]
        beta_obs = beta_j[county_idx]
        
        y_hat = alpha_obs + beta_obs * floor_measure
        y_sim = torch.normal(y_hat, sigma)
        
        y_sim_list.append(y_sim)
    
    return torch.stack(y_sim_list)


def run_mcmc_radon(log_radon, county, floor_measure, mn_counties):
    """Run PyMC MCMC for radon model."""
    coords = {"county": mn_counties}
    
    with pm.Model(coords=coords) as model:
        floor_idx = pm.MutableData("floor_idx", floor_measure, dims="obs_id")
        county_idx = pm.MutableData("county_idx", county, dims="obs_id")
        
        mu_a = pm.Uniform("mu_a", lower=-3, upper=3)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=3)
        
        z_a = pm.Normal("z_a", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")
        
        mu_b = pm.Uniform("mu_b", lower=-3, upper=3)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=3)
        
        z_b = pm.Normal("z_b", mu=0, sigma=1, dims="county")
        beta = pm.Deterministic("beta", mu_b + z_b * sigma_b, dims="county")
        
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=3)
        
        y_hat = alpha[county_idx] + beta[county_idx] * floor_idx
        
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, 
                          observed=log_radon, dims="obs_id")
        
        trace = pm.sample(draws=10000, tune=5000, chains=4, 
                         target_accept=0.95, progressbar=False)
    
    post = az.extract(trace, var_names=["mu_a", "mu_b", "sigma_a", 
                                        "sigma_b", "sigma_y"])
    
    samples = np.column_stack([
        post["mu_a"].values,
        post["mu_b"].values,
        post["sigma_a"].values,
        post["sigma_b"].values,
        post["sigma_y"].values
    ])
    
    return samples


class ResidualBlock(nn.Module):
    def __init__(self, d_hid, p_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_hid)
        self.linear1 = nn.Linear(d_hid, d_hid)
        self.norm2 = nn.LayerNorm(d_hid)
        self.linear2 = nn.Linear(d_hid, d_hid)
        self.dropout = nn.Dropout(p_drop)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.dropout(x)
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return residual + x


class ImprovedEmbeddingNet(nn.Module):
    def __init__(self, d_in, d_hid=256, n_blocks=5, p_drop=0.15):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.LayerNorm(d_hid),
            nn.ReLU(),
            nn.Dropout(p_drop)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(d_hid, p_drop) for _ in range(n_blocks)
        ])
        
        self.output_norm = nn.LayerNorm(d_hid)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_norm(x)


class ThetaEmbed(nn.Module):
    def __init__(self, d_in=5, d_hid=256, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, θ):
        return self.net(θ)


def run_snpe_radon(prior, simulator_normalized, x_obs, num_rounds=10, 
                   sims_per_round=5000):
    """Run SNPE-C inference."""
    embedding_net = ImprovedEmbeddingNet(
        d_in=len(x_obs),
        d_hid=256,
        n_blocks=5,
        p_drop=0.15
    )
    
    snpe = SNPE_C(
        prior=prior,
        density_estimator=posterior_nn(
            model='maf',
            hidden_features=256,
            num_transforms=15,
            num_blocks=3,
            embedding_net=embedding_net,
            z_score_theta='independent',
            z_score_x='independent'
        )
    )
    
    train_kw = {
        'training_batch_size': 512,
        'learning_rate': 5e-4,
        'validation_fraction': 0.15,
        'stop_after_epochs': 30,
        'max_num_epochs': 100,
        'clip_max_norm': 5.0,
        'show_train_summary': False
    }
    
    proposal = prior
    for round_idx in range(num_rounds):
        theta = proposal.sample((sims_per_round,))
        x = simulator_normalized(theta)
        
        snpe.append_simulations(theta, x, proposal=proposal)
        snpe.train(**train_kw)
        
        posterior = snpe.build_posterior()
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    return posterior


def run_snle_radon(prior, simulator_normalized, x_obs, num_rounds=8,
                   sims_per_round=3000):
    """Run SNLE inference."""
    snle = SNLE(
        prior=prior,
        density_estimator=likelihood_nn(
            model='maf',
            hidden_features=256,
            num_transforms=12,
            num_blocks=3,
            z_score_theta='independent',
            z_score_x='independent'
        )
    )
    
    train_kw = {
        'training_batch_size': 256,
        'learning_rate': 3e-4,
        'validation_fraction': 0.15,
        'stop_after_epochs': 25,
        'max_num_epochs': 80,
        'clip_max_norm': 5.0,
        'show_train_summary': False
    }
    
    proposal = prior
    for round_idx in range(num_rounds):
        theta, x = simulate_for_sbi(
            simulator_normalized,
            proposal=proposal,
            num_simulations=sims_per_round,
            num_workers=1
        )
        
        snle.append_simulations(theta, x)
        snle.train(**train_kw)
        
        posterior = snle.build_posterior(
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={
                'num_chains': 20,
                'thin': 5,
                'warmup_steps': 100,
                'init_strategy': 'proposal'
            }
        )
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    return posterior


def plot_density_comparison(samples_mcmc, samples_snpe, samples_snle, param_names):
    """Plot overlay density plots."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    colors = {'MCMC': 'blue', 'SNPE-C': 'green', 'SNLE': 'red'}
    
    for i, (ax, param) in enumerate(zip(axes, param_names)):
        sns.kdeplot(samples_mcmc[:, i], ax=ax, label='MCMC',
                   color=colors['MCMC'], fill=True, alpha=0.4)
        sns.kdeplot(samples_snpe[:, i], ax=ax, label='SNPE-C',
                   color=colors['SNPE-C'], fill=True, alpha=0.4)
        sns.kdeplot(samples_snle[:, i], ax=ax, label='SNLE',
                   color=colors['SNLE'], fill=True, alpha=0.4)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('radon_results/posterior_comparison.png', dpi=300)
    plt.close()


def plot_pairplot(samples, param_names, filename):
    """Create pairplot for posterior samples."""
    df = pd.DataFrame(samples, columns=param_names)
    g = sns.pairplot(df, diag_kind='kde', corner=True, 
                     plot_kws={'s': 5, 'alpha': 0.5})
    g.fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_ppc(samples, log_radon, county, floor_measure, num_counties, filename):
    """Posterior predictive check."""
    n_draws = 100
    grid = np.linspace(log_radon.min(), log_radon.max(), 200)
    
    county_torch = torch.as_tensor(county, dtype=torch.long)
    floor_torch = torch.as_tensor(floor_measure, dtype=torch.float32)
    
    kdes = []
    indices = np.random.choice(len(samples), n_draws, replace=False)
    
    for idx in indices:
        theta = torch.tensor(samples[idx:idx+1], dtype=torch.float32)
        y_sim = radon_simulator(theta, county_torch, floor_torch, 
                               num_counties).squeeze(0).numpy()
        kdes.append(gaussian_kde(y_sim)(grid))
    
    kdes = np.array(kdes)
    median_kde = np.median(kdes, axis=0)
    p25_kde = np.percentile(kdes, 25, axis=0)
    p75_kde = np.percentile(kdes, 75, axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(log_radon, bins=30, stat='density', color='blue',
                alpha=0.25, label='Observed')
    plt.plot(grid, median_kde, color='blue', lw=2, 
            label='Median predictive')
    plt.fill_between(grid, p25_kde, p75_kde, color='blue', alpha=0.25,
                    label='25-75% predictive')
    
    plt.xlabel('log(radon)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("Radon Hierarchical Model")
    print("="*70)
    
    os.makedirs('radon_results', exist_ok=True)
    
    log_radon, county, floor_measure, mn_counties = load_radon_data()
    num_counties = len(mn_counties)
    
    print(f"Data: {len(log_radon)} observations, {num_counties} counties")
    
    county_torch = torch.as_tensor(county, dtype=torch.long)
    floor_torch = torch.as_tensor(floor_measure, dtype=torch.float32)
    
    lower_bounds = torch.tensor([-3.0, -3.0, 0.0, 0.0, 0.0])
    upper_bounds = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
    prior = BoxUniform(low=lower_bounds, high=upper_bounds)
    
    num_prior_sims = 2000
    theta_prior = prior.sample((num_prior_sims,))
    summary_stats_list = []
    
    for i in range(num_prior_sims):
        theta = theta_prior[i:i+1]
        y_sim = radon_simulator(theta, county_torch, floor_torch, 
                               num_counties).squeeze(0)
        summ_stats = calculate_summary_statistics_enhanced(
            y_sim, county_torch, floor_torch
        )
        summary_stats_list.append(summ_stats.numpy())
    
    summary_stats_array = np.array(summary_stats_list)
    mean_summary_stats = summary_stats_array.mean(axis=0)
    std_summary_stats = summary_stats_array.std(axis=0)
    std_summary_stats = np.where(std_summary_stats < 1e-8, 1.0, std_summary_stats)
    
    observed_summary_stats = calculate_summary_statistics_enhanced(
        log_radon, county_torch, floor_torch
    )
    observed_summary_stats_np = observed_summary_stats.numpy()
    observed_summary_stats_normalized = (
        (observed_summary_stats_np - mean_summary_stats) / std_summary_stats
    )
    observed_summary_stats_normalized = torch.tensor(
        observed_summary_stats_normalized, dtype=torch.float32
    )
    
    def radon_simulator_normalized(theta):
        y_sim_tensor = radon_simulator(theta, county_torch, floor_torch, 
                                       num_counties)
        summary_stats_list = []
        
        for i in range(y_sim_tensor.shape[0]):
            y_sim = y_sim_tensor[i]
            summ_stats = calculate_summary_statistics_enhanced(
                y_sim, county_torch, floor_torch
            )
            summary_stats_list.append(summ_stats)
        
        summary_stats_sim = torch.stack(summary_stats_list)
        summary_stats_normalized = (
            (summary_stats_sim - torch.tensor(mean_summary_stats, dtype=torch.float32)) / 
            torch.tensor(std_summary_stats, dtype=torch.float32)
        )
        return summary_stats_normalized
    
    print("\nRunning MCMC")
    start_mcmc = time.time()
    samples_mcmc = run_mcmc_radon(log_radon, county, floor_measure, mn_counties)
    mcmc_time = time.time() - start_mcmc
    print(f"MCMC time: {mcmc_time:.2f}s")
    
    print("\nRunning SNPE-C")
    start_snpe = time.time()
    posterior_snpe = run_snpe_radon(prior, radon_simulator_normalized, 
                                    observed_summary_stats_normalized)
    samples_snpe = posterior_snpe.sample(
        (10000,), x=observed_summary_stats_normalized
    ).numpy()
    snpe_time = time.time() - start_snpe
    print(f"SNPE-C time: {snpe_time:.2f}s")
    
    print("\nRunning SNLE")
    start_snle = time.time()
    posterior_snle = run_snle_radon(prior, radon_simulator_normalized,
                                    observed_summary_stats_normalized)
    samples_snle = posterior_snle.sample(
        (5000,), x=observed_summary_stats_normalized
    ).numpy()
    snle_time = time.time() - start_snle
    print(f"SNLE time: {snle_time:.2f}s")
    
    param_names = [r'$\mu_{\alpha}$', r'$\mu_{\beta}$', 
                   r'$\sigma_{\alpha}$', r'$\sigma_{\beta}$', r'$\sigma_y$']
    
    print("\nSummary Statistics:")
    print("="*70)
    print(f"{'Parameter':<20} {'Method':<12} {'Mean':>12} {'Std':>12}")
    print("-"*70)
    
    for i, param in enumerate(param_names):
        mean_mcmc = samples_mcmc[:, i].mean()
        std_mcmc = samples_mcmc[:, i].std()
        print(f"{param:<20} {'MCMC':<12} {mean_mcmc:>12.4f} {std_mcmc:>12.4f}")
        
        mean_snpe = samples_snpe[:, i].mean()
        std_snpe = samples_snpe[:, i].std()
        print(f"{'':20} {'SNPE-C':<12} {mean_snpe:>12.4f} {std_snpe:>12.4f}")
        
        mean_snle = samples_snle[:, i].mean()
        std_snle = samples_snle[:, i].std()
        print(f"{'':20} {'SNLE':<12} {mean_snle:>12.4f} {std_snle:>12.4f}")
        print("-"*70)
    
    print(f"\nRuntime Comparison:")
    print(f"  MCMC:   {mcmc_time:.1f}s")
    print(f"  SNPE-C: {snpe_time:.1f}s ({snpe_time/mcmc_time:.2f}× MCMC)")
    print(f"  SNLE:   {snle_time:.1f}s ({snle_time/mcmc_time:.2f}× MCMC)")
    
    plot_density_comparison(samples_mcmc, samples_snpe, samples_snle, param_names)
    
    plot_pairplot(samples_mcmc, param_names, 'radon_results/pairplot_mcmc.png')
    plot_pairplot(samples_snpe, param_names, 'radon_results/pairplot_snpe.png')
    plot_pairplot(samples_snle, param_names, 'radon_results/pairplot_snle.png')
    
    plot_ppc(samples_mcmc, log_radon, county, floor_measure, num_counties,
            'radon_results/ppc_mcmc.png')
    plot_ppc(samples_snpe, log_radon, county, floor_measure, num_counties,
            'radon_results/ppc_snpe.png')
    plot_ppc(samples_snle, log_radon, county, floor_measure, num_counties,
            'radon_results/ppc_snle.png')
    
    results_df = pd.DataFrame({
        'Parameter': param_names * 3,
        'Method': ['MCMC']*5 + ['SNPE-C']*5 + ['SNLE']*5,
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
    results_df.to_csv('radon_results/parameter_estimates.csv', index=False)
    
    runtime_df = pd.DataFrame({
        'Method': ['MCMC', 'SNPE-C', 'SNLE'],
        'Runtime (s)': [mcmc_time, snpe_time, snle_time],
        'Speedup': [1.0, mcmc_time/snpe_time, mcmc_time/snle_time]
    })
    runtime_df.to_csv('radon_results/runtime_comparison.csv', index=False)
    


if __name__ == "__main__":
    main()

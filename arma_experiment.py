"""
ARMA(2,1) Case Study
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS
from sbi import inference as sbi_inference
from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from statsmodels.tsa.stattools import pacf
from scipy.stats import skew, kurtosis, gaussian_kde
import time
import os
import matplotlib.pyplot as plt
from utils import (compute_metrics, compute_coverage, save_results_table,
                   plot_density_comparison, plot_pairplot, plot_traceplot)


# Configuration
T = 500
phi_true = [0.22, -0.1]
theta_true = 0.5
sigma2_true = 1.0
sigma_true = np.sqrt(sigma2_true)

# Prior bounds
prior_min = torch.tensor([-1.0, -1.0, -1.0, 0.1])
prior_max = torch.tensor([1.0, 1.0, 1.0, 5.0])
prior = BoxUniform(low=prior_min, high=prior_max)


def generate_arma_data(n_samples, phi, theta, sigma2, seed=None):
    """Generate ARMA(2,1) time series."""
    if seed is not None:
        np.random.seed(seed)
    
    phi1, phi2 = phi
    sigma = np.sqrt(sigma2)
    
    y = np.zeros(n_samples)
    epsilon = np.random.normal(0.0, sigma, n_samples)
    
    for t in range(2, n_samples):
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + epsilon[t] + theta * epsilon[t-1]
    
    return y[2:]


def arma_numpyro_model(y):
    """NumPyro model for ARMA(2,1)."""
    n = y.shape[0]
    
    phi1 = numpyro.sample('phi1', dist.Uniform(-1, 1))
    phi2 = numpyro.sample('phi2', dist.Uniform(-1, 1))
    theta = numpyro.sample('theta', dist.Uniform(-1, 1))
    sigma = numpyro.sample('sigma', dist.Uniform(0.1, 5.0))
    
    y_prev2 = y[0]
    y_prev1 = y[1]
    epsilon_prev = 0.0
    
    def transition_fn(carry, t):
        y_prev2, y_prev1, epsilon_prev = carry
        mu_t = phi1 * y_prev1 + phi2 * y_prev2 + theta * epsilon_prev
        y_t = y[t]
        epsilon_t = y_t - mu_t
        numpyro.sample(f'y_{t}', dist.Normal(mu_t, sigma), obs=y_t)
        return (y_prev1, y_t, epsilon_t), None
    
    scan(transition_fn, (y_prev2, y_prev1, epsilon_prev), jnp.arange(2, n))


def run_mcmc_arma(y_data):
    """Run MCMC inference for single dataset."""
    y_jnp = jnp.array(y_data)
    
    rng_key = random.PRNGKey(np.random.randint(0, 10000))
    nuts_kernel = NUTS(arma_numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=10000, num_chains=4)
    
    start_time = time.time()
    mcmc.run(rng_key, y=y_jnp)
    runtime = time.time() - start_time
    
    samples = mcmc.get_samples(group_by_chain=False)
    
    samples_array = np.column_stack([
        np.array(samples['phi1']),
        np.array(samples['phi2']),
        np.array(samples['theta']),
        np.array(samples['sigma'])
    ])
    
    return samples_array, runtime


def simulate_arma(theta, n_samples):
    """Simulate ARMA(2,1) for SBI."""
    phi1 = theta[0].item()
    phi2 = theta[1].item()
    theta_ma = theta[2].item()
    sigma = theta[3].item()
    
    if sigma <= 0 or not np.isfinite(sigma):
        return torch.full((n_samples,), float('nan'))
    
    y = torch.zeros(n_samples + 2)
    epsilon = sigma * torch.randn(n_samples + 2)
    
    y[0] = torch.randn(1).item()
    y[1] = torch.randn(1).item()
    
    for t in range(2, n_samples + 2):
        y_prev1 = y[t-1]
        y_prev2 = y[t-2]
        epsilon_prev = epsilon[t-1]
        mu_t = phi1 * y_prev1 + phi2 * y_prev2 + theta_ma * epsilon_prev
        y[t] = mu_t + epsilon[t]
    
    y = y[2:]
    
    if not torch.isfinite(y).all():
        return torch.full((n_samples,), float('nan'))
    
    return y


def calculate_summary_stats(y):
    """Calculate summary statistics for ARMA data."""
    y = y.detach().numpy() if torch.is_tensor(y) else y
    n = len(y)
    burn_in = int(0.1 * n)
    y = y[burn_in:]
    
    mean_y = np.mean(y)
    var_y = np.var(y, ddof=0) + 1e-6
    var_y_log = np.log(var_y)
    
    autocorr_y = [np.corrcoef(y[:-lag], y[lag:])[0, 1] if lag < len(y) else 0.0
                  for lag in range(1, 4)]
    autocorr_y = np.nan_to_num(autocorr_y)
    
    pacf_values = pacf(y, nlags=2, method='yw_adjusted')
    pacf_y = pacf_values[1:]
    
    skewness_y = skew(y)
    kurtosis_y = kurtosis(y, fisher=True)
    
    summary_stats = np.concatenate((
        [mean_y, var_y_log],
        autocorr_y,
        pacf_y,
        [skewness_y, kurtosis_y]
    ))
    
    summary_stats = np.nan_to_num(summary_stats, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(summary_stats, dtype=torch.float32)


def compute_normalization_stats(simulator_fn, prior, n_sims=1000, n_samples=500):
    """Compute mean and std for normalizing summary statistics."""
    stats_list = []
    for _ in range(n_sims):
        theta = prior.sample((1,)).squeeze(0)
        y_sim = simulator_fn(theta, n_samples)
        stats = calculate_summary_stats(y_sim)
        stats_list.append(stats.numpy())
    
    stats_array = np.array(stats_list)
    mean_s = np.mean(stats_array, axis=0)
    std_s = np.std(stats_array, axis=0)
    std_s[std_s == 0] = 1e-6
    
    return mean_s, std_s


def create_simulator_wrapper(mean_s, std_s, n_samples):
    """Create simulator that returns normalized summary statistics."""
    mean_s_torch = torch.tensor(mean_s, dtype=torch.float32)
    std_s_torch = torch.tensor(std_s, dtype=torch.float32)
    
    def wrapper(theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        
        n_sims = theta.shape[0]
        stats_list = []
        
        for i in range(n_sims):
            y_sim = simulate_arma(theta[i], n_samples)
            stats = calculate_summary_stats(y_sim)
            norm_stats = (stats - mean_s_torch) / std_s_torch
            norm_stats[torch.isnan(norm_stats)] = 0.0
            norm_stats[torch.isinf(norm_stats)] = 0.0
            stats_list.append(norm_stats)
        
        return torch.stack(stats_list)
    
    return wrapper


class SummaryStatsEmbedding(nn.Module):
    """Embedding network for summary statistics (SNPE-C)."""
    def __init__(self, input_dim, hidden_dim=250):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
    
    def forward(self, x):
        return self.fc(x)


class ParameterEmbedding(nn.Module):
    """Embedding network for parameters (SNLE)."""
    def __init__(self, input_dim=4, hidden_dim=250):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
    
    def forward(self, theta):
        return self.fc(theta)


def train_snpe_arma(prior, simulator, x_obs, summary_dim):
    """Train SNPE-C from scratch for single dataset."""
    embedding_net = SummaryStatsEmbedding(input_dim=summary_dim)
    embedding_net.train()
    
    density_estimator = posterior_nn(
        model='maf',
        hidden_features=64,
        num_transforms=10,
        embedding_net=embedding_net,
        z_score_theta='independent',
        z_score_x='independent'
    )
    
    training_params = {
        'training_batch_size': 100,
        'stop_after_epochs': 20,
        'validation_fraction': 0.1,
        'learning_rate': 0.00001,
        'max_num_epochs': 1000,
        'show_train_summary': False
    }
    
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    
    inference = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator)
    
    start_time = time.time()
    
    num_rounds = 10
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference.append_simulations(theta, x, proposal=proposal)
        density_est = inference.train(**training_params)
        optimizer.step()
        scheduler.step()
        posterior = inference.build_posterior(density_est)
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((10000,), x=x_obs)
    
    return samples.numpy(), runtime


def train_snle_arma(prior, simulator, x_obs):
    """Train SNLE from scratch for single dataset."""
    theta_embedding = ParameterEmbedding(input_dim=4, hidden_dim=250)
    theta_embedding.train()
    
    density_estimator = likelihood_nn(
        model='maf',
        hidden_features=64,
        num_transforms=10,
        embedding_net=theta_embedding,
        z_score_theta='independent',
        z_score_x='independent'
    )
    
    training_params = {
        'training_batch_size': 100,
        'stop_after_epochs': 20,
        'validation_fraction': 0.1,
        'learning_rate': 0.00001,
        'max_num_epochs': 1000,
        'show_train_summary': False
    }
    
    optimizer = torch.optim.Adam(theta_embedding.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    
    inference = sbi_inference.SNLE(prior=prior, density_estimator=density_estimator)
    
    start_time = time.time()
    
    num_rounds = 10
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference.append_simulations(theta, x)
        density_est = inference.train(**training_params)
        optimizer.step()
        scheduler.step()
        posterior = inference.build_posterior(
            density_est,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((5000,), x=x_obs)
    
    return samples.numpy(), runtime


def compute_nltp_kde(samples, true_value):
    """Compute NLTP using KDE."""
    kde = gaussian_kde(samples.T)
    pdf_at_true = kde(true_value)
    if pdf_at_true > 0:
        return -np.log(pdf_at_true)
    else:
        return 1e10


def posterior_predictive_arma(theta_samples, n_orig, obs_len, n_sims=500):
    """Generate posterior predictive samples."""
    indices = np.random.choice(len(theta_samples), size=n_sims, replace=False)
    all_sims = np.zeros((n_sims, obs_len))
    
    for i, idx in enumerate(indices):
        phi1 = theta_samples[idx, 0]
        phi2 = theta_samples[idx, 1]
        theta = theta_samples[idx, 2]
        sigma = theta_samples[idx, 3]
        sigma2 = sigma**2
        sim_data = generate_arma_data(n_orig, [phi1, phi2], theta, sigma2)
        all_sims[i, :] = sim_data
    
    return all_sims


def plot_ppc_arma(y_obs, y_pred, method_name, save_path=None):
    """Plot posterior predictive check."""
    median_ppc = np.median(y_pred, axis=0)
    lower_ppc = np.percentile(y_pred, 25, axis=0)
    upper_ppc = np.percentile(y_pred, 75, axis=0)
    
    plt.figure(figsize=(16, 6))
    plt.plot(y_obs, label='Observed', color='black')
    plt.plot(median_ppc, label='Median', color='blue')
    plt.fill_between(range(len(y_obs)), lower_ppc, upper_ppc,
                    color='blue', alpha=0.3, label='25-75 percentile')
    plt.xlabel('Time')
    plt.ylabel('y_t')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_standard_inference():
    """Run standard (non-amortized) inference on 50 datasets."""
    print("="*80)
    print("STANDARD INFERENCE: Training from scratch for each dataset")
    print("="*80)
    
    # Compute normalization stats once
    print("Computing normalization statistics")
    mean_s, std_s = compute_normalization_stats(simulate_arma, prior, n_sims=1000, n_samples=T)
    simulator = create_simulator_wrapper(mean_s, std_s, T)
    
    # Get summary dimension
    dummy_theta = prior.sample((1,))
    dummy_x = simulator(dummy_theta)
    summary_dim = dummy_x.shape[1]
    
    n_realizations = 50
    all_results = []
    true_values = np.array([phi_true[0], phi_true[1], theta_true, sigma_true])
    
    for real_idx in range(n_realizations):
        print(f"\n--- Realization {real_idx + 1}/{n_realizations} ---")
        
        seed = 1000 + real_idx
        y_data = generate_arma_data(T + 2, phi_true, theta_true, sigma2_true, seed=seed)
        
        # MCMC
        print("Running MCMC")
        mcmc_samples, mcmc_time = run_mcmc_arma(y_data)
        mcmc_metrics = compute_metrics(mcmc_samples, true_values)
        mcmc_nltp = compute_nltp_kde(mcmc_samples, true_values)
        mcmc_coverage = compute_coverage(mcmc_samples, true_values)
        
        # Compute normalized summary stats for SBI
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        summary_stats_obs = calculate_summary_stats(y_tensor)
        norm_stats_obs = (summary_stats_obs - torch.tensor(mean_s)) / torch.tensor(std_s)
        
        # SNPE-C
        print("Running SNPE-C")
        snpe_samples, snpe_time = train_snpe_arma(prior, simulator, norm_stats_obs, summary_dim)
        snpe_metrics = compute_metrics(snpe_samples, true_values)
        snpe_coverage = compute_coverage(snpe_samples, true_values)
        
        # SNLE
        print("Running SNLE")
        snle_samples, snle_time = train_snle_arma(prior, simulator, norm_stats_obs)
        snle_metrics = compute_metrics(snle_samples, true_values)
        snle_coverage = compute_coverage(snle_samples, true_values)
        
        result = {
            'realization': real_idx,
            'MCMC': {**mcmc_metrics, 'runtime': mcmc_time, 'nltp': mcmc_nltp,
                    'coverage': mcmc_coverage, 'samples': mcmc_samples},
            'SNPE_C': {**snpe_metrics, 'runtime': snpe_time,
                      'coverage': snpe_coverage, 'samples': snpe_samples},
            'SNLE': {**snle_metrics, 'runtime': snle_time,
                    'coverage': snle_coverage, 'samples': snle_samples}
        }
        all_results.append(result)
        
        print(f"  MCMC: {mcmc_time:.2f}s, SNPE-C: {snpe_time:.2f}s, SNLE: {snle_time:.2f}s")
    
    return all_results, mean_s, std_s


def train_amortized_networks(mean_s, std_s):
    """Train networks once for amortized inference."""
    print("\n" + "="*80)
    print("AMORTIZED INFERENCE: Training networks once")
    print("="*80)
    
    simulator = create_simulator_wrapper(mean_s, std_s, T)
    
    # Get summary dimension
    dummy_theta = prior.sample((1,))
    dummy_x = simulator(dummy_theta)
    summary_dim = dummy_x.shape[1]
    
    # Train SNPE-C
    embedding_net_snpe = SummaryStatsEmbedding(input_dim=summary_dim)
    embedding_net_snpe.train()
    
    density_estimator_snpe = posterior_nn(
        model='maf',
        hidden_features=64,
        num_transforms=10,
        embedding_net=embedding_net_snpe,
        z_score_theta='independent',
        z_score_x='independent'
    )
    
    training_params = {
        'training_batch_size': 100,
        'stop_after_epochs': 20,
        'validation_fraction': 0.1,
        'learning_rate': 0.00001,
        'max_num_epochs': 1000,
        'show_train_summary': False
    }
    
    optimizer_snpe = torch.optim.Adam(embedding_net_snpe.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler_snpe = CosineAnnealingLR(optimizer_snpe, T_max=1000)
    
    inference_snpe = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator_snpe)
    
    start_time_snpe = time.time()
    num_rounds = 10
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}")
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference_snpe.append_simulations(theta, x, proposal=proposal)
        density_est = inference_snpe.train(**training_params)
        optimizer_snpe.step()
        scheduler_snpe.step()
        posterior_snpe = inference_snpe.build_posterior(density_est)
        dummy_obs = simulator(prior.sample((1,)))
        posterior_snpe = posterior_snpe.set_default_x(dummy_obs.squeeze(0))
        proposal = posterior_snpe
    
    snpe_training_time = time.time() - start_time_snpe
    print(f"SNPE-C training time: {snpe_training_time:.2f}s")
    
    # Train SNLE
    theta_embedding = ParameterEmbedding(input_dim=4, hidden_dim=250)
    theta_embedding.train()
    
    density_estimator_snle = likelihood_nn(
        model='maf',
        hidden_features=64,
        num_transforms=10,
        embedding_net=theta_embedding,
        z_score_theta='independent',
        z_score_x='independent'
    )
    
    optimizer_snle = torch.optim.Adam(theta_embedding.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler_snle = CosineAnnealingLR(optimizer_snle, T_max=1000)
    
    inference_snle = sbi_inference.SNLE(prior=prior, density_estimator=density_estimator_snle)
    
    start_time_snle = time.time()
    proposal = prior
    
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}")
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference_snle.append_simulations(theta, x)
        density_est = inference_snle.train(**training_params)
        optimizer_snle.step()
        scheduler_snle.step()
        posterior_snle = inference_snle.build_posterior(
            density_est,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        dummy_obs = simulator(prior.sample((1,)))
        posterior_snle = posterior_snle.set_default_x(dummy_obs.squeeze(0))
        proposal = posterior_snle
    
    snle_training_time = time.time() - start_time_snle
    print(f"SNLE training time: {snle_training_time:.2f}s")
    
    return posterior_snpe, posterior_snle, snpe_training_time, snle_training_time


def run_amortized_inference(posterior_snpe, posterior_snle, mean_s, std_s):
    """Run fast amortized inference on 50 datasets."""
    print("\n" + "="*80)
    print("Running amortized inference on 50 datasets")
    print("="*80)
    
    n_realizations = 50
    snpe_times = []
    snle_times = []
    
    for real_idx in range(n_realizations):
        seed = 1000 + real_idx
        y_data = generate_arma_data(T + 2, phi_true, theta_true, sigma2_true, seed=seed)
        
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        summary_stats_obs = calculate_summary_stats(y_tensor)
        norm_stats_obs = (summary_stats_obs - torch.tensor(mean_s)) / torch.tensor(std_s)
        
        # SNPE-C inference
        start = time.time()
        snpe_samples = posterior_snpe.sample((10000,), x=norm_stats_obs)
        snpe_times.append(time.time() - start)
        
        # SNLE inference
        start = time.time()
        snle_samples = posterior_snle.sample((5000,), x=norm_stats_obs)
        snle_times.append(time.time() - start)
        
        if (real_idx + 1) % 10 == 0:
            print(f"Completed {real_idx + 1}/{n_realizations} datasets")
    
    print(f"\nAverage SNPE-C inference time: {np.mean(snpe_times):.4f}s")
    print(f"Average SNLE inference time: {np.mean(snle_times):.4f}s")
    
    return snpe_times, snle_times


if __name__ == "__main__":
    output_dir = "arma_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard inference
    standard_results, mean_s, std_s = run_standard_inference()
    
    # Save results
    param_names = ['phi1', 'phi2', 'theta', 'sigma']
    true_values = np.array([phi_true[0], phi_true[1], theta_true, sigma_true])
    
    save_results_table(standard_results, f"{output_dir}/standard_results.csv")
    
    # Plot results for last realization
    last_result = standard_results[-1]
    samples_dict = {
        'MCMC': last_result['MCMC']['samples'],
        'SNPE_C': last_result['SNPE_C']['samples'],
        'SNLE': last_result['SNLE']['samples']
    }
    
    plot_density_comparison(samples_dict, true_values, 
                           param_names, f"{output_dir}/density_comparison.png")
    
    # PPCs
    y_obs = generate_arma_data(T + 2, phi_true, theta_true, sigma2_true)
    for method in ['MCMC', 'SNPE_C', 'SNLE']:
        y_pred = posterior_predictive_arma(last_result[method]['samples'], T + 2, T)
        plot_ppc_arma(y_obs, y_pred, method, f"{output_dir}/{method}_ppc.png")
    
    # Amortized inference
    posterior_snpe, posterior_snle, train_time_snpe, train_time_snle = train_amortized_networks(mean_s, std_s)
    snpe_times, snle_times = run_amortized_inference(posterior_snpe, posterior_snle, mean_s, std_s)
"""
SIR Epidemic Model
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, multivariate_normal
from sbi import inference as sbi_inference
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
import time
import os
from utils import (compute_metrics, compute_coverage, save_results_table,
                   plot_density_comparison, plot_pairplot, plot_traceplot)


# Configuration
N = 1000000
T = 160
num_obs = 10
obs_times = np.linspace(0, T, num_obs)

# True parameters
beta_true = 0.4
gamma_true = 1 / 8
theta_true = np.log([beta_true, gamma_true])

# Prior (LogNormal)
loc = torch.tensor([np.log(0.4), np.log(1/8)], dtype=torch.float32)
scale = torch.tensor([0.5, 0.2], dtype=torch.float32)
prior = torch.distributions.Independent(torch.distributions.LogNormal(loc, scale), 1)


def stochastic_sir_gillespie(beta, gamma, N, T, initial_state):
    """Simulate SIR model using Gillespie algorithm."""
    S, I, R = initial_state
    time = 0.0
    
    times = [time]
    S_series = [S]
    I_series = [I]
    R_series = [R]
    
    N_total = N
    
    while time < T and I > 0:
        rate_infection = beta * S * I / N_total
        rate_recovery = gamma * I
        total_rate = rate_infection + rate_recovery
        
        if total_rate == 0:
            break
        
        tau = np.random.exponential(scale=1/total_rate)
        time += tau
        
        if np.random.rand() < rate_infection / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
        
        times.append(time)
        S_series.append(S)
        I_series.append(I)
        R_series.append(R)
    
    return np.array(times), np.array(S_series), np.array(I_series), np.array(R_series)


def simulator(theta):
    """Simulator wrapper for SBI."""
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    
    num_simulations = theta.shape[0]
    observations = []
    
    initial_state = (N - 1, 1, 0)
    
    for i in range(num_simulations):
        beta = theta[i, 0].item()
        gamma = theta[i, 1].item()
        
        try:
            times, S_series, I_series, R_series = stochastic_sir_gillespie(beta, gamma, N, T, initial_state)
            
            if times[-1] < T:
                raise ValueError("Simulation ended before T.")
            
            I_series_interp = np.interp(obs_times, times, I_series)
            I_series_interp = np.clip(I_series_interp, 0, N)
        except Exception:
            observations.append(torch.full((num_obs,), float('nan')))
            continue
        
        observations.append(torch.tensor(I_series_interp, dtype=torch.float32))
    
    return torch.stack(observations)


def drift_sir(S, I, theta, N):
    """Drift function for SDE approximation."""
    beta, gamma = np.exp(theta)
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    return np.array([dS, dI])


def diffusion_sir(S, I, theta, N):
    """Diffusion function for SDE approximation."""
    beta, gamma = np.exp(theta)
    var_S = max(beta * S * I / N, 1e-6)
    var_I = max(beta * S * I / N + gamma * I, 1e-6)
    cov_SI = -beta * S * I / N
    cov_matrix = np.array([[var_S, cov_SI], [cov_SI, var_I]])
    return cov_matrix


def simulate_sir_sde(theta, N, T, delta_t, initial_state):
    """Simulate SIR using SDE approximation."""
    S0, I0 = initial_state
    num_steps = int(T / delta_t)
    times = np.linspace(0, T, num_steps + 1)
    S_path = np.zeros(num_steps + 1)
    I_path = np.zeros(num_steps + 1)
    S_path[0] = S0
    I_path[0] = I0
    
    for t in range(num_steps):
        S = S_path[t]
        I = I_path[t]
        if I <= 0:
            S_path[t+1:] = S
            I_path[t+1:] = 0
            break
        
        drift_vec = drift_sir(S, I, theta, N) * delta_t
        diffusion_mat = diffusion_sir(S, I, theta, N) * delta_t
        noise = np.random.multivariate_normal(mean=[0, 0], cov=diffusion_mat)
        S_next = S + drift_vec[0] + noise[0]
        I_next = I + drift_vec[1] + noise[1]
        
        S_next = max(min(S_next, N), 0)
        I_next = max(min(I_next, N - S_next), 0)
        
        S_path[t+1] = S_next
        I_path[t+1] = I_next
    
    return times, S_path, I_path


def llik_euler(S_path, I_path, theta, delta_t, N):
    """Log-likelihood under Euler-Maruyama approximation."""
    log_likelihood = 0.0
    T = len(S_path) - 1
    for t in range(T):
        S_t = S_path[t]
        I_t = I_path[t]
        S_next = S_path[t+1]
        I_next = I_path[t+1]
        drift_vec = drift_sir(S_t, I_t, theta, N) * delta_t
        cov_matrix = diffusion_sir(S_t, I_t, theta, N) * delta_t
        cov_matrix += 1e-6 * np.eye(2)
        mu = np.array([S_t, I_t]) + drift_vec
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            return -np.inf
        det_cov = np.linalg.det(cov_matrix)
        if det_cov <= 0 or not np.isfinite(det_cov):
            return -np.inf
        diff = np.array([S_next, I_next]) - mu
        log_p = -0.5 * np.log(det_cov) - 0.5 * diff.T @ cov_inv @ diff
        log_likelihood += log_p
    return log_likelihood


def run_mcmc_sir(I_obs, num_iterations=50000, burn_in=10000):
    """Run MCMC inference for single dataset."""
    theta_current = np.log([0.3, 1/10])
    
    def log_prior(theta):
        beta, gamma = np.exp(theta)
        logp_beta = stats.lognorm(s=0.5, scale=0.4).logpdf(beta)
        logp_gamma = stats.lognorm(s=0.2, scale=1/8).logpdf(gamma)
        return logp_beta + logp_gamma
    
    delta_t = 1.0
    times_grid = np.arange(0, T + delta_t, delta_t)
    obs_indices = np.searchsorted(times_grid, obs_times)
    
    S0 = N - I_obs[0]
    I0 = I_obs[0]
    initial_state = (S0, I0)
    
    times_sim, S_current, I_current = simulate_sir_sde(theta_current, N, T, delta_t, initial_state)
    if len(times_sim) != len(times_grid):
        S_current = np.interp(times_grid, times_sim, S_current)
        I_current = np.interp(times_grid, times_sim, I_current)
    
    theta_prop_std = np.array([0.1, 0.1])
    
    samples_theta = []
    acceptance_counts = {'theta': 0, 'latent_states': 0}
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        times_sim, S_proposed, I_proposed = simulate_sir_sde(theta_current, N, T, delta_t, initial_state)
        if len(times_sim) != len(times_grid):
            S_proposed = np.interp(times_grid, times_sim, S_proposed)
            I_proposed = np.interp(times_grid, times_sim, I_proposed)
        
        log_target_current = llik_euler(S_current, I_current, theta_current, delta_t, N)
        log_target_proposed = llik_euler(S_proposed, I_proposed, theta_current, delta_t, N)
        log_accept_ratio_latent = log_target_proposed - log_target_current
        
        if np.isfinite(log_accept_ratio_latent) and np.log(np.random.rand()) < log_accept_ratio_latent:
            S_current = S_proposed.copy()
            I_current = I_proposed.copy()
            acceptance_counts['latent_states'] += 1
        
        theta_proposal = theta_current + np.random.normal(0, theta_prop_std)
        log_prior_current = log_prior(theta_current)
        log_prior_proposal = log_prior(theta_proposal)
        
        I_sim_current_obs = I_current[obs_indices]
        log_lik_current = -0.5 * np.sum(((I_obs - I_sim_current_obs) / 10) ** 2)
        log_lik_proposal = -0.5 * np.sum(((I_obs - I_sim_current_obs) / 10) ** 2)
        log_accept_ratio_theta = (log_prior_proposal + log_lik_proposal) - (log_prior_current + log_lik_current)
        
        if np.isfinite(log_accept_ratio_theta) and np.log(np.random.rand()) < log_accept_ratio_theta:
            theta_current = theta_proposal.copy()
            acceptance_counts['theta'] += 1
        
        if iteration >= burn_in:
            samples_theta.append(theta_current.copy())
    
    runtime = time.time() - start_time
    samples_theta = np.array(samples_theta)
    
    return samples_theta, runtime


def train_snpe_sir(prior, simulator_fn, x_o):
    """Train SNPE-C from scratch for single dataset."""
    training_params = {
        'learning_rate': 0.0005,
        'validation_fraction': 0.1,
        'stop_after_epochs': 100,
        'max_num_epochs': 1000,
        'show_train_summary': False
    }
    
    inference = sbi_inference.SNPE(prior=prior, density_estimator='maf')
    
    start_time = time.time()
    
    num_rounds = 5
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator_fn(theta)
        
        valid_indices = ~torch.isnan(x).any(dim=1)
        theta = theta[valid_indices]
        x = x[valid_indices]
        
        if len(theta) == 0:
            break
        
        _ = inference.append_simulations(theta, x, proposal=proposal).train(**training_params)
        posterior = inference.build_posterior()
        posterior = posterior.set_default_x(x_o)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((10000,), x=x_o)
    
    samples_log = torch.log(samples)
    return samples_log.numpy(), runtime


def train_snle_sir(prior, simulator_fn, x_o):
    """Train SNLE from scratch for single dataset."""
    training_params = {
        'learning_rate': 0.0005,
        'validation_fraction': 0.1,
        'stop_after_epochs': 100,
        'max_num_epochs': 1000,
        'show_train_summary': False
    }
    
    inference = sbi_inference.SNLE(prior=prior, density_estimator='maf')
    
    start_time = time.time()
    
    num_rounds = 5
    num_sims_per_round = 1000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator_fn(theta)
        
        valid_indices = ~torch.isnan(x).any(dim=1)
        theta = theta[valid_indices]
        x = x[valid_indices]
        
        if len(theta) == 0:
            break
        
        likelihood_estimator = inference.append_simulations(theta, x).train(**training_params)
        posterior = inference.build_posterior(
            likelihood_estimator,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        posterior = posterior.set_default_x(x_o)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((5000,), x=x_o)
    
    samples_log = torch.log(samples)
    return samples_log.numpy(), runtime


def compute_nltp_kde(samples, true_value):
    """Compute NLTP using KDE for MCMC samples."""
    kde = gaussian_kde(samples.T)
    pdf_at_true = kde(true_value)
    if pdf_at_true > 0:
        return -np.log(pdf_at_true)
    else:
        return 1e10


def posterior_predictive_sir(theta_samples, initial_state, n_samples=500):
    """Generate posterior predictive samples."""
    indices = np.random.choice(len(theta_samples), size=n_samples, replace=False)
    
    simulated_I = []
    
    for idx in indices:
        theta_log = theta_samples[idx]
        beta, gamma = np.exp(theta_log)
        times_sim, S_series_sim, I_series_sim, R_series_sim = stochastic_sir_gillespie(
            beta, gamma, N, T, initial_state)
        
        if times_sim[-1] < obs_times[-1]:
            continue
        
        I_series = np.interp(obs_times, times_sim, I_series_sim)
        simulated_I.append(I_series)
    
    return np.array(simulated_I)


def plot_ppc_sir(I_obs, I_pred, method_name, save_path=None):
    """Plot posterior predictive check."""
    median_I = np.median(I_pred, axis=0)
    lower_I = np.percentile(I_pred, 25, axis=0)
    upper_I = np.percentile(I_pred, 75, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(obs_times, I_obs, color='black', label='Observed', zorder=5)
    plt.plot(obs_times, median_I, color='blue', label='Median')
    plt.fill_between(obs_times, lower_I, upper_I, color='blue', alpha=0.3, label='25-75 percentile')
    
    plt.xlabel('Time')
    plt.ylabel('Number of Infectious Individuals')
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
    
    n_realizations = 50
    all_results = []
    
    for real_idx in range(n_realizations):
        print(f"\n--- Realization {real_idx + 1}/{n_realizations} ---")
        
        torch.manual_seed(1000 + real_idx)
        np.random.seed(1000 + real_idx)
        
        initial_state = (N - 1, 1, 0)
        times, S_series, I_series, R_series = stochastic_sir_gillespie(
            beta_true, gamma_true, N, T, initial_state)
        
        I_obs = np.interp(obs_times, times, I_series)
        x_o = torch.tensor(I_obs, dtype=torch.float32)
        
        print("Running MCMC")
        mcmc_samples, mcmc_time = run_mcmc_sir(I_obs)
        mcmc_metrics = compute_metrics(mcmc_samples, theta_true)
        mcmc_nltp = compute_nltp_kde(mcmc_samples, theta_true)
        mcmc_coverage = compute_coverage(mcmc_samples, theta_true)
        
        print("Running SNPE-C")
        snpe_samples, snpe_time = train_snpe_sir(prior, simulator, x_o)
        snpe_metrics = compute_metrics(snpe_samples, theta_true)
        snpe_coverage = compute_coverage(snpe_samples, theta_true)
        
        print("Running SNLE")
        snle_samples, snle_time = train_snle_sir(prior, simulator, x_o)
        snle_metrics = compute_metrics(snle_samples, theta_true)
        snle_coverage = compute_coverage(snle_samples, theta_true)
        
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
    
    return all_results


if __name__ == "__main__":
    output_dir = "sir_results"
    os.makedirs(output_dir, exist_ok=True)
    
    standard_results = run_standard_inference()
    
    param_names = ['log_beta', 'log_gamma']
    save_results_table(standard_results, f"{output_dir}/standard_results.csv")
    
    last_result = standard_results[-1]
    samples_dict = {
        'MCMC': last_result['MCMC']['samples'],
        'SNPE_C': last_result['SNPE_C']['samples'],
        'SNLE': last_result['SNLE']['samples']
    }
    
    plot_density_comparison(samples_dict, theta_true, 
                           param_names, f"{output_dir}/density_comparison.png")
    
    initial_state = (N - 1, 1, 0)
    for method in ['MCMC', 'SNPE_C', 'SNLE']:
        I_pred = posterior_predictive_sir(last_result[method]['samples'], initial_state)
        times, S_series, I_series, R_series = stochastic_sir_gillespie(
            beta_true, gamma_true, N, T, initial_state)
        I_obs = np.interp(obs_times, times, I_series)
        plot_ppc_sir(I_obs, I_pred, method, f"{output_dir}/{method}_ppc.png")

"""
Lotka-Volterra Predator-Prey Model
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, multivariate_normal
from sbi import inference as sbi_inference
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
import time
import os
from utils import (compute_metrics, compute_coverage, save_results_table,
                   plot_density_comparison, plot_pairplot, plot_traceplot)


# Configuration
dur = 30
dt = 0.2
max_steps = 1000000
x_start = 100
y_start = 50

# True parameters in log scale
theta_true = np.log([1, 0.01, 0.01, 0.5])
alpha_true, beta_true, delta_true, gamma_true = np.exp(theta_true)

# Prior bounds
prior_min = torch.tensor([-5.0, -5.0, -5.0, -5.0])
prior_max = torch.tensor([2.0, 2.0, 2.0, 2.0])
prior = BoxUniform(low=prior_min, high=prior_max)


def gen_exponential(lam):
    """Generate exponential random variable."""
    return -torch.log(torch.rand(1)) / lam


def update_state(reac, x_current, y_current):
    """Update population state based on reaction."""
    if reac == 0:
        x_new = x_current + 1
        y_new = y_current
    elif reac == 1:
        x_new = max(x_current - 1, 0)
        y_new = y_current
    elif reac == 2:
        x_new = x_current
        y_new = y_current + 1
    elif reac == 3:
        x_new = x_current
        y_new = max(y_current - 1, 0)
    return x_new, y_new


def lotka_volterra_gillespie(theta):
    """Simulate Lotka-Volterra using Gillespie algorithm."""
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()
    
    if theta.ndim == 2 and theta.shape[0] == 1:
        theta = theta[0]
    
    alpha, beta, delta, gamma = np.exp(theta)
    
    num_sim_steps = int(dur / dt)
    path = torch.zeros((2, num_sim_steps + 1))
    nbr_steps = 0
    current_time = 0
    time_var = 0
    
    path[0, 0] = x_start
    path[1, 0] = y_start
    
    for i in range(num_sim_steps):
        x_current = path[0, i]
        y_current = path[1, i]
        
        while current_time < (i + 1) * dt:
            x_int = int(x_current.item())
            y_int = int(y_current.item())
            
            x_int = max(x_int, 0)
            y_int = max(y_int, 0)
            
            r1 = alpha * x_int
            r2 = beta * x_int * y_int
            r3 = delta * x_int * y_int
            r4 = gamma * y_int
            
            r_sum = r1 + r2 + r3 + r4
            
            if r_sum == 0:
                time_var = float('inf')
                break
            
            tau = gen_exponential(r_sum).item()
            time_var += tau
            
            if time_var > (i + 1) * dt:
                break
            
            probs = torch.tensor([r1, r2, r3, r4]) / r_sum
            reac = torch.multinomial(probs, 1).item()
            
            x_current, y_current = update_state(reac, x_int, y_int)
            x_current = torch.tensor(x_current, dtype=torch.float32)
            y_current = torch.tensor(y_current, dtype=torch.float32)
            
            nbr_steps += 1
            
            if nbr_steps > max_steps:
                return path
            
            current_time = time_var
        
        path[0, i + 1] = x_current
        path[1, i + 1] = y_current
        current_time = (i + 1) * dt
    
    return path


def calculate_summary_statistics(simulation_output):
    """Calculate summary statistics for LV model."""
    x_series = torch.tensor(simulation_output[0])
    y_series = torch.tensor(simulation_output[1])
    
    mean_x = x_series.mean()
    mean_y = y_series.mean()
    var_x = x_series.var(unbiased=False)
    var_y = y_series.var(unbiased=False)
    
    epsilon = 1e-6
    std_x = torch.sqrt(var_x + epsilon)
    std_y = torch.sqrt(var_y + epsilon)
    
    x_tmp = (x_series - mean_x) / std_x
    y_tmp = (y_series - mean_y) / std_y
    
    autocorr_x_lag1 = torch.dot(x_tmp[1:], x_tmp[:-1]) / (len(x_series) - 1)
    autocorr_y_lag1 = torch.dot(y_tmp[1:], y_tmp[:-1]) / (len(y_series) - 1)
    autocorr_x_lag2 = torch.dot(x_tmp[2:], x_tmp[:-2]) / (len(x_series) - 2)
    autocorr_y_lag2 = torch.dot(y_tmp[2:], y_tmp[:-2]) / (len(y_series) - 2)
    crosscorr_xy = torch.dot(x_tmp, y_tmp) / (len(x_series) - 1)
    
    var_x_log = torch.log(var_x + epsilon)
    var_y_log = torch.log(var_y + epsilon)
    
    return torch.tensor([mean_x, var_x_log, autocorr_x_lag1, autocorr_x_lag2,
                         mean_y, var_y_log, autocorr_y_lag1, autocorr_y_lag2, crosscorr_xy])


def gen_summary_stats_mean_and_std(simulation_function, prior, nbr_sim=1000, seed=100, cutoff=0.0125):
    """Generate normalization statistics."""
    torch.manual_seed(seed)
    summary_stats_list = []
    for _ in range(nbr_sim):
        theta_sampled = prior.sample((1,)).squeeze(0)
        simulation_output = simulation_function(theta_sampled)
        summary_stats = calculate_summary_statistics(simulation_output)
        summary_stats_list.append(summary_stats.numpy())
    
    summary_stats_array = np.array(summary_stats_list)
    m_s = stats.trim_mean(summary_stats_array, cutoff, axis=0)
    s_s = np.std(summary_stats_array, axis=0)
    return m_s, s_s


def normalize_summary_stats(s, m_s, s_s):
    """Normalize summary statistics."""
    return (s - m_s) / s_s


def simulation_wrapper(m_s, s_s):
    """Create simulation wrapper with normalization."""
    def wrapper(theta):
        num_simulations = len(theta)
        normalized_summary_stats_list = []
        for i in range(num_simulations):
            simulated_data = lotka_volterra_gillespie(theta[i])
            summary_stats = calculate_summary_statistics(simulated_data)
            normalized_summary_stats = normalize_summary_stats(summary_stats, m_s, s_s)
            normalized_summary_stats_list.append(normalized_summary_stats)
        return torch.stack(normalized_summary_stats_list)
    return wrapper


def drift(x, y, theta):
    """Drift function for SDE approximation."""
    alpha, beta, delta, gamma = np.exp(theta)
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return np.array([dx, dy])


def diffusion(x, y, sigma):
    """Diffusion function for SDE approximation."""
    sigma1, sigma2 = sigma
    var_x = max((sigma1 * x) ** 2, 1e-6)
    var_y = max((sigma2 * y) ** 2, 1e-6)
    cov = np.array([[var_x, 0], [0, var_y]])
    return cov


def llik_euler(x_path, y_path, theta, sigma, delta_t):
    """Log-likelihood under Euler-Maruyama approximation."""
    log_likelihood = 0.0
    T = len(x_path) - 1
    for t in range(T):
        x_t = x_path[t]
        y_t = y_path[t]
        x_next = x_path[t+1]
        y_next = y_path[t+1]
        mu = np.array([x_t, y_t]) + drift(x_t, y_t, theta) * delta_t
        cov = diffusion(x_t, y_t, sigma) * delta_t
        cov += 1e-6 * np.eye(2)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return -np.inf
        det_cov = np.linalg.det(cov)
        if det_cov <= 0 or not np.isfinite(det_cov):
            return -np.inf
        diff = np.array([x_next, y_next]) - mu
        log_p = -0.5 * np.log(det_cov) - 0.5 * diff.T @ cov_inv @ diff
        log_likelihood += log_p
    return log_likelihood


def mdb_path(x0, y0, x1, y1, theta, sigma, inter, m):
    """Generate modified diffusion bridge path."""
    dt_local = inter / m
    x_path = np.linspace(x0, x1, m+1)
    y_path = np.linspace(y0, y1, m+1)
    
    x_prop = x_path.copy()
    y_prop = y_path.copy()
    
    for i in range(1, m):
        s = i * dt_local
        weight = max((inter - s) / (inter - s + dt_local), 1e-6)
        mu_x = weight * x_prop[i] + x1 * dt_local / (inter - s + dt_local)
        mu_y = weight * y_prop[i] + y1 * dt_local / (inter - s + dt_local)
        cov = weight * diffusion(x_prop[i], y_prop[i], sigma) * dt_local
        cov += 1e-6 * np.eye(2)
        prop_mean = np.array([mu_x, mu_y])
        try:
            sample = multivariate_normal.rvs(mean=prop_mean, cov=cov)
        except np.linalg.LinAlgError:
            return x_prop, y_prop
        x_prop[i+1] = sample[0]
        y_prop[i+1] = sample[1]
    return x_prop, y_prop


def run_mcmc_lv(simulated_data, num_iterations=100000, burn_in=20000):
    """Run MCMC inference for single dataset."""
    num_time_points = simulated_data.shape[1]
    time_points = np.linspace(0, dur, num_time_points)
    
    delta_t_mcmc = 0.05
    t_grid = np.arange(0, dur + delta_t_mcmc, delta_t_mcmc)
    obs_indices = np.searchsorted(t_grid, time_points)
    
    theta_current = np.log([0.8, 0.02, 0.02, 0.5])
    sigma_current = np.array([0.01, 0.01])
    
    X_current = np.interp(t_grid, time_points, simulated_data[0, :])
    Y_current = np.interp(t_grid, time_points, simulated_data[1, :])
    
    theta_prop_std = np.array([0.1, 0.1, 0.1, 0.1])
    sigma_prop_std = np.array([0.005, 0.005])
    
    samples_theta = []
    acceptance_counts = {'theta': 0, 'latent_states': 0}
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        X_proposed = X_current.copy()
        Y_proposed = Y_current.copy()
        log_target_current = 0.0
        log_target_proposed = 0.0
        
        for i in range(len(obs_indices) - 1):
            idx_start = obs_indices[i]
            idx_end = obs_indices[i+1]
            m = idx_end - idx_start
            inter = (idx_end - idx_start) * delta_t_mcmc
            
            X0 = X_current[idx_start]
            X1 = X_current[idx_end]
            Y0 = Y_current[idx_start]
            Y1 = Y_current[idx_end]
            
            x_prop, y_prop = mdb_path(X0, Y0, X1, Y1, theta_current, sigma_current, inter, m)
            
            X_proposed[idx_start:idx_end+1] = x_prop
            Y_proposed[idx_start:idx_end+1] = y_prop
            
            log_target_current += llik_euler(X_current[idx_start:idx_end+1], 
                                            Y_current[idx_start:idx_end+1], 
                                            theta_current, sigma_current, delta_t_mcmc)
            log_target_proposed += llik_euler(X_proposed[idx_start:idx_end+1], 
                                             Y_proposed[idx_start:idx_end+1], 
                                             theta_current, sigma_current, delta_t_mcmc)
        
        log_accept_ratio_latent = log_target_proposed - log_target_current
        
        if np.isfinite(log_accept_ratio_latent) and np.log(np.random.rand()) < log_accept_ratio_latent:
            X_current = X_proposed.copy()
            Y_current = Y_proposed.copy()
            acceptance_counts['latent_states'] += 1
        
        theta_proposal = theta_current + np.random.normal(0, theta_prop_std)
        sigma_proposal = sigma_current + np.random.normal(0, sigma_prop_std)
        
        within_bounds = (
            -5 <= theta_proposal[0] <= 2 and
            -5 <= theta_proposal[1] <= 2 and
            -5 <= theta_proposal[2] <= 2 and
            -5 <= theta_proposal[3] <= 2 and
            0 <= sigma_proposal[0] <= 0.5 and
            0 <= sigma_proposal[1] <= 0.5
        )
        
        if within_bounds:
            log_target_current = llik_euler(X_current, Y_current, theta_current, sigma_current, delta_t_mcmc)
            log_target_proposal = llik_euler(X_current, Y_current, theta_proposal, sigma_proposal, delta_t_mcmc)
            log_accept_ratio_theta = log_target_proposal - log_target_current
            
            if np.isfinite(log_accept_ratio_theta) and np.log(np.random.rand()) < log_accept_ratio_theta:
                theta_current = theta_proposal.copy()
                sigma_current = sigma_proposal.copy()
                acceptance_counts['theta'] += 1
        
        if iteration >= burn_in:
            samples_theta.append(theta_current.copy())
    
    runtime = time.time() - start_time
    samples_theta = np.array(samples_theta)
    
    return samples_theta, runtime


def train_snpe_lv(prior, simulator, x_obs):
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
    
    num_rounds = 10
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x, proposal=proposal).train(**training_params)
        posterior = inference.build_posterior()
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((10000,), x=x_obs)
    
    return samples.numpy(), runtime


def train_snle_lv(prior, simulator, x_obs):
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
    
    num_rounds = 10
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        likelihood_estimator = inference.append_simulations(theta, x).train(**training_params)
        posterior = inference.build_posterior(
            likelihood_estimator,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        posterior = posterior.set_default_x(x_obs)
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((5000,), x=x_obs)
    
    return samples.numpy(), runtime


def compute_nltp_kde(samples, true_value):
    """Compute NLTP using KDE for MCMC samples."""
    kde = gaussian_kde(samples.T)
    pdf_at_true = kde(true_value)
    if pdf_at_true > 0:
        return -np.log(pdf_at_true)
    else:
        return 1e10


def posterior_predictive_lv(theta_samples, n_samples=500):
    """Generate posterior predictive samples."""
    indices = np.random.choice(len(theta_samples), size=n_samples, replace=False)
    
    prey_sims = []
    predator_sims = []
    
    for idx in indices:
        theta_i = torch.tensor(theta_samples[idx])
        sim_data = lotka_volterra_gillespie(theta_i)
        prey_sims.append(sim_data[0, :].numpy())
        predator_sims.append(sim_data[1, :].numpy())
    
    return np.array(prey_sims), np.array(predator_sims)


def plot_ppc_lv(x_obs, x_pred, y_pred, method_name, save_path=None):
    """Plot posterior predictive check."""
    num_time_points = x_obs.shape[1]
    time_points = np.linspace(0, dur, num_time_points)
    
    median_prey = np.median(x_pred, axis=0)
    lower_prey = np.percentile(x_pred, 25, axis=0)
    upper_prey = np.percentile(x_pred, 75, axis=0)
    
    median_predator = np.median(y_pred, axis=0)
    lower_predator = np.percentile(y_pred, 25, axis=0)
    upper_predator = np.percentile(y_pred, 75, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(time_points, x_obs[0, :], color='black', label='Observed Prey', alpha=0.7)
    plt.plot(time_points, x_obs[1, :], color='black', linestyle='--', label='Observed Predator', alpha=0.7)
    
    plt.plot(time_points, median_prey, color='blue', label='Prey Median')
    plt.fill_between(time_points, lower_prey, upper_prey, color='blue', alpha=0.3)
    
    plt.plot(time_points, median_predator, color='orange', label='Predator Median')
    plt.fill_between(time_points, lower_predator, upper_predator, color='orange', alpha=0.3)
    
    plt.xlabel('Time')
    plt.ylabel('Population')
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
    
    mu_s, std_s = gen_summary_stats_mean_and_std(lotka_volterra_gillespie, prior)
    
    prior_processed, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = simulation_wrapper(mu_s, std_s)
    simulator_wrapped = process_simulator(simulator, prior_processed, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapped, prior_processed)
    
    n_realizations = 50
    all_results = []
    
    for real_idx in range(n_realizations):
        print(f"\n--- Realization {real_idx + 1}/{n_realizations} ---")
        
        torch.manual_seed(1000 + real_idx)
        np.random.seed(1000 + real_idx)
        
        simulated_data = lotka_volterra_gillespie(torch.tensor(theta_true))
        
        summary_stats_obs = calculate_summary_statistics(simulated_data)
        normalized_summary_stats_obs = normalize_summary_stats(summary_stats_obs, mu_s, std_s)
        
        print("Running MCMC")
        mcmc_samples, mcmc_time = run_mcmc_lv(simulated_data)
        mcmc_metrics = compute_metrics(mcmc_samples, theta_true)
        mcmc_nltp = compute_nltp_kde(mcmc_samples, theta_true)
        mcmc_coverage = compute_coverage(mcmc_samples, theta_true)
        
        print("Running SNPE-C")
        snpe_samples, snpe_time = train_snpe_lv(prior_processed, simulator_wrapped, normalized_summary_stats_obs)
        snpe_metrics = compute_metrics(snpe_samples, theta_true)
        snpe_coverage = compute_coverage(snpe_samples, theta_true)
        
        print("Running SNLE")
        snle_samples, snle_time = train_snle_lv(prior_processed, simulator_wrapped, normalized_summary_stats_obs)
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
    
    return all_results, mu_s, std_s


if __name__ == "__main__":
    output_dir = "lv_results"
    os.makedirs(output_dir, exist_ok=True)
    
    standard_results, mu_s, std_s = run_standard_inference()
    
    param_names = ['log_alpha', 'log_beta', 'log_delta', 'log_gamma']
    save_results_table(standard_results, f"{output_dir}/standard_results.csv")
    
    last_result = standard_results[-1]
    samples_dict = {
        'MCMC': last_result['MCMC']['samples'],
        'SNPE_C': last_result['SNPE_C']['samples'],
        'SNLE': last_result['SNLE']['samples']
    }
    
    plot_density_comparison(samples_dict, theta_true, 
                           param_names, f"{output_dir}/density_comparison.png")
    
    simulated_data = lotka_volterra_gillespie(torch.tensor(theta_true))
    for method in ['MCMC', 'SNPE_C', 'SNLE']:
        x_pred, y_pred = posterior_predictive_lv(last_result[method]['samples'])
        plot_ppc_lv(simulated_data.numpy(), x_pred, y_pred, method, 
                   f"{output_dir}/{method}_ppc.png")
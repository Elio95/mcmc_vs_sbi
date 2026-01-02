"""
Gaussian Random Walk Case Study
"""

import numpy as np
import torch
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sbi import inference as sbi_inference
from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from scipy.stats import gaussian_kde
import time
import os
import matplotlib.pyplot as plt
from utils import (compute_metrics, compute_coverage, save_results_table,
                   plot_density_comparison, plot_pairplot, plot_traceplot)


# Configuration
d = 10
T = 50
alpha = 0.9
theta_true = torch.ones(d)
x0 = torch.zeros(d)

# Prior
prior_mean = torch.zeros(d)
prior_cov = 10.0 * torch.eye(d)
prior = torch.distributions.MultivariateNormal(prior_mean, prior_cov)


def generate_grw_data(alpha, theta, x0, T):
    """Generate Gaussian Random Walk trajectory."""
    d = theta.shape[0]
    x = torch.zeros((T + 1, d), dtype=torch.float32)
    x[0] = x0.float()
    for t in range(T):
        epsilon = torch.randn(d)
        x[t + 1] = alpha * x[t] + theta + epsilon
    return x[1:]


def grw_numpyro_model(x, x0, alpha, T, d):
    """NumPyro model for GRW."""
    theta = numpyro.sample('theta', dist.MultivariateNormal(
        loc=jnp.zeros(d), covariance_matrix=10 * jnp.eye(d)))
    
    x_prev = x0
    for t in range(T):
        mu = alpha * x_prev + theta
        x_curr = x[t]
        numpyro.sample(f'x_{t}', dist.MultivariateNormal(
            loc=mu, covariance_matrix=jnp.eye(d)), obs=x_curr)
        x_prev = x_curr


def run_mcmc_grw(x_observed, x0, alpha, T, d):
    """Run MCMC inference for single dataset."""
    x_obs_jnp = jnp.array(x_observed.numpy())
    x0_jnp = jnp.array(x0.numpy())
    
    rng_key = random.PRNGKey(np.random.randint(0, 10000))
    nuts_kernel = NUTS(grw_numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=10000, num_chains=4)
    
    start_time = time.time()
    mcmc.run(rng_key, x=x_obs_jnp, x0=x0_jnp, alpha=alpha, T=T, d=d)
    runtime = time.time() - start_time
    
    samples = mcmc.get_samples(group_by_chain=False)
    theta_samples = np.array(samples['theta'])
    
    return theta_samples, runtime


def simulator_grw(theta):
    """Simulator wrapper for SBI."""
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    
    n_sims = theta.shape[0]
    x_sims = torch.zeros((n_sims, T, d))
    
    for i in range(n_sims):
        x_sims[i] = generate_grw_data(alpha, theta[i], x0, T)
    
    return x_sims.reshape(n_sims, -1)


def train_snpe_grw(prior, simulator, x_obs):
    """Train SNPE-C from scratch for single dataset."""
    density_estimator = posterior_nn(
        model='maf',
        hidden_features=30,
        num_transforms=2
    )
    
    training_params = {
        'training_batch_size': 128,
        'learning_rate': 0.0005,
        'max_num_epochs': 1000,
        'stop_after_epochs': 10,
        'show_train_summary': False
    }
    
    inference = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator)
    
    start_time = time.time()
    
    num_rounds = 5
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference.append_simulations(theta, x, proposal=proposal)
        density_est = inference.train(**training_params)
        posterior = inference.build_posterior(density_est)
        posterior = posterior.set_default_x(x_obs.reshape(-1))
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((10000,), x=x_obs.reshape(-1))
    
    return samples.numpy(), runtime


def train_snle_grw(prior, simulator, x_obs):
    """Train SNLE from scratch for single dataset."""
    density_estimator = likelihood_nn(
        model='maf',
        hidden_features=30,
        num_transforms=2
    )
    
    training_params = {
        'training_batch_size': 128,
        'learning_rate': 0.0005,
        'max_num_epochs': 1000,
        'stop_after_epochs': 10,
        'show_train_summary': False
    }
    
    inference = sbi_inference.SNLE(prior=prior, density_estimator=density_estimator)
    
    start_time = time.time()
    
    num_rounds = 5
    num_sims_per_round = 2000
    proposal = prior
    
    for round_idx in range(num_rounds):
        theta = proposal.sample((num_sims_per_round,))
        x = simulator(theta)
        inference.append_simulations(theta, x)
        density_est = inference.train(**training_params)
        posterior = inference.build_posterior(
            density_est,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        posterior = posterior.set_default_x(x_obs.reshape(-1))
        proposal = posterior
    
    runtime = time.time() - start_time
    samples = posterior.sample((10000,), x=x_obs.reshape(-1))
    
    return samples.numpy(), runtime


def compute_nltp_kde(samples, true_value):
    """Compute NLTP using KDE for MCMC samples."""
    kde = gaussian_kde(samples.T)
    pdf_at_true = kde(true_value.numpy())
    if pdf_at_true > 0:
        return -np.log(pdf_at_true)
    else:
        return 1e10


def posterior_predictive_grw(theta_samples, x0, alpha, T, n_samples=500):
    """Generate posterior predictive samples."""
    indices = np.random.choice(len(theta_samples), size=n_samples, replace=False)
    x_pred = np.zeros((n_samples, T, d))
    
    for i, idx in enumerate(indices):
        theta_i = torch.tensor(theta_samples[idx])
        x_pred[i] = generate_grw_data(alpha, theta_i, x0, T).numpy()
    
    return x_pred


def plot_ppc_grw(x_obs, x_pred, method_name, save_path=None):
    """Plot posterior predictive check."""
    median_pred = np.percentile(x_pred, 50, axis=0)
    lower_pred = np.percentile(x_pred, 2.5, axis=0)
    upper_pred = np.percentile(x_pred, 97.5, axis=0)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    axes = axes.flatten()
    
    for k in range(d):
        axes[k].plot(x_obs[:, k], color='black', label='Observed' if k == 0 else None)
        axes[k].plot(median_pred[:, k], color='blue', label='Median' if k == 0 else None)
        axes[k].fill_between(range(T), lower_pred[:, k], upper_pred[:, k],
                            color='blue', alpha=0.25, label='95% CI' if k == 0 else None)
        if k % 5 == 0:
            axes[k].set_ylabel('State')
        if k >= 5:
            axes[k].set_xlabel('Time')
    
    axes[0].legend()
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
        
        seed = 1000 + real_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        x_obs = generate_grw_data(alpha, theta_true, x0, T)
        
        # MCMC
        print("Running MCMC")
        mcmc_samples, mcmc_time = run_mcmc_grw(x_obs, x0, alpha, T, d)
        mcmc_metrics = compute_metrics(mcmc_samples, theta_true.numpy())
        mcmc_nltp = compute_nltp_kde(mcmc_samples, theta_true)
        mcmc_coverage = compute_coverage(mcmc_samples, theta_true.numpy())
        
        # SNPE-C
        print("Running SNPE-C")
        snpe_samples, snpe_time = train_snpe_grw(prior, simulator_grw, x_obs)
        snpe_metrics = compute_metrics(snpe_samples, theta_true.numpy())
        snpe_coverage = compute_coverage(snpe_samples, theta_true.numpy())
        
        # SNLE
        print("Running SNLE")
        snle_samples, snle_time = train_snle_grw(prior, simulator_grw, x_obs)
        snle_metrics = compute_metrics(snle_samples, theta_true.numpy())
        snle_coverage = compute_coverage(snle_samples, theta_true.numpy())
        
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


def train_amortized_networks():
    """Train networks once for amortized inference."""
    print("\n" + "="*80)
    print("AMORTIZED INFERENCE: Training networks once")
    print("="*80)
    
    # Train SNPE-C
    density_estimator_snpe = posterior_nn(
        model='maf',
        hidden_features=30,
        num_transforms=2
    )
    
    training_params = {
        'training_batch_size': 128,
        'learning_rate': 0.0005,
        'max_num_epochs': 1000,
        'stop_after_epochs': 10,
        'show_train_summary': False
    }
    
    inference_snpe = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator_snpe)
    
    start_time_snpe = time.time()
    num_rounds = 10
    num_sims_per_round = 3000
    proposal = prior
    
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}")
        theta = proposal.sample((num_sims_per_round,))
        x = simulator_grw(theta)
        inference_snpe.append_simulations(theta, x, proposal=proposal)
        density_est = inference_snpe.train(**training_params)
        posterior = inference_snpe.build_posterior(density_est)
        dummy_x = simulator_grw(prior.sample((1,)))
        posterior = posterior.set_default_x(dummy_x.reshape(-1))
        proposal = posterior
    
    snpe_training_time = time.time() - start_time_snpe
    print(f"SNPE-C training time: {snpe_training_time:.2f}s")
    
    # Train SNLE
    density_estimator_snle = likelihood_nn(
        model='maf',
        hidden_features=30,
        num_transforms=2
    )
    
    inference_snle = sbi_inference.SNLE(prior=prior, density_estimator=density_estimator_snle)
    
    start_time_snle = time.time()
    proposal = prior
    
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}")
        theta = proposal.sample((num_sims_per_round,))
        x = simulator_grw(theta)
        inference_snle.append_simulations(theta, x)
        density_est = inference_snle.train(**training_params)
        posterior = inference_snle.build_posterior(
            density_est,
            sample_with='mcmc',
            mcmc_method='slice_np_vectorized'
        )
        dummy_x = simulator_grw(prior.sample((1,)))
        posterior = posterior.set_default_x(dummy_x.reshape(-1))
        proposal = posterior
    
    snle_training_time = time.time() - start_time_snle
    print(f"SNLE training time: {snle_training_time:.2f}s")
    
    return posterior, posterior, snpe_training_time, snle_training_time


def run_amortized_inference(posterior_snpe, posterior_snle):
    """Run fast amortized inference on 50 datasets."""
    print("\n" + "="*80)
    print("Running amortized inference on 50 datasets")
    print("="*80)
    
    n_realizations = 50
    snpe_times = []
    snle_times = []
    
    for real_idx in range(n_realizations):
        seed = 1000 + real_idx
        torch.manual_seed(seed)
        x_obs = generate_grw_data(alpha, theta_true, x0, T)
        
        # SNPE-C inference
        start = time.time()
        snpe_samples = posterior_snpe.sample((10000,), x=x_obs.reshape(-1))
        snpe_times.append(time.time() - start)
        
        # SNLE inference
        start = time.time()
        snle_samples = posterior_snle.sample((10000,), x=x_obs.reshape(-1))
        snle_times.append(time.time() - start)
        
        if (real_idx + 1) % 10 == 0:
            print(f"Completed {real_idx + 1}/{n_realizations} datasets")
    
    print(f"\nAverage SNPE-C inference time: {np.mean(snpe_times):.4f}s")
    print(f"Average SNLE inference time: {np.mean(snle_times):.4f}s")
    
    return snpe_times, snle_times


if __name__ == "__main__":
    output_dir = "grw_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard inference
    standard_results = run_standard_inference()
    
    # Save results
    param_names = [f'theta_{i+1}' for i in range(d)]
    save_results_table(standard_results, f"{output_dir}/standard_results.csv")
    
    # Plot results for last realization
    last_result = standard_results[-1]
    samples_dict = {
        'MCMC': last_result['MCMC']['samples'],
        'SNPE_C': last_result['SNPE_C']['samples'],
        'SNLE': last_result['SNLE']['samples']
    }
    
    plot_density_comparison(samples_dict, theta_true.numpy(), 
                           param_names[:4], f"{output_dir}/density_comparison.png")
    
    # PPCs
    x_obs = generate_grw_data(alpha, theta_true, x0, T).numpy()
    for method in ['MCMC', 'SNPE_C', 'SNLE']:
        x_pred = posterior_predictive_grw(last_result[method]['samples'], x0, alpha, T)
        plot_ppc_grw(x_obs, x_pred, method, f"{output_dir}/{method}_ppc.png")
    
    # Amortized inference
    posterior_snpe, posterior_snle, train_time_snpe, train_time_snle = train_amortized_networks()
    snpe_times, snle_times = run_amortized_inference(posterior_snpe, posterior_snle)

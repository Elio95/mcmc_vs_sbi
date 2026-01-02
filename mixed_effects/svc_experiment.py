"""
Simple Variance Components Model 
"""

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
from statsmodels.regression.mixed_linear_model import MixedLM
from sbi.utils.torchutils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from sbi.inference import SNPE_C, SNLE
import sys
sys.path.append('.')
from utils import compute_metrics, save_results_table

np.random.seed(2025)
torch.manual_seed(2025)
sns.set(style="ticks", font_scale=1.1)


def generate_svc_data(I, J, sigma_mu2, sigma_eps2):
    """
    Generate data from simple variance components model.
    
    Args:
        I: number of groups
        J: observations per group
        sigma_mu2: between-group variance
        sigma_eps2: within-group variance
    
    Returns:
        y: observations (I*J,)
        subj_idx: group indicators (I*J,)
    """
    sigma_mu = math.sqrt(sigma_mu2)
    sigma_eps = math.sqrt(sigma_eps2)
    
    mu_i = np.random.normal(0, sigma_mu, size=I)
    subj_idx = np.repeat(np.arange(I), J)
    y = np.random.normal(mu_i[subj_idx], sigma_eps)
    
    return y, subj_idx


def summary_stats(y, subj_idx, n_groups):
    """
    Compute 4D summary statistics for variance components.
    
    Stats: overall variance, mean within-group variance,
           variance of group means, ratio between/within variance
    """
    df = pd.DataFrame({"y": y, "g": subj_idx})
    
    overall_var = y.var(ddof=0)
    
    grouped = df.groupby("g")["y"]
    group_means = grouped.mean().values
    group_vars = grouped.var(ddof=0).fillna(0).values
    
    var_of_means = group_means.var(ddof=0)
    mean_within_var = group_vars.mean()
    ratio_bw_wi = var_of_means / (mean_within_var + 1e-10)
    
    return torch.tensor([overall_var, mean_within_var, var_of_means, ratio_bw_wi],
                       dtype=torch.float32)


def run_mcmc_svc(y_obs, subj_idx, n_groups):
    """Run PyMC MCMC for variance components."""
    with pm.Model() as model:
        sigma_mu2 = pm.Uniform("sigma_mu2", 0, 5)
        sigma_eps2 = pm.Uniform("sigma_eps2", 0, 5)
        
        sigma_mu = pm.Deterministic("sigma_mu", pm.math.sqrt(sigma_mu2))
        sigma_eps = pm.Deterministic("sigma_eps", pm.math.sqrt(sigma_eps2))
        
        mu = pm.Normal("mu", mu=0., sigma=sigma_mu, shape=n_groups)
        y = pm.Normal("y", mu=mu[subj_idx], sigma=sigma_eps, observed=y_obs)
        
        trace = pm.sample(10000, tune=5000, target_accept=0.9, progressbar=False)
    
    post = az.extract(trace, var_names=["sigma_mu2", "sigma_eps2"])
    samples = np.column_stack([post["sigma_mu2"], post["sigma_eps2"]])
    
    return samples


def run_reml_svc(y_obs, subj_idx):
    """Run REML frequentist estimation."""
    df_data = pd.DataFrame({"y": y_obs, "g": subj_idx})
    mod = MixedLM.from_formula("y ~ 1", groups="g", data=df_data)
    res = mod.fit(reml=True)
    
    sigma_mu2_hat = float(res.cov_re.iloc[0, 0])
    sigma_eps2_hat = float(res.scale)
    
    return sigma_mu2_hat, sigma_eps2_hat


class EmbeddingNet(nn.Module):
    """Embedding network for summary statistics."""
    def __init__(self, d_in=4, d_hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.LayerNorm(d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


def build_simulator(I, J, subj_idx):
    """Build simulator function for SBI."""
    def simulator(theta_batch):
        out = []
        for sigma_mu2, sigma_eps2 in theta_batch:
            sigma_mu = float(torch.sqrt(torch.clamp(sigma_mu2, min=1e-12)))
            sigma_eps = float(torch.sqrt(torch.clamp(sigma_eps2, min=1e-12)))
            
            mu = np.random.normal(0, sigma_mu, size=I)
            y = np.random.normal(mu[subj_idx], sigma_eps)
            
            out.append(summary_stats(y, subj_idx, I))
        return torch.stack(out)
    
    return simulator


def run_snpe_svc(prior, simulator, x_obs, num_rounds=10, sims_per_round=1000):
    """Run SNPE-C inference."""
    density_estimator = posterior_nn(
        model="maf",
        num_transforms=10,
        hidden_features=64,
        embedding_net=EmbeddingNet()
    )
    
    snpe = SNPE_C(prior=prior, density_estimator=density_estimator)
    
    train_params = {
        'learning_rate': 5e-4,
        'validation_fraction': 0.10,
        'training_batch_size': 256,
        'stop_after_epochs': 20,
        'show_train_summary': False
    }
    
    proposal = prior
    
    for r in range(num_rounds):
        theta = proposal.sample((sims_per_round,))
        X = simulator(theta)
        X_norm = (X - x_obs.mean(0)) / (x_obs.std(0) + 1e-8)
        
        snpe.append_simulations(theta, X_norm, proposal=proposal)
        snpe.train(**train_params)
        
        posterior = snpe.build_posterior()
        proposal = posterior.set_default_x(x_obs)
    
    return posterior


def run_snle_svc(prior, simulator, x_obs, num_rounds=10, sims_per_round=2000):
    """Run SNLE inference."""
    snle = SNLE(
        prior=prior,
        density_estimator=likelihood_nn(
            model="maf",
            hidden_features=64,
            num_transforms=10
        )
    )
    
    proposal = prior
    
    for r in range(num_rounds):
        theta = proposal.sample((sims_per_round,))
        X = simulator(theta)
        X_norm = (X - x_obs.mean(0)) / (x_obs.std(0) + 1e-8)
        
        snle.append_simulations(theta, X)
        snle.train(
            learning_rate=5e-4,
            validation_fraction=0.1,
            training_batch_size=256,
            stop_after_epochs=20,
            show_train_summary=False
        )
        
        posterior = snle.build_posterior(
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized"
        )
        proposal = posterior.set_default_x(x_obs)
    
    return posterior


def standard_inference_experiment(n_realizations=50, I=50, J=500):
    """
    Standard inference: train from scratch for each dataset.
    """
    print("\n" + "="*70)
    print("STANDARD INFERENCE (Train from scratch per dataset)")
    print("="*70)
    
    true_sigma_mu2 = 1.0
    true_sigma_eps2 = 0.5
    true_params = np.array([true_sigma_mu2, true_sigma_eps2])
    
    prior = BoxUniform(torch.zeros(2), torch.full((2,), 5.0))
    
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        y_obs, subj_idx = generate_svc_data(I, J, true_sigma_mu2, true_sigma_eps2)
        x_obs = summary_stats(y_obs, subj_idx, I)
        
        simulator = build_simulator(I, J, subj_idx)
        
        # Normalize statistics
        ref_sims = simulator(prior.sample((1000,)))
        mu_s = ref_sims.mean(0)
        sigma_s = ref_sims.std(0).clamp_min(1e-8)
        x_obs_norm = (x_obs - mu_s) / sigma_s
        
        # MCMC
        print("Running MCMC")
        start_time = time.time()
        samples_mcmc = run_mcmc_svc(y_obs, subj_idx, I)
        mcmc_time = time.time() - start_time
        
        # SNPE-C
        print("Running SNPE-C")
        start_time = time.time()
        posterior_snpe = run_snpe_svc(prior, simulator, x_obs_norm)
        samples_snpe = posterior_snpe.sample((10000,), x=x_obs_norm).numpy()
        snpe_time = time.time() - start_time
        
        # SNLE
        print("Running SNLE")
        start_time = time.time()
        posterior_snle = run_snle_svc(prior, simulator, x_obs_norm)
        samples_snle = posterior_snle.sample((5000,), x=x_obs_norm).numpy()
        snle_time = time.time() - start_time
        
        # Compute metrics
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
    
    # Save results
    os.makedirs('svc_results', exist_ok=True)
    save_results_table(results_list, 'svc_results/standard_inference_summary.csv')
    
    print("\nStandard inference complete!")
    return results_list


def amortized_inference_experiment(n_realizations=50, I=50, J=500):
    """
    Amortized inference: train once, apply to multiple datasets.
    """
    print("\n" + "="*70)
    print("AMORTIZED INFERENCE (Train once, apply to many datasets)")
    print("="*70)
    
    true_sigma_mu2 = 1.0
    true_sigma_eps2 = 0.5
    true_params = np.array([true_sigma_mu2, true_sigma_eps2])
    
    prior = BoxUniform(torch.zeros(2), torch.full((2,), 5.0))
    
    # Generate dummy data for training
    y_dummy, subj_idx_dummy = generate_svc_data(I, J, true_sigma_mu2, true_sigma_eps2)
    simulator = build_simulator(I, J, subj_idx_dummy)
    
    # Compute normalization
    ref_sims = simulator(prior.sample((1000,)))
    mu_s = ref_sims.mean(0)
    sigma_s = ref_sims.std(0).clamp_min(1e-8)
    
    # Train SNPE-C once
    x_dummy = summary_stats(y_dummy, subj_idx_dummy, I)
    x_dummy_norm = (x_dummy - mu_s) / sigma_s
    
    start_train_snpe = time.time()
    posterior_snpe = run_snpe_svc(prior, simulator, x_dummy_norm, num_rounds=10, sims_per_round=2000)
    snpe_train_time = time.time() - start_train_snpe
    print(f"SNPE-C training time: {snpe_train_time:.2f}s")
    
    # Train SNLE once
    start_train_snle = time.time()
    posterior_snle = run_snle_svc(prior, simulator, x_dummy_norm, num_rounds=10, sims_per_round=2000)
    snle_train_time = time.time() - start_train_snle
    print(f"SNLE training time: {snle_train_time:.2f}s")
    
    # Apply to multiple datasets
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        y_obs, subj_idx = generate_svc_data(I, J, true_sigma_mu2, true_sigma_eps2)
        x_obs = summary_stats(y_obs, subj_idx, I)
        x_obs_norm = (x_obs - mu_s) / sigma_s
        
        # SNPE-C inference (fast)
        start_time = time.time()
        samples_snpe = posterior_snpe.sample((10000,), x=x_obs_norm).numpy()
        snpe_time = time.time() - start_time
        
        # SNLE inference (fast)
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
    
    # Save results
    os.makedirs('svc_results', exist_ok=True)
    save_results_table(results_list, 'svc_results/amortized_inference_summary.csv',
                      methods=['SNPE_C', 'SNLE'])
    
    print("\nAmortized inference complete!")
    print(f"Training time - SNPE-C: {snpe_train_time:.2f}s, SNLE: {snle_train_time:.2f}s")
    
    return results_list


def plot_single_realization(samples_mcmc, samples_snpe, samples_snle, true_params):
    """Plot results for a single realization."""
    param_names = [r"$\sigma_\mu^2$", r"$\sigma_\varepsilon^2$"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'MCMC': 'blue', 'SNPE-C': 'green', 'SNLE': 'red'}
    
    for j, (param, true_val) in enumerate(zip(param_names, true_params)):
        ax = axes[j]
        
        sns.kdeplot(samples_mcmc[:, j], fill=True, alpha=0.4, color=colors['MCMC'],
                   label='MCMC', ax=ax)
        sns.kdeplot(samples_snpe[:, j], fill=True, alpha=0.4, color=colors['SNPE-C'],
                   label='SNPE-C', ax=ax)
        sns.kdeplot(samples_snle[:, j], fill=True, alpha=0.4, color=colors['SNLE'],
                   label='SNLE', ax=ax)
        
        ax.axvline(true_val, color='black', linestyle='--', linewidth=2, label='True')
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('svc_results/posterior_comparison.png', dpi=300)
    plt.close()


def main():
    print("Simple Variance Components Model")
    print("="*70)
    
    # Standard inference (train from scratch per dataset)
    results_standard = standard_inference_experiment(n_realizations=50, I=50, J=500)
    
    # Amortized inference (train once, apply many times)
    results_amortized = amortized_inference_experiment(n_realizations=50, I=50, J=500)
    
    # Plot example from standard inference
    first_result = results_standard[0]
    plot_single_realization(
        first_result['MCMC']['samples'],
        first_result['SNPE_C']['samples'],
        first_result['SNLE']['samples'],
        np.array([1.0, 0.5])
    )

if __name__ == "__main__":
    main()

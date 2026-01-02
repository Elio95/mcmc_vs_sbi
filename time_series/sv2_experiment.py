"""
Stochastic Volatility Model with Regressors (SV2): 5D joint inference (mu, phi, log_sigma, beta0, beta1)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as scipy_norm
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LinearRegression
from sbi.inference import SNPE_C, SNLE
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
import time
import os
import math


def simulator_sv2_full(params, x_mat):
    """
    Simulate returns from SV2 model with regressors.
    
    Args:
        params: (mu, phi, log_sigma, beta0, beta1)
        x_mat: design matrix [T x 2]
    
    Returns:
        returns: simulated observations
    """
    mu, phi, log_sigma, beta0, beta1 = params
    sigma = torch.exp(log_sigma)
    T = x_mat.shape[0]
    
    h = torch.zeros(T)
    h[0] = mu
    
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sigma * torch.randn(1)
    
    beta = torch.tensor([beta0, beta1])
    mean_returns = x_mat @ beta
    returns = mean_returns + torch.exp(h / 2.0) * torch.randn(T)
    
    return returns


class FullPriorSV2(torch.distributions.Distribution):
    """
    5D prior: (mu, phi, log_sigma, beta0, beta1)
    Matched to R MCMC implementation with Beta(5, 1.5) for phi.
    """
    def __init__(self, a_phi=5.0, b_phi=1.5, validate_args=False):
        super().__init__(validate_args=validate_args)
        self.a_phi = a_phi
        self.b_phi = b_phi
        
        self.mu_dist = Normal(0, 10)
        self.phi_dist = Beta(a_phi, b_phi)
        self.logsig_dist = Normal(torch.tensor(np.log(0.3)), torch.tensor(0.2))
        self.beta0_dist = Normal(0, 1)
        self.beta1_dist = Normal(0, 1)

    def sample(self, sample_shape=torch.Size()):
        mu = self.mu_dist.sample(sample_shape)
        phi_raw = self.phi_dist.sample(sample_shape)
        phi = 2.0 * phi_raw - 1.0
        log_sig = self.logsig_dist.sample(sample_shape)
        beta0 = self.beta0_dist.sample(sample_shape)
        beta1 = self.beta1_dist.sample(sample_shape)
        
        return torch.stack([mu, phi, log_sig, beta0, beta1], dim=-1)

    def log_prob(self, value):
        mu = value[..., 0]
        phi = value[..., 1]
        log_sig = value[..., 2]
        beta0 = value[..., 3]
        beta1 = value[..., 4]

        lp_mu = self.mu_dist.log_prob(mu)
        
        inside_mask = (phi > -1.0) & (phi < 1.0)
        phi_raw = 0.5 * (phi + 1.0)
        
        lp_phi = torch.full_like(mu, -1e15)
        valid_inds = inside_mask.nonzero(as_tuple=True)
        if len(valid_inds[0]) > 0:
            phi_raw_valid = phi_raw[valid_inds]
            inside_phi_raw = (phi_raw_valid > 0.0) & (phi_raw_valid < 1.0)
            if inside_phi_raw.any():
                valid2_inds = inside_phi_raw.nonzero(as_tuple=True)
                pr_2 = phi_raw_valid[valid2_inds]
                logp_phiraw = self.phi_dist.log_prob(pr_2) + math.log(0.5)
                idx_all = (valid_inds[0][valid2_inds],)
                lp_phi[idx_all] = logp_phiraw

        lp_logsig = self.logsig_dist.log_prob(log_sig)
        lp_beta0 = self.beta0_dist.log_prob(beta0)
        lp_beta1 = self.beta1_dist.log_prob(beta1)

        total_lp = lp_mu + lp_phi + lp_logsig + lp_beta0 + lp_beta1
        total_lp = torch.where(inside_mask, total_lp, torch.full_like(total_lp, -1e15))

        return total_lp


def calculate_enhanced_summary_statistics(returns, x_mat):
    """
    Enhanced summary statistics with OLS estimates and volatility features.
    """
    # Full sample OLS
    model_ols = LinearRegression(fit_intercept=False)
    model_ols.fit(x_mat, returns)
    beta_ols = model_ols.coef_
    
    residuals = returns - x_mat @ beta_ols
    n = len(returns)
    residual_var = np.sum(residuals**2) / (n - 2)
    XtX_inv = np.linalg.inv(x_mat.T @ x_mat)
    cov_beta = residual_var * XtX_inv
    se_beta0 = np.sqrt(cov_beta[0, 0])
    se_beta1 = np.sqrt(cov_beta[1, 1])
    
    ols_full = np.array([beta_ols[0], beta_ols[1], se_beta0, se_beta1])
    
    # Subsample OLS by volatility
    abs_residuals = np.abs(residuals)
    median_abs_resid = np.median(abs_residuals)
    
    low_vol_mask = abs_residuals <= median_abs_resid
    if low_vol_mask.sum() > 10:
        model_low = LinearRegression(fit_intercept=False)
        model_low.fit(x_mat[low_vol_mask], returns[low_vol_mask])
        beta0_low, beta1_low = model_low.coef_
    else:
        beta0_low, beta1_low = beta_ols[0], beta_ols[1]
    
    high_vol_mask = abs_residuals > median_abs_resid
    if high_vol_mask.sum() > 10:
        model_high = LinearRegression(fit_intercept=False)
        model_high.fit(x_mat[high_vol_mask], returns[high_vol_mask])
        beta0_high, beta1_high = model_high.coef_
    else:
        beta0_high, beta1_high = beta_ols[0], beta_ols[1]
    
    ols_subsample = np.array([beta0_low, beta1_low, beta0_high, beta1_high])
    
    # Rolling window OLS
    window_size = max(200, n // 10)
    beta0_rolling = []
    beta1_rolling = []
    
    for i in range(0, n - window_size + 1, window_size // 2):
        window_x = x_mat[i:i+window_size]
        window_y = returns[i:i+window_size]
        try:
            model_window = LinearRegression(fit_intercept=False)
            model_window.fit(window_x, window_y)
            beta0_rolling.append(model_window.coef_[0])
            beta1_rolling.append(model_window.coef_[1])
        except:
            pass
    
    if len(beta0_rolling) > 0:
        ols_rolling = np.array([
            np.mean(beta0_rolling), np.std(beta0_rolling),
            np.mean(beta1_rolling), np.std(beta1_rolling)
        ])
    else:
        ols_rolling = np.array([beta_ols[0], 0.0, beta_ols[1], 0.0])
    
    # Residual statistics
    resid_stats = []
    resid_stats.append(np.log(np.var(residuals) + 1e-8))
    resid_stats.append(skew(residuals))
    resid_stats.append(kurtosis(residuals, fisher=True))
    
    try:
        acf_abs = acf(abs_residuals, nlags=5, fft=True)
        resid_stats.extend(acf_abs[1:6])
    except:
        resid_stats.extend([0.0] * 5)
    
    try:
        acf_sq = acf(residuals**2, nlags=3, fft=True)
        resid_stats.extend(acf_sq[1:4])
    except:
        resid_stats.extend([0.0] * 3)
    
    resid_stats.append(np.percentile(residuals, 5))
    resid_stats.append(np.percentile(residuals, 95))
    resid_stats = np.array(resid_stats)
    
    # Return dynamics
    return_dynamics = []
    try:
        acf_ret = acf(returns, nlags=8, fft=True)
        return_dynamics.extend(acf_ret[1:9])
    except:
        return_dynamics.extend([0.0] * 8)
    
    return_dynamics.append(np.mean(returns))
    return_dynamics.append(np.std(returns))
    
    if n > 10:
        lag_abs_ret = np.concatenate(([0], abs_residuals[:-1]))
        try:
            cross_corr = np.corrcoef(returns, lag_abs_ret)[0, 1]
        except:
            cross_corr = 0.0
    else:
        cross_corr = 0.0
    return_dynamics.append(cross_corr)
    
    if low_vol_mask.sum() > 0 and high_vol_mask.sum() > 0:
        mean_ret_low = np.mean(np.abs(returns[low_vol_mask]))
        mean_ret_high = np.mean(np.abs(returns[high_vol_mask]))
        vol_ratio = mean_ret_high / (mean_ret_low + 1e-8)
    else:
        vol_ratio = 1.0
    return_dynamics.append(vol_ratio)
    
    std_ret = np.std(returns)
    pct_tails = np.mean(np.abs(returns) > 2 * std_ret)
    return_dynamics.append(pct_tails)
    
    cumret = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumret)
    drawdown = running_max - cumret
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    return_dynamics.append(max_drawdown)
    return_dynamics = np.array(return_dynamics)
    
    # Weighted regressions
    weighted_stats = []
    
    weights_inv_abs = 1.0 / (abs_residuals + 1e-6)
    weights_inv_abs = weights_inv_abs / np.sum(weights_inv_abs) * len(weights_inv_abs)
    weights_inv_abs = np.clip(weights_inv_abs, 0.01, 100)
    
    try:
        model_w1 = LinearRegression(fit_intercept=False)
        model_w1.fit(x_mat, returns, sample_weight=weights_inv_abs)
        beta0_w1, beta1_w1 = model_w1.coef_
    except:
        beta0_w1, beta1_w1 = beta_ols[0], beta_ols[1]
    weighted_stats.extend([beta0_w1, beta1_w1])
    
    weights_inv_sq = 1.0 / (residuals**2 + 1e-6)
    weights_inv_sq = weights_inv_sq / np.sum(weights_inv_sq) * len(weights_inv_sq)
    weights_inv_sq = np.clip(weights_inv_sq, 0.01, 100)
    
    try:
        model_w2 = LinearRegression(fit_intercept=False)
        model_w2.fit(x_mat, returns, sample_weight=weights_inv_sq)
        beta0_w2, beta1_w2 = model_w2.coef_
    except:
        beta0_w2, beta1_w2 = beta_ols[0], beta_ols[1]
    weighted_stats.extend([beta0_w2, beta1_w2])
    
    log_resid_sq = np.log(residuals**2 + 1e-8)
    h_approx = np.zeros_like(log_resid_sq)
    h_approx[0] = log_resid_sq[0]
    alpha = 0.9
    for t in range(1, len(log_resid_sq)):
        h_approx[t] = alpha * h_approx[t-1] + (1-alpha) * log_resid_sq[t]
    
    weights_exp = np.exp(-h_approx * 0.5)
    weights_exp = weights_exp / np.sum(weights_exp) * len(weights_exp)
    weights_exp = np.clip(weights_exp, 0.01, 100)
    
    try:
        model_w3 = LinearRegression(fit_intercept=False)
        model_w3.fit(x_mat, returns, sample_weight=weights_exp)
        beta0_w3, beta1_w3 = model_w3.coef_
    except:
        beta0_w3, beta1_w3 = beta_ols[0], beta_ols[1]
    weighted_stats.extend([beta0_w3, beta1_w3])
    
    diff_01 = beta0_w1 - beta_ols[0]
    diff_11 = beta1_w1 - beta_ols[1]
    weighted_stats.extend([diff_01, diff_11])
    weighted_stats = np.array(weighted_stats)
    
    all_stats = np.concatenate([
        ols_full, ols_subsample, ols_rolling,
        resid_stats, return_dynamics, weighted_stats
    ])
    
    all_stats = np.nan_to_num(all_stats, nan=0.0, posinf=10.0, neginf=-10.0)
    return torch.from_numpy(all_stats.astype(np.float32))


def simulator_with_stats(params, x_mat):
    returns_sim = simulator_sv2_full(params, x_mat)
    
    if torch.isnan(returns_sim).any() or torch.isinf(returns_sim).any():
        return torch.zeros(48)
    
    stats = calculate_enhanced_summary_statistics(returns_sim.numpy(), x_mat)
    return stats


def compute_normalization(prior, x_mat, num_sims=5000):
    """Compute normalization statistics from prior simulations."""
    print(f"Computing normalization from {num_sims} prior simulations")
    
    stats_list = []
    for i in range(num_sims):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{num_sims}")
        
        params = prior.sample((1,)).squeeze(0)
        stats = simulator_with_stats(params, x_mat)
        
        if torch.isnan(stats).any() or torch.isinf(stats).any():
            continue
        if stats.abs().sum() < 0.01:
            continue
        
        stats_list.append(stats.numpy())
    
    arr = np.array(stats_list)
    mean_stats = arr.mean(axis=0)
    std_stats = arr.std(axis=0)
    std_stats[std_stats < 1e-8] = 1.0
    
    print(f"Computed normalization from {len(stats_list)} valid simulations")
    return mean_stats, std_stats


def build_normalized_simulator(x_mat, mean_stats, std_stats):
    """Build normalized simulator."""
    def simulator_batch(params_batch):
        if params_batch.dtype != torch.float32:
            params_batch = params_batch.float()

        out = []
        for i in range(params_batch.shape[0]):
            stats = simulator_with_stats(params_batch[i], x_mat)
            
            if torch.isnan(stats).any() or torch.isinf(stats).any():
                out.append(torch.zeros(len(mean_stats)))
                continue
            
            if stats.abs().sum() < 0.01:
                out.append(torch.zeros(len(mean_stats)))
                continue
            
            stats_norm = (stats - torch.from_numpy(mean_stats)) / torch.from_numpy(std_stats)
            
            if torch.isnan(stats_norm).any() or torch.isinf(stats_norm).any():
                out.append(torch.zeros(len(mean_stats)))
                continue
            
            out.append(stats_norm)
        
        return torch.stack(out, dim=0)
    
    return simulator_batch


class EnhancedEmbeddingNet(nn.Module):
    """
    Embedding network for summary statistics.
    """
    def __init__(self, input_dim=48, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class ParameterEmbeddingNet5D(nn.Module):
    """Embedding network for 5D parameters (for SNLE)."""
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
    
    def forward(self, theta):
        if theta.dtype != torch.float32:
            theta = theta.float()
        return self.net(theta)


def run_snpe(prior, simulator, s_obs_norm, num_rounds=10, sims_per_round=5000):
    print("\n" + "="*80)
    print("Running SNPE-C")
    print("="*80)
    
    embedding_net = EnhancedEmbeddingNet(input_dim=48, hidden_dim=128)
    
    density_estimator = posterior_nn(
        model="maf",
        hidden_features=64,
        num_transforms=10,
        embedding_net=embedding_net,
        z_score_theta="independent",
        z_score_x="independent"
    )
    
    snpe = SNPE_C(prior=prior, density_estimator=density_estimator)
    
    training_params = {
        'training_batch_size': 256,
        'learning_rate': 1e-4,
        'validation_fraction': 0.1,
        'stop_after_epochs': 20,
        'max_num_epochs': 600,
        'clip_max_norm': 5.0,
        'show_train_summary': True
    }
    
    proposal = prior
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        
        theta = proposal.sample((sims_per_round,)).float()
        x = simulator(theta)
        
        valid_mask = (
            torch.isfinite(x).all(dim=1) &
            (x.abs().sum(dim=1) > 0.01) &
            (x.abs().max(dim=1)[0] < 100)
        )
        
        theta_valid = theta[valid_mask]
        x_valid = x[valid_mask]
        
        print(f"  Valid simulations: {len(theta_valid)}/{sims_per_round}")
        
        if len(theta_valid) < 2000:
            print("  Resampling to get 2000 valid simulations")
            attempts = 0
            while len(theta_valid) < 2000 and attempts < 5:
                theta_extra = proposal.sample((sims_per_round,)).float()
                x_extra = simulator(theta_extra)
                
                valid_mask_extra = (
                    torch.isfinite(x_extra).all(dim=1) &
                    (x_extra.abs().sum(dim=1) > 0.01) &
                    (x_extra.abs().max(dim=1)[0] < 100)
                )
                
                theta_valid = torch.cat([theta_valid, theta_extra[valid_mask_extra]])
                x_valid = torch.cat([x_valid, x_extra[valid_mask_extra]])
                attempts += 1
        
        density_est = snpe.append_simulations(
            theta_valid, x_valid, proposal=proposal
        ).train(**training_params)
        
        posterior = snpe.build_posterior(density_est).set_default_x(s_obs_norm)
        proposal = posterior
    
    runtime = time.time() - start_time
    print(f"\nSNPE-C complete in {runtime:.1f}s ({runtime/60:.1f} min)")
    
    return posterior, runtime


def run_snle(prior, simulator, s_obs_norm, num_rounds=10, sims_per_round=5000):

    print("\n" + "="*80)
    print("Running SNLE")
    print("="*80)
    
    embedding_net_theta = ParameterEmbeddingNet5D(input_dim=5, hidden_dim=128)
    embedding_net_x = EnhancedEmbeddingNet(input_dim=48, hidden_dim=128)
    
    density_estimator = likelihood_nn(
        model="maf",
        hidden_features=64,
        num_transforms=10,
        embedding_net_theta=embedding_net_theta,
        embedding_net_x=embedding_net_x,
        z_score_theta="independent",
        z_score_x="independent"
    )
    
    snle = SNLE(prior=prior, density_estimator=density_estimator)
    
    training_params = {
        'training_batch_size': 256,
        'learning_rate': 1e-4,
        'validation_fraction': 0.1,
        'stop_after_epochs': 20,
        'max_num_epochs': 600,
        'clip_max_norm': 5.0,
        'show_train_summary': True
    }
    
    proposal = prior
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        
        theta = proposal.sample((sims_per_round,)).float()
        x = simulator(theta)
        
        valid_mask = (
            torch.isfinite(x).all(dim=1) &
            (x.abs().sum(dim=1) > 0.01) &
            (x.abs().max(dim=1)[0] < 100)
        )
        
        theta_valid = theta[valid_mask]
        x_valid = x[valid_mask]
        
        print(f"  Valid simulations: {len(theta_valid)}/{sims_per_round}")
        
        if len(theta_valid) < 2000:
            print("  Resampling to get 2000 valid simulations")
            attempts = 0
            while len(theta_valid) < 2000 and attempts < 5:
                theta_extra = proposal.sample((sims_per_round,)).float()
                x_extra = simulator(theta_extra)
                
                valid_mask_extra = (
                    torch.isfinite(x_extra).all(dim=1) &
                    (x_extra.abs().sum(dim=1) > 0.01) &
                    (x_extra.abs().max(dim=1)[0] < 100)
                )
                
                theta_valid = torch.cat([theta_valid, theta_extra[valid_mask_extra]])
                x_valid = torch.cat([x_valid, x_extra[valid_mask_extra]])
                attempts += 1
        
        likelihood_est = snle.append_simulations(
            theta_valid, x_valid, proposal=proposal
        ).train(**training_params)
        
        posterior = snle.build_posterior(
            likelihood_est,
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={
                'num_chains': 20,
                'thin': 5,
                'warmup_steps': 150,
                'init_strategy': 'proposal'
            }
        ).set_default_x(s_obs_norm)
        
        proposal = posterior
    
    runtime = time.time() - start_time
    print(f"\nSNLE complete in {runtime:.1f}s ({runtime/60:.1f} min)")
    
    return posterior, runtime


def plot_comparison(samples_snpe, samples_snle, mcmc_means, mcmc_stds, save_path=None):
    """Plot posterior comparison."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    param_labels = [r'$\mu$', r'$\phi$', r'$\log(\sigma)$', r'$\beta_0$', r'$\beta_1$']
    
    for i, (ax, label, mcmc_mean, mcmc_std) in enumerate(
        zip(axes, param_labels, mcmc_means, mcmc_stds)
    ):
        ax.hist(samples_snle[:, i], bins=50, density=True, alpha=0.5,
                color='red', label='SNLE', edgecolor='darkred', linewidth=0.5)
        
        ax.hist(samples_snpe[:, i], bins=50, density=True, alpha=0.5,
                color='green', label='SNPE-C', edgecolor='darkgreen', linewidth=0.5)
        
        x_range = np.linspace(
            min(samples_snle[:, i].min(), samples_snpe[:, i].min()),
            max(samples_snle[:, i].max(), samples_snpe[:, i].max()),
            100
        )
        ax.plot(x_range, scipy_norm.pdf(x_range, mcmc_mean, mcmc_std),
                'b-', linewidth=2.5, label='MCMC', alpha=0.8)
        
        ax.axvline(samples_snle[:, i].mean(), color='red', linestyle='--', 
                   linewidth=2, alpha=0.7)
        ax.axvline(samples_snpe[:, i].mean(), color='green', linestyle='--',
                   linewidth=2, alpha=0.7)
        ax.axvline(mcmc_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(label, fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('5D Joint Inference: SNLE vs SNPE-C vs MCMC',
                 fontsize=15, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_results(samples_snpe, samples_snle, mcmc_means, mcmc_stds):
    param_names = ['mu', 'phi', 'log(sigma)', 'beta_0', 'beta_1']
    
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)
    
    print(f"\n{'Parameter':<12} {'SNLE Mean':>12} {'SNLE Std':>12} " +
          f"{'SNPE-C Mean':>12} {'SNPE-C Std':>12} {'MCMC Mean':>12} {'MCMC Std':>12}")
    print("-"*110)
    
    for i, (param, mcmc_mean, mcmc_std) in enumerate(
        zip(param_names, mcmc_means, mcmc_stds)
    ):
        snle_mean = samples_snle[:, i].mean()
        snle_std = samples_snle[:, i].std()
        snpe_mean = samples_snpe[:, i].mean()
        snpe_std = samples_snpe[:, i].std()
        
        print(f"{param:<12} {snle_mean:>12.6f} {snle_std:>12.6f} " +
              f"{snpe_mean:>12.6f} {snpe_std:>12.6f} " +
              f"{mcmc_mean:>12.6f} {mcmc_std:>12.6f}")
    
    print("\n" + "="*80)
    print("Error Analysis (% error vs MCMC)")
    print("="*80)
    
    print(f"\n{'Parameter':<12} {'SNLE Error':>15} {'SNPE-C Error':>15} {'Winner':>12}")
    print("-"*56)
    
    for i, (param, mcmc_mean) in enumerate(zip(param_names, mcmc_means)):
        snle_error = abs(samples_snle[:, i].mean() - mcmc_mean) / abs(mcmc_mean) * 100
        snpe_error = abs(samples_snpe[:, i].mean() - mcmc_mean) / abs(mcmc_mean) * 100
        
        winner = "SNLE" if snle_error < snpe_error else ("SNPE-C" if snpe_error < snle_error else "Tie")
        
        print(f"{param:<12} {snle_error:>14.1f}% {snpe_error:>14.1f}% {winner:>12}")
    
    snle_mae = np.mean([abs(samples_snle[:, i].mean() - mcmc_means[i]) / abs(mcmc_means[i]) * 100
                        for i in range(5)])
    snpe_mae = np.mean([abs(samples_snpe[:, i].mean() - mcmc_means[i]) / abs(mcmc_means[i]) * 100
                        for i in range(5)])
    
    print(f"\nMean Absolute Error:")
    print(f"  SNLE:   {snle_mae:.1f}%")
    print(f"  SNPE-C: {snpe_mae:.1f}%")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load observed data (from R MCMC run)
    # Replace with your actual data loading
    T = 1500
    y_obs = torch.randn(T) * 0.3  # Placeholder
    x_mat = np.column_stack([np.ones(T), np.random.randn(T)])  # Placeholder
    
    print("SV2 Experiment: 5D Joint Inference with OLS-Enhanced Statistics")
    print(f"Sample size: {T}")
    print(f"Parameters: (mu, phi, log_sigma, beta0, beta1)")
    
    # Initialize prior
    prior = FullPriorSV2(a_phi=5.0, b_phi=1.5)
    
    # Compute normalization
    mean_stats, std_stats = compute_normalization(prior, x_mat, num_sims=5000)
    
    # Build normalized simulator
    simulator = build_normalized_simulator(x_mat, mean_stats, std_stats)
    
    # Normalize observed statistics
    s_obs = calculate_enhanced_summary_statistics(y_obs.numpy(), x_mat)
    s_obs_norm = (s_obs - torch.from_numpy(mean_stats)) / torch.from_numpy(std_stats)
    
    
    # Run SNPE-C
    posterior_snpe, runtime_snpe = run_snpe(prior, simulator, s_obs_norm)
    
    # Run SNLE
    posterior_snle, runtime_snle = run_snle(prior, simulator, s_obs_norm)
    
    # Sample posteriors
    print("\nSampling from posteriors")
    samples_snpe = posterior_snpe.sample((10000,), x=s_obs_norm).cpu().numpy()
    samples_snle = posterior_snle.sample((10000,), x=s_obs_norm, 
                                         show_progress_bars=True).cpu().numpy()
    
    # MCMC reference (from R implementation)
    mcmc_means = [-9.6574, 0.9721, -1.2649, 0.0009, -0.0621]
    mcmc_stds = [0.2008, 0.0061, 0.0859, 0.0001, 0.0192]
    
    # Print results
    print_results(samples_snpe, samples_snle, mcmc_means, mcmc_stds)
    
    print(f"\nRuntimes:")
    print(f"  SNPE-C: {runtime_snpe:.1f}s ({runtime_snpe/60:.1f} min)")
    print(f"  SNLE:   {runtime_snle:.1f}s ({runtime_snle/60:.1f} min)")
    
    # Plot comparison
    os.makedirs('sv2_results', exist_ok=True)
    plot_comparison(samples_snpe, samples_snle, mcmc_means, mcmc_stds,
                   save_path='sv2_results/posterior_comparison.png')

if __name__ == "__main__":
    main()

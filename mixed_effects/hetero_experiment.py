"""
Heteroscedastic Regression Model
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
from torch.distributions import constraints
from sbi.utils.torchutils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from sbi.inference import SNPE_C, SNLE
import sys
sys.path.append('.')
from utils import compute_metrics, save_results_table

np.random.seed(2025)
torch.manual_seed(2025)
sns.set(style="ticks", font_scale=1.1)

N_SUBJ = 50
T_TIME = 12
X_MIN, X_MAX = 0., 10.

BETA_TRUE = {'beta0': 1.5, 'beta1': 0.6, 'beta2': 0.05}
GAMMA0_TRUE = math.log(1.0)
B_TRUE = 0.25
SIGMA_U_TRUE2 = 1.0
TAU_TRUE2 = 0.5

subject_ids = np.repeat(np.arange(N_SUBJ), T_TIME)
time_vec = np.tile(np.arange(T_TIME), N_SUBJ)
x_vec = np.random.uniform(X_MIN, X_MAX, size=N_SUBJ*T_TIME)

LOG_SIG_MIN, LOG_SIG_MAX = -3.0, 3.0


def simulate_y(theta, clamp_log_sigma=True):
    """Simulate heteroscedastic panel data."""
    β0, β1, β2, sig_u2, tau2 = [float(v) for v in theta]
    sig_u = math.sqrt(max(1e-12, sig_u2))
    tau = math.sqrt(max(1e-12, tau2))
    
    u = np.random.normal(0, sig_u, N_SUBJ)
    a = np.random.normal(0, tau, T_TIME)
    
    mu = β0 + u[subject_ids] + β1 * x_vec + β2 * time_vec
    log_sigma = GAMMA0_TRUE + a[time_vec] + B_TRUE * x_vec
    
    if clamp_log_sigma:
        log_sigma = np.clip(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
    
    y = np.random.normal(mu, np.exp(log_sigma))
    return y


def safe_var(a):
    v = float(np.var(a))
    return 0.0 if not np.isfinite(v) else v


def safe_std(a):
    s = float(np.std(a))
    return 0.0 if not np.isfinite(s) else s


def safe_mean(a):
    m = float(np.mean(a))
    return 0.0 if not np.isfinite(m) else m


def safe_ratio(num, den):
    return float(num/(den+1e-12))


def safe_r2(y, yhat):
    vy = np.var(y)
    if not np.isfinite(vy) or vy < 1e-12:
        return 0.0
    ve = np.var(y - yhat)
    return float(1.0 - ve/(vy+1e-12))


def safe_lstsq(X, y):
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        if not np.all(np.isfinite(b)):
            b = np.zeros(X.shape[1])
    except Exception:
        b = np.zeros(X.shape[1])
    return b


def _safe_corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def _wls(X, y, w):
    """Weighted least squares."""
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Xw = X * sw[:, None]
    yw = y * sw
    bh = safe_lstsq(Xw, yw)
    yhat = X @ bh
    r = y - yhat
    mad = np.median(np.abs(r - np.median(r))) + 1e-12
    denom = np.array([(X[:, i]**2 * w).sum() for i in range(X.shape[1])]) + 1e-12
    se_proxy = mad / np.sqrt(denom)
    r2 = safe_r2(y, yhat)
    return bh, se_proxy, yhat, r, r2


def targeted_summaries(y, x, subj, t):
    """FGLS-enhanced 54-dimensional summary statistics."""
    df = pd.DataFrame({"y": y, "x": x, "subj": subj, "time": t})
    
    X = np.column_stack((np.ones_like(x), x, t))
    beta_hat = safe_lstsq(X, y)
    yhat = X @ beta_hat
    r = y - yhat
    mad = np.median(np.abs(r - np.median(r))) + 1e-12
    denom = np.array([(X[:, i]**2).sum() for i in range(X.shape[1])]) + 1e-12
    se_beta_proxy = mad / np.sqrt(denom)
    
    z = np.log(np.abs(r) + 1e-12)
    time_dum = np.eye(T_TIME)[t]
    Zs = np.column_stack((np.ones_like(x), x, time_dum[:, 1:]))
    gscale = safe_lstsq(Zs, z)
    zhat = Zs @ gscale
    sigma_hat = np.exp(zhat)
    w = 1.0 / (sigma_hat**2 + 1e-12)
    
    sc_intercept = float(gscale[0])
    sc_slope_x = float(gscale[1])
    sc_time_mean = float(np.mean(gscale[2:])) if gscale.size > 2 else 0.0
    sc_time_sd = float(np.std(gscale[2:])) if gscale.size > 2 else 0.0
    R2_z_tfe = safe_r2(z, zhat)
    
    beta_wls, se_beta_wls, yhat_w, r_w, R2_wls = _wls(X, y, w)
    
    gsub = df.groupby("subj")
    y_dm = df.y - gsub.y.transform("mean")
    x_dm = df.x - gsub.x.transform("mean")
    t_dm = df.time - gsub.time.transform("mean")
    Xw_within = np.column_stack((x_dm.values, t_dm.values))
    bwls, se_w_within, _, _, R2_wls_within = _wls(Xw_within, y_dm.values, w)
    
    def _wls_R2_drop(col_to_drop):
        cols = [0, 1, 2]
        cols.remove(col_to_drop)
        Xd = X[:, cols]
        _, _, yhat_d, _, R2d = _wls(Xd, y, w)
        return R2d
    
    R2_wls_drop_x = _wls_R2_drop(1)
    R2_wls_drop_t = _wls_R2_drop(2)
    dR2_x = float(max(0.0, R2_wls - R2_wls_drop_x))
    dR2_t = float(max(0.0, R2_wls - R2_wls_drop_t))
    
    r_tilde = r / (sigma_hat + 1e-12)
    
    mean_by_subj = gsub.y.mean().values
    between_subj_var = safe_var(mean_by_subj)
    within_var_by_subj = gsub.y.var(ddof=0).fillna(0).values
    mean_within_var = safe_mean(within_var_by_subj)
    ratio_bw_wi = safe_ratio(between_subj_var, mean_within_var)
    overall_var = safe_var(y)
    icc_proxy = safe_ratio(between_subj_var, overall_var)
    
    res_sd_by_subj = df.assign(r=r).groupby("subj").r.std(ddof=0).fillna(0).values
    res_tilde_sd_by_subj = df.assign(rt=r_tilde).groupby("subj").rt.std(ddof=0).fillna(0).values
    mean_res_sd_subj = safe_mean(res_sd_by_subj)
    sd_res_sd_subj = safe_std(res_sd_by_subj)
    mean_res_tsd_subj = safe_mean(res_tilde_sd_by_subj)
    sd_res_tsd_subj = safe_std(res_tilde_sd_by_subj)
    
    slopes_zx_by_t = []
    for tt in np.unique(t):
        mask = (t == tt)
        if mask.sum() >= 3 and np.std(x[mask]) > 1e-10:
            gt = safe_lstsq(np.column_stack((np.ones(mask.sum()), x[mask])), z[mask])
            slopes_zx_by_t.append(float(gt[1]))
        else:
            slopes_zx_by_t.append(0.0)
    
    slopes_zx_by_t = np.asarray(slopes_zx_by_t, float)
    mean_slope_zx_t = safe_mean(slopes_zx_by_t)
    sd_slope_zx_t = safe_std(slopes_zx_by_t)
    
    res_var_by_time = df.assign(r=r).groupby("time").r.var(ddof=0).fillna(0).values
    res_tilde_var_by_t = df.assign(rt=r_tilde).groupby("time").rt.var(ddof=0).fillna(0).values
    mean_res_var_time = safe_mean(res_var_by_time)
    sd_res_var_time = safe_std(res_var_by_time)
    mean_res_tvar_time = safe_mean(res_tilde_var_by_t)
    sd_res_tvar_time = safe_std(res_tilde_var_by_t)
    
    def _logvar_contrast(arr, key):
        q25, q75 = np.percentile(df[key], [25, 75])
        lo = arr[df[key] < q25]
        hi = arr[df[key] > q75]
        vh, vl = safe_var(hi)+1e-12, safe_var(lo)+1e-12
        return float(np.log(vh) - np.log(vl))
    
    dlog_var_x = _logvar_contrast(r, "x")
    dlog_var_t = _logvar_contrast(r, "time")
    dlog_var_x_til = _logvar_contrast(r_tilde, "x")
    dlog_var_t_til = _logvar_contrast(r_tilde, "time")
    
    corr_abs_x = _safe_corr(np.abs(r), x)
    corr_abs_time = _safe_corr(np.abs(r), t)
    corr_abs_x_t = _safe_corr(np.abs(r_tilde), x)
    corr_abs_t_t = _safe_corr(np.abs(r_tilde), t)
    
    cov_r_x = float(np.cov(r, x, bias=True)[0, 1])
    cov_r_t = float(np.cov(r, t, bias=True)[0, 1])
    cov_rt_x = float(np.cov(r_tilde, x, bias=True)[0, 1])
    cov_rt_t = float(np.cov(r_tilde, t, bias=True)[0, 1])
    
    m_y = safe_mean(y)
    sd_y = safe_std(y)
    sd_r = safe_std(r)
    sd_rt = safe_std(r_tilde)
    
    stats = np.array([
        beta_hat[0], beta_hat[1], beta_hat[2],
        se_beta_proxy[0], se_beta_proxy[1], se_beta_proxy[2],
        R2_z_tfe, sc_slope_x, sc_time_mean, sc_time_sd, sc_intercept,
        beta_wls[0], beta_wls[1], beta_wls[2],
        se_beta_wls[0], se_beta_wls[1], se_beta_wls[2], R2_wls,
        bwls[0], bwls[1], R2_wls_within,
        dR2_x, dR2_t,
        between_subj_var, mean_within_var, ratio_bw_wi, icc_proxy, overall_var,
        mean_res_sd_subj, sd_res_sd_subj, mean_res_tsd_subj, sd_res_tsd_subj,
        mean_slope_zx_t, sd_slope_zx_t,
        mean_res_var_time, sd_res_var_time, mean_res_tvar_time, sd_res_tvar_time,
        dlog_var_x, dlog_var_t, dlog_var_x_til, dlog_var_t_til,
        corr_abs_x, corr_abs_time, corr_abs_x_t, corr_abs_t_t,
        cov_r_x, cov_r_t, cov_rt_x, cov_rt_t,
        m_y, sd_y, sd_r, sd_rt
    ], dtype=np.float32)
    
    return torch.tensor(np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6))


def simulator_summaries(theta_batch, clamp=True):
    """Vectorized simulator returning summary statistics."""
    outs = []
    for row in theta_batch:
        y = simulate_y(row.cpu().numpy(), clamp_log_sigma=clamp)
        s = targeted_summaries(y, x_vec, subject_ids, time_vec)
        outs.append(s)
    S = torch.stack(outs)
    return torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)


def run_mcmc_hetero(y_obs):
    """Run PyMC MCMC for heteroscedastic model."""
    eps = 1e-8
    
    with pm.Model() as model:
        x_data = pm.MutableData("x", x_vec)
        t_data = pm.MutableData("time", time_vec)
        
        beta = pm.Normal("beta", mu=0., sigma=10., shape=3)
        sigma_u2 = pm.Uniform("sigma_u2", lower=eps, upper=5.0)
        tau2 = pm.Uniform("tau2", lower=eps, upper=5.0)
        
        sigma_u = pm.Deterministic("sigma_u", pm.math.sqrt(sigma_u2))
        tau = pm.Deterministic("tau", pm.math.sqrt(tau2))
        
        u = pm.Normal("u", mu=0., sigma=sigma_u, shape=N_SUBJ)
        a = pm.Normal("a", mu=0., sigma=tau, shape=T_TIME)
        
        mu_y = beta[0] + u[subject_ids] + beta[1]*x_data + beta[2]*t_data
        log_sigma_y = GAMMA0_TRUE + a[time_vec] + B_TRUE*x_data
        sigma_y = pm.Deterministic("sigma_y", pm.math.exp(log_sigma_y))
        
        y_like = pm.Normal("y_like", mu=mu_y, sigma=sigma_y, observed=y_obs)
        trace = pm.sample(8000, tune=2000, target_accept=0.95, progressbar=False)
    
    post = az.extract(trace, var_names=["beta", "sigma_u2", "tau2"], combined=True)
    beta_s = np.asarray(post["beta"].transpose("sample", "beta_dim_0").values)
    sigma_u2s = np.asarray(post["sigma_u2"].values).reshape(-1, 1)
    tau2_s = np.asarray(post["tau2"].values).reshape(-1, 1)
    
    return np.hstack([beta_s, sigma_u2s, tau2_s])


class XEmbed(nn.Module):
    def __init__(self, d_in, d_hid=256, p=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.LayerNorm(d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class ThetaEmbed(nn.Module):
    def __init__(self, d_in=5, d_hid=256, p=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU()
        )
    
    def forward(self, θ):
        return self.net(θ)


def run_snpe_hetero(prior, simulator, x_obs, num_rounds=1, sims_per_round=100000):
    """Run SNPE-C inference."""
    D = x_obs.numel()
    
    snpe = SNPE_C(
        prior=prior,
        density_estimator=posterior_nn(
            model="maf",
            num_transforms=10,
            hidden_features=256,
            embedding_net=XEmbed(d_in=D, d_hid=256, p=0.10),
            z_score_theta="independent",
            z_score_x="independent",
        )
    )
    
    train_kw = {
        'learning_rate': 3e-4,
        'validation_fraction': 0.15,
        'training_batch_size': 1024,
        'stop_after_epochs': 40,
        'max_num_epochs': 600,
        'show_train_summary': False,
    }
    
    proposal = prior
    for r in range(num_rounds):
        θr = proposal.sample((sims_per_round,))
        Xr = simulator(θr)
        mask = torch.isfinite(Xr).all(dim=1)
        snpe.append_simulations(θr[mask], Xr[mask], proposal=proposal)
        snpe.train(**train_kw)
        posterior = snpe.build_posterior()
        proposal = posterior.set_default_x(x_obs)
    
    return posterior


def run_snle_hetero(prior, simulator, x_obs, num_rounds=1, sims_per_round=100000):
    """Run SNLE inference."""
    snle = SNLE(
        prior=prior,
        density_estimator=likelihood_nn(
            model="maf",
            num_transforms=10,
            hidden_features=256,
            embedding_net=ThetaEmbed(),
            z_score_theta="independent",
            z_score_x="independent",
        )
    )
    
    train_kw = {
        'validation_fraction': 0.15,
        'training_batch_size': 1024,
        'stop_after_epochs': 40,
        'max_num_epochs': 600,
        'show_train_summary': False,
    }
    
    proposal = prior
    for r in range(num_rounds):
        θr = proposal.sample((sims_per_round,))
        Xr = simulator(θr)
        mask = torch.isfinite(Xr).all(dim=1)
        snle.append_simulations(θr[mask], Xr[mask])
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
        BETA_TRUE['beta0'], BETA_TRUE['beta1'], BETA_TRUE['beta2'],
        SIGMA_U_TRUE2, TAU_TRUE2
    ])
    
    eps = 1e-8
    norm_prior = torch.distributions.Normal(torch.zeros(3), 10.*torch.ones(3))
    box_prior = BoxUniform(low=torch.tensor([eps, eps]), high=torch.tensor([5., 5.]))
    
    class MixedPrior(torch.distributions.Distribution):
        support = constraints.real
        has_rsample = False
        
        def sample(self, sample_shape=torch.Size()):
            n = int(torch.tensor(sample_shape).prod().item()) or 1
            beta = norm_prior.sample((n,))
            sigs = box_prior.sample((n,))
            return torch.cat([beta, sigs], dim=1)
        
        def log_prob(self, θ):
            β, sig = θ[..., :3], θ[..., 3:]
            lp = norm_prior.log_prob(β).sum(-1)
            inside = ((sig >= eps) & (sig <= 5.)).all(-1)
            log_unif = -2.0 * math.log(5.0 - eps)
            lp_support = torch.full_like(lp, log_unif)
            lp_bad = torch.full_like(lp, -1e10)
            return torch.where(inside, lp + lp_support, lp_bad)
    
    prior = MixedPrior()
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        y_obs = simulate_y(true_params, clamp_log_sigma=False)
        x_obs_full = targeted_summaries(y_obs, x_vec, subject_ids, time_vec)
        
        ref = simulator_summaries(prior.sample((14000,)), clamp=True)
        ref = ref[torch.isfinite(ref).all(dim=1)]
        
        q01 = torch.quantile(ref, 0.01, dim=0)
        q99 = torch.quantile(ref, 0.99, dim=0)
        ref_c = torch.minimum(torch.maximum(ref, q01), q99)
        med = ref_c.median(0).values
        mad = (ref_c - med).abs().median(0).values.clamp_min(1e-6)
        MAD2SD = 1.4826
        
        def normalize_stats(S):
            if S.ndim == 1:
                S = S.unsqueeze(0)
            S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)
            S = torch.minimum(torch.maximum(S, q01), q99)
            R = (S - med) / (mad * MAD2SD)
            C = (R - R.mean(0)).T @ (R - R.mean(0)) / max(1, R.shape[0]-1)
            eigvals, eigvecs = torch.linalg.eigh(C + 1e-6*torch.eye(C.shape[0]))
            W_zca = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
            mu_R = R.mean(0)
            Z = (R - mu_R) @ W_zca.T
            return torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-50.0, 50.0)
        
        x_obs = normalize_stats(x_obs_full).squeeze(0)
        
        def sims_to_features(θ):
            S = simulator_summaries(θ, clamp=True)
            return normalize_stats(S)
        
        print("  Running MCMC")
        start_time = time.time()
        samples_mcmc = run_mcmc_hetero(y_obs)
        mcmc_time = time.time() - start_time
        
        print("  Running SNPE-C")
        start_time = time.time()
        posterior_snpe = run_snpe_hetero(prior, sims_to_features, x_obs)
        samples_snpe = posterior_snpe.sample((12000,), x=x_obs).numpy()
        snpe_time = time.time() - start_time
        
        print("  Running SNLE")
        start_time = time.time()
        posterior_snle = run_snle_hetero(prior, sims_to_features, x_obs)
        samples_snle = posterior_snle.sample((8000,), x=x_obs).numpy()
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
    
    os.makedirs('hetero_results', exist_ok=True)
    save_results_table(results_list, 'hetero_results/standard_inference_summary.csv')
    
    print("\nStandard inference complete!")
    return results_list


def amortized_inference_experiment(n_realizations=50):
    """Train once, apply to multiple datasets."""
    print("\n" + "="*70)
    print("AMORTIZED INFERENCE (Train once, apply to many datasets)")
    print("="*70)
    
    true_params = np.array([
        BETA_TRUE['beta0'], BETA_TRUE['beta1'], BETA_TRUE['beta2'],
        SIGMA_U_TRUE2, TAU_TRUE2
    ])
    
    eps = 1e-8
    norm_prior = torch.distributions.Normal(torch.zeros(3), 10.*torch.ones(3))
    box_prior = BoxUniform(low=torch.tensor([eps, eps]), high=torch.tensor([5., 5.]))
    
    class MixedPrior(torch.distributions.Distribution):
        support = constraints.real
        has_rsample = False
        
        def sample(self, sample_shape=torch.Size()):
            n = int(torch.tensor(sample_shape).prod().item()) or 1
            beta = norm_prior.sample((n,))
            sigs = box_prior.sample((n,))
            return torch.cat([beta, sigs], dim=1)
        
        def log_prob(self, θ):
            β, sig = θ[..., :3], θ[..., 3:]
            lp = norm_prior.log_prob(β).sum(-1)
            inside = ((sig >= eps) & (sig <= 5.)).all(-1)
            log_unif = -2.0 * math.log(5.0 - eps)
            lp_support = torch.full_like(lp, log_unif)
            lp_bad = torch.full_like(lp, -1e10)
            return torch.where(inside, lp + lp_support, lp_bad)
    
    prior = MixedPrior()
    
    y_dummy = simulate_y(true_params, clamp_log_sigma=False)
    x_dummy_full = targeted_summaries(y_dummy, x_vec, subject_ids, time_vec)
    
    ref = simulator_summaries(prior.sample((14000,)), clamp=True)
    ref = ref[torch.isfinite(ref).all(dim=1)]
    
    q01 = torch.quantile(ref, 0.01, dim=0)
    q99 = torch.quantile(ref, 0.99, dim=0)
    ref_c = torch.minimum(torch.maximum(ref, q01), q99)
    med = ref_c.median(0).values
    mad = (ref_c - med).abs().median(0).values.clamp_min(1e-6)
    MAD2SD = 1.4826
    
    def normalize_stats(S):
        if S.ndim == 1:
            S = S.unsqueeze(0)
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)
        S = torch.minimum(torch.maximum(S, q01), q99)
        R = (S - med) / (mad * MAD2SD)
        C = (R - R.mean(0)).T @ (R - R.mean(0)) / max(1, R.shape[0]-1)
        eigvals, eigvecs = torch.linalg.eigh(C + 1e-6*torch.eye(C.shape[0]))
        W_zca = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        mu_R = R.mean(0)
        Z = (R - mu_R) @ W_zca.T
        return torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-50.0, 50.0)
    
    x_dummy = normalize_stats(x_dummy_full).squeeze(0)
    
    def sims_to_features(θ):
        S = simulator_summaries(θ, clamp=True)
        return normalize_stats(S)
    
    start_train_snpe = time.time()
    posterior_snpe = run_snpe_hetero(prior, sims_to_features, x_dummy,
                                     num_rounds=1, sims_per_round=100000)
    snpe_train_time = time.time() - start_train_snpe
    print(f"SNPE-C training time: {snpe_train_time:.2f}s")
    
    start_train_snle = time.time()
    posterior_snle = run_snle_hetero(prior, sims_to_features, x_dummy,
                                     num_rounds=1, sims_per_round=100000)
    snle_train_time = time.time() - start_train_snle
    print(f"SNLE training time: {snle_train_time:.2f}s")
    
    results_list = []
    
    for real_idx in range(n_realizations):
        print(f"\nRealization {real_idx + 1}/{n_realizations}")
        
        y_obs = simulate_y(true_params, clamp_log_sigma=False)
        x_obs_full = targeted_summaries(y_obs, x_vec, subject_ids, time_vec)
        x_obs = normalize_stats(x_obs_full).squeeze(0)
        
        start_time = time.time()
        samples_snpe = posterior_snpe.sample((12000,), x=x_obs).numpy()
        snpe_time = time.time() - start_time
        
        start_time = time.time()
        samples_snle = posterior_snle.sample((8000,), x=x_obs).numpy()
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
    
    os.makedirs('hetero_results', exist_ok=True)
    save_results_table(results_list, 'hetero_results/amortized_inference_summary.csv',
                      methods=['SNPE_C', 'SNLE'])
    
    print("\nAmortized inference complete!")
    print(f"Training time - SNPE-C: {snpe_train_time:.2f}s, SNLE: {snle_train_time:.2f}s")
    
    return results_list


def plot_single_realization(samples_mcmc, samples_snpe, samples_snle, true_params):
    """Plot results for a single realization."""
    param_names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$",
                   r"$\sigma_u^2$", r"$\tau^2$"]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
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
    plt.savefig('hetero_results/posterior_comparison.png', dpi=300)
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
    n_draws = 200
    grid = np.linspace(-10, 20, 200)
    
    y_true = simulate_y(true_params, clamp_log_sigma=False)
    kdes_ppc = []
    
    indices = np.random.choice(len(samples), n_draws, replace=False)
    for idx in indices:
        y_sim = simulate_y(samples[idx], clamp_log_sigma=False)
        kdes_ppc.append(gaussian_kde(y_sim)(grid))
    
    kdes_ppc = np.array(kdes_ppc)
    median_kde = np.median(kdes_ppc, axis=0)
    p25_kde = np.percentile(kdes_ppc, 25, axis=0)
    p75_kde = np.percentile(kdes_ppc, 75, axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(y_true, bins=30, stat='density', color='blue', alpha=0.25,
                label='Observed')
    plt.plot(grid, median_kde, color='blue', lw=2, label='Median predictive')
    plt.fill_between(grid, p25_kde, p75_kde, color='blue', alpha=0.25,
                    label='25-75% predictive')
    
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("Heteroscedastic Regression Model")
    print("="*70)
    
    results_standard = standard_inference_experiment(n_realizations=50)
    results_amortized = amortized_inference_experiment(n_realizations=50)
    
    first_result = results_standard[0]
    true_params = np.array([
        BETA_TRUE['beta0'], BETA_TRUE['beta1'], BETA_TRUE['beta2'],
        SIGMA_U_TRUE2, TAU_TRUE2
    ])
    
    plot_single_realization(
        first_result['MCMC']['samples'],
        first_result['SNPE_C']['samples'],
        first_result['SNLE']['samples'],
        true_params
    )
    
    param_names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$",
                   r"$\sigma_u^2$", r"$\tau^2$"]
    
    plot_pairplot(first_result['MCMC']['samples'], param_names, true_params,
                 'hetero_results/pairplot_mcmc.png')
    plot_pairplot(first_result['SNPE_C']['samples'], param_names, true_params,
                 'hetero_results/pairplot_snpe.png')
    plot_pairplot(first_result['SNLE']['samples'], param_names, true_params,
                 'hetero_results/pairplot_snle.png')
    
    plot_ppc(first_result['MCMC']['samples'], true_params,
            'hetero_results/ppc_mcmc.png')
    plot_ppc(first_result['SNPE_C']['samples'], true_params,
            'hetero_results/ppc_snpe.png')
    plot_ppc(first_result['SNLE']['samples'], true_params,
            'hetero_results/ppc_snle.png')


if __name__ == "__main__":
    main()

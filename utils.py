"""
Utility functions for MCMC vs SBI experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_coverage(samples, true_value, alpha=0.05):
    """
    Compute credible interval coverage.
    
    Args:
        samples: Array of posterior samples (n_samples,) or (n_samples, n_params)
        true_value: True parameter value(s)
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        coverage: Boolean or array of booleans indicating coverage
    """
    lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return (true_value >= lower) & (true_value <= upper)


def compute_metrics(samples, true_values):
    """
    Compute RMSE, MAE, bias for parameter estimates.
    
    Args:
        samples: Array of posterior samples (n_samples, n_params)
        true_values: Array of true parameter values (n_params,)
    
    Returns:
        dict with metrics
    """
    means = np.mean(samples, axis=0)
    bias = means - true_values
    rmse = np.sqrt(mean_squared_error(true_values, means))
    mae = mean_absolute_error(true_values, means)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'mae': mae,
        'mean': means,
        'std': np.std(samples, axis=0)
    }


def save_results_table(results_list, output_path, methods=['MCMC', 'SNPE_C', 'SNLE']):
    """
    Save aggregated results across realizations to CSV.
    
    Args:
        results_list: List of result dicts for each realization
        output_path: Path to save CSV file
        methods: List of method names
    """
    summary_data = []
    
    n_realizations = len(results_list)
    param_names = list(results_list[0][methods[0]]['mean'].keys())
    
    for param in param_names:
        for method in methods:
            means = [r[method]['mean'][param] for r in results_list]
            stds = [r[method]['std'][param] for r in results_list]
            biases = [r[method]['bias'][param] for r in results_list]
            
            summary_data.append({
                'Parameter': param,
                'Method': method,
                'Mean_of_Means': np.mean(means),
                'Std_of_Means': np.std(means),
                'Mean_of_Stds': np.mean(stds),
                'Mean_Bias': np.mean(biases),
                'RMSE': np.sqrt(np.mean(np.array(biases)**2))
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def plot_density_comparison(samples_dict, true_values, param_names, save_path=None):
    """
    Create overlay density plots for all methods.
    
    Args:
        samples_dict: Dict with keys ['MCMC', 'SNPE_C', 'SNLE'] and values as sample arrays
        true_values: Array of true parameter values
        param_names: List of parameter names for labels
        save_path: Path to save figure (optional)
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4), sharey=True)
    
    if n_params == 1:
        axes = [axes]
    
    colors = {'MCMC': 'blue', 'SNPE_C': 'green', 'SNLE': 'red'}
    
    for i, param in enumerate(param_names):
        for method, samples in samples_dict.items():
            if samples.ndim == 1:
                data = samples
            else:
                data = samples[:, i]
            sns.kdeplot(data, label=method, color=colors[method], 
                       fill=True, alpha=0.5, ax=axes[i])
        
        axes[i].axvline(true_values[i], color='black', linestyle='--', 
                       linewidth=2, label='True Value')
        axes[i].set_xlabel(param)
        
        if i == 0:
            axes[i].set_ylabel('Density')
            axes[i].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pairplot(samples, true_values, param_names, save_path=None):
    """
    Create corner/pair plot for posterior samples.
    
    Args:
        samples: Array of samples (n_samples, n_params)
        true_values: Array of true parameter values
        param_names: List of parameter names
        save_path: Path to save figure
    """
    df = pd.DataFrame(samples, columns=param_names)
    
    g = sns.pairplot(df, diag_kind='kde', corner=True)
    axes = g.axes
    n_params = len(param_names)
    
    for i in range(n_params):
        for j in range(i):
            ax = axes[i, j]
            ax.plot(true_values[j], true_values[i], 'r+', markersize=10)
            ax.set_xlabel(param_names[j])
            ax.set_ylabel(param_names[i])
    
    for i in range(n_params):
        ax = axes[i, i]
        ax.axvline(true_values[i], color='r', linestyle='--')
        ax.set_xlabel(param_names[i])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_traceplot(samples, true_values, param_names, save_path=None):
    """
    Create trace plots for posterior samples.
    
    Args:
        samples: Array of samples (n_samples, n_params)
        true_values: Array of true parameter values
        param_names: List of parameter names
        save_path: Path to save figure
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params))
    
    if n_params == 1:
        axes = [axes]
    
    for i, param in enumerate(param_names):
        if samples.ndim == 1:
            axes[i].plot(samples, alpha=0.7)
        else:
            axes[i].plot(samples[:, i], alpha=0.7)
        axes[i].axhline(true_values[i], color='red', linestyle='--')
        axes[i].set_ylabel(param)
        if i < n_params - 1:
            axes[i].set_xticklabels([])
    
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

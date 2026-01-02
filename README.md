# Simulation-Based Bayesian Inference via Neural Posterior Modeling

Code repository for reproducing the numerical experiments from the PhD thesis implementing inference using Markov Chain Monte Carlo (MCMC) and Simulation-Based Inference (SBI) methods, with a focus on neural posterior estimation.

The experiments cover both **time-series models** and **mixed-effects models**, using standard Bayesian inference and amortized SBI approaches.

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Each experiment can be run independently from the command line.

### Time-Series Chapter

```bash
python time_series/grw_experiment.py
python time_series/arma_experiment.py
python time_series/lv_experiment.py
python time_series/sir_experiment.py
python time_series/sv1_experiment.py
python time_series/sv2_experiment.py
```

### Mixed-Effects Chapter

```bash
python mixed_effects/svc_experiment.py
python mixed_effects/pk_experiment.py
python mixed_effects/hetero_experiment.py
python mixed_effects/radon_experiment.py
```

## Inference Setups

Each experiment supports **two inference paradigms**:

1. **Standard inference**  
   - 50 independent dataset realizations  
   - A new SBI network is trained from scratch for each realization  

2. **Amortized inference**  
   - A single SBI network is trained once  
   - The trained network is reused to infer posteriors for 50 realizations  

Depending on the model and inference method, experiments may take anywhere from a **few hours to several days** to complete.

## Output Structure

Each experiment creates its own results directory with the following structure:

```text
{experiment}_results/
├── standard_inference_summary.csv      # Aggregated metrics over 50 runs
├── amortized_inference_summary.csv     # Metrics for amortized inference
├── posterior_comparison.png            # Posterior density overlays
├── pairplot_MCMC.png                   # MCMC posterior pair plot
├── pairplot_SNPE_C.png                 # SNPE-C posterior pair plot
└── pairplot_SNLE.png                   # SNLE posterior pair plot
```

All figures and summary tables are saved automatically to facilitate comparison across inference methods.


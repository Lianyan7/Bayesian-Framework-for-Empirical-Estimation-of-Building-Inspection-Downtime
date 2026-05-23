# A Bayesian Framework for Empirical Estimation of Post-earthquake Building Inspection Time

Supplementary materials for the paper **"A Bayesian framework for the empirical estimation of building inspection time following earthquakes"**.

This repository provides the analysis code and aggregated model outputs supporting the Bayesian estimation of post-earthquake building inspection time. The framework estimates agency- and facility-specific inspection-time distributions and reports posterior predictive summaries, prior sensitivity checks, and MCMC convergence diagnostics.

## Citation

Li, L., Chang-Richards, A., et al. (2025). *A Bayesian framework for the empirical estimation of building inspection downtime following earthquakes.*

## Repository Contents

- `mcmc_posterior_estimation_and_diagnostics.py`  
  Main Python script for posterior estimation, posterior predictive summaries, MCMC diagnostics, trace/autocorrelation plots, and posterior predictive checks.

- `supplementary-results/`  
  Aggregated supplementary outputs used in the revised manuscript.

- `supplementary-results/prior-sensitivity/`  
  Prior sensitivity analysis comparing NZ-context and REDi-informed priors.

- `supplementary-results/mcmc-diagnostics/`  
  MCMC convergence diagnostic summaries, trace plots, and autocorrelation plots.

- `supplementary-results/posterior-predictive-figures/`  
  Posterior predictive PDF and CDF figures.

- `Summarised statistics of the collected empirical dataset.xlsx`  
  Aggregated statistical summaries of the empirical dataset.

## Code

`mcmc_posterior_estimation_and_diagnostics.py` reproduces the posterior estimation workflow, including:

- agency-facility-specific likelihood models;
- MCMC posterior estimation using PyMC;
- posterior predictive summaries;
- MCMC convergence diagnostics;
- trace and autocorrelation plots;
- posterior predictive checks.

## Supplementary Results

The `supplementary-results/` folder contains aggregated outputs used in the revised manuscript, including:

- prior sensitivity analysis comparing NZ-context and REDi-informed priors;
- MCMC diagnostic summaries, including R-hat, bulk ESS, tail ESS, and Monte Carlo standard error;
- trace and autocorrelation plots;
- posterior predictive PDF and CDF figures.

These files contain aggregated model outputs only. They do not contain raw building-level inspection records, claim identifiers, addresses, or proprietary property information.

## Data Availability

The empirical dataset used in the study contains sensitive information from private insurance claim settlements and proprietary property records. Under the University of Auckland Human Participants Ethics Committee approval and data-sharing agreements with contributing institutions, the raw data cannot be released publicly.

Aggregated statistical summaries, analysis code, model outputs, sensitivity results, and MCMC diagnostics are provided in this repository to support transparency and reproducibility.

## Requirements

The analysis was conducted in Python using:

- `numpy`
- `pandas`
- `scipy`
- `pymc`
- `arviz`
- `matplotlib`
- `openpyxl`

Install dependencies with:

`pip install numpy pandas scipy pymc arviz matplotlib openpyxl`

## License

This repository is released under the MIT License.

## Contact

For questions about the code or supplementary materials, please contact the corresponding author or open an issue in this repository.

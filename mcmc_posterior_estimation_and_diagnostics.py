"""
Bayesian posterior estimation for post-earthquake inspection time.

This script estimates posterior and posterior predictive distributions for
inspection time by agency-facility group using selected likelihoods.

Reviewer-facing outputs:
1. MCMC settings: chains, tuning iterations, retained posterior draws.
2. Posterior summaries and posterior predictive summaries.
3. MCMC diagnostics: R-hat, bulk ESS, tail ESS, MCSE, ESS ratios.
4. Trace plots and autocorrelation plots for each agency-facility stratum.
5. Posterior predictive CDF checks against empirical data.

Note:
MCMC diagnostics assess sampler convergence only. Likelihood adequacy is
assessed separately through distributional diagnostics and posterior predictive
checks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_func
from scipy.optimize import brentq


# ==================================================
# User settings
# ==================================================
file_path = Path("Paper 6-data.xlsx")
output_file_path = Path("MCMC_Posterior_Results_by_Facility.xlsx")

output_dir = Path("mcmc_outputs")
trace_dir = output_dir / "trace_plots"
autocorr_dir = output_dir / "autocorrelation_plots"
ppc_dir = output_dir / "posterior_predictive_checks"

for d in [output_dir, trace_dir, autocorr_dir, ppc_dir]:
    d.mkdir(parents=True, exist_ok=True)

sheet_names = ["NHC", "CCC", "CERA"]
downtime_col = "Inspection Downtime"
facility_col = "Facility"

draws = 3000
tune = 2000
chains = 4
cores = 1
target_accept = 0.90
random_seed = 123
parameter_prior_sd = 0.75

retained_draws_total = draws * chains


# ==================================================
# NZ-context priors and selected likelihoods
# ==================================================
prior_moments = {
    ("NHC", "Essential"): {
        "mean": 709.39,
        "sd": 405.46,
        "likelihood": "Weibull",
    },
    ("NHC", "Non-Essential"): {
        "mean": 657.23,
        "sd": 367.00,
        "likelihood": "Weibull",
    },
    ("CCC", "Essential"): {
        "mean": 16.36,
        "sd": 32.43,
        "likelihood": "Lognormal",
    },
    ("CCC", "Non-Essential"): {
        "mean": 30.11,
        "sd": 48.96,
        "likelihood": "Lognormal",
    },
    ("CERA", "Essential"): {
        "mean": 735.53,
        "sd": 238.88,
        "likelihood": "Weibull",
    },
    ("CERA", "Non-Essential"): {
        "mean": 764.80,
        "sd": 241.79,
        "likelihood": "Gamma",
    },
}


# ==================================================
# Parameter conversion utilities
# ==================================================
def lognormal_params_from_mean_sd(mean, sd):
    sigma = np.sqrt(np.log(1.0 + (sd / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma


def gamma_params_from_mean_sd(mean, sd):
    alpha = (mean / sd) ** 2
    beta = mean / (sd**2)  # rate
    return alpha, beta


def weibull_cv(shape):
    g1 = gamma_func(1.0 + 1.0 / shape)
    g2 = gamma_func(1.0 + 2.0 / shape)
    return np.sqrt(g2 / (g1**2) - 1.0)


def weibull_params_from_mean_sd(mean, sd):
    target_cv = sd / mean

    def objective(shape):
        return weibull_cv(shape) - target_cv

    shape = brentq(objective, 0.1, 50.0)
    scale = mean / gamma_func(1.0 + 1.0 / shape)
    return shape, scale


def standardize_facility(value):
    text = str(value).strip().lower()
    if "non" in text:
        return "Non-Essential"
    if "ess" in text:
        return "Essential"
    return str(value).strip()


def safe_name(text):
    return str(text).replace(" ", "_").replace("-", "_").replace("/", "_")


# ==================================================
# Data handling
# ==================================================
def get_group_data(df, facility):
    df = df.copy()
    df[downtime_col] = pd.to_numeric(df[downtime_col], errors="coerce")
    df = df.dropna(subset=[downtime_col])
    df = df[df[downtime_col] > 0]

    if facility_col in df.columns:
        df["_facility_clean"] = df[facility_col].apply(standardize_facility)
        data = df.loc[df["_facility_clean"] == facility, downtime_col].values
    else:
        data = df[downtime_col].values

    return np.asarray(data, dtype=float)


# ==================================================
# Bayesian model
# ==================================================
def fit_mcmc_model(data, likelihood, prior_mean, prior_sd):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < 5:
        raise ValueError("Too few observations for MCMC fitting.")

    with pm.Model() as model:

        if likelihood == "Lognormal":
            mu0, sigma0 = lognormal_params_from_mean_sd(prior_mean, prior_sd)
            log_data = np.log(data)

            mu = pm.Normal("mu", mu=mu0, sigma=parameter_prior_sd)

            sigma = pm.HalfNormal(
                "sigma",
                sigma=max(1.0, sigma0 * 1.5),
            )

            pm.Normal("log_y", mu=mu, sigma=sigma, observed=log_data)

        elif likelihood == "Gamma":
            alpha0, beta0 = gamma_params_from_mean_sd(prior_mean, prior_sd)

            alpha = pm.LogNormal(
                "alpha",
                mu=np.log(alpha0),
                sigma=parameter_prior_sd,
            )
            beta = pm.LogNormal(
                "beta",
                mu=np.log(beta0),
                sigma=parameter_prior_sd,
            )

            pm.Gamma("y", alpha=alpha, beta=beta, observed=data)

        elif likelihood == "Weibull":
            shape0, scale0 = weibull_params_from_mean_sd(prior_mean, prior_sd)

            shape = pm.LogNormal(
                "shape",
                mu=np.log(shape0),
                sigma=parameter_prior_sd,
            )
            scale = pm.LogNormal(
                "scale",
                mu=np.log(scale0),
                sigma=parameter_prior_sd,
            )

            pm.Weibull("y", alpha=shape, beta=scale, observed=data)

        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return trace


# ==================================================
# Posterior and posterior predictive summaries
# ==================================================
def posterior_predictive_samples_from_trace(trace, likelihood, rng):
    posterior = trace.posterior

    if likelihood == "Lognormal":
        mu = posterior["mu"].values.ravel()
        sigma = posterior["sigma"].values.ravel()
        return rng.lognormal(mean=mu, sigma=sigma)

    if likelihood == "Gamma":
        alpha = posterior["alpha"].values.ravel()
        beta = posterior["beta"].values.ravel()
        return rng.gamma(shape=alpha, scale=1.0 / beta)

    if likelihood == "Weibull":
        shape = posterior["shape"].values.ravel()
        scale = posterior["scale"].values.ravel()
        return scale * rng.weibull(a=shape, size=len(shape))

    raise ValueError(f"Unsupported likelihood: {likelihood}")


def summarize_distribution_moments(trace, likelihood):
    posterior = trace.posterior

    if likelihood == "Lognormal":
        mu = posterior["mu"].values.ravel()
        sigma = posterior["sigma"].values.ravel()

        mean_samples = np.exp(mu + 0.5 * sigma**2)
        var_samples = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)

        param_summary = {
            "param1_name": "mu",
            "param1_mean": np.mean(mu),
            "param1_sd": np.std(mu, ddof=1),
            "param2_name": "sigma",
            "param2_mean": np.mean(sigma),
            "param2_sd": np.std(sigma, ddof=1),
        }

    elif likelihood == "Gamma":
        alpha = posterior["alpha"].values.ravel()
        beta = posterior["beta"].values.ravel()

        mean_samples = alpha / beta
        var_samples = alpha / (beta**2)

        param_summary = {
            "param1_name": "alpha",
            "param1_mean": np.mean(alpha),
            "param1_sd": np.std(alpha, ddof=1),
            "param2_name": "beta_rate",
            "param2_mean": np.mean(beta),
            "param2_sd": np.std(beta, ddof=1),
        }

    elif likelihood == "Weibull":
        shape = posterior["shape"].values.ravel()
        scale = posterior["scale"].values.ravel()

        g1 = gamma_func(1.0 + 1.0 / shape)
        g2 = gamma_func(1.0 + 2.0 / shape)

        mean_samples = scale * g1
        var_samples = scale**2 * (g2 - g1**2)

        param_summary = {
            "param1_name": "shape",
            "param1_mean": np.mean(shape),
            "param1_sd": np.std(shape, ddof=1),
            "param2_name": "scale",
            "param2_mean": np.mean(scale),
            "param2_sd": np.std(scale, ddof=1),
        }

    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    rng = np.random.default_rng(random_seed)
    pred_samples = posterior_predictive_samples_from_trace(trace, likelihood, rng)

    pred_mean = np.mean(pred_samples)
    pred_sd = np.std(pred_samples, ddof=1)

    return {
        **param_summary,
        "posterior_distribution_mean": np.mean(mean_samples),
        "posterior_distribution_mean_sd": np.std(mean_samples, ddof=1),
        "posterior_distribution_mean_q2.5": np.quantile(mean_samples, 0.025),
        "posterior_distribution_mean_q97.5": np.quantile(mean_samples, 0.975),
        "posterior_predictive_mean": pred_mean,
        "posterior_predictive_sd": pred_sd,
        "posterior_predictive_cov": pred_sd / pred_mean,
        "posterior_predictive_median": np.quantile(pred_samples, 0.50),
        "posterior_predictive_q2.5": np.quantile(pred_samples, 0.025),
        "posterior_predictive_q97.5": np.quantile(pred_samples, 0.975),
        "posterior_predictive_p90": np.quantile(pred_samples, 0.90),
        "posterior_predictive_p95": np.quantile(pred_samples, 0.95),
    }


# ==================================================
# Reviewer-facing diagnostics
# ==================================================
def make_diagnostic_summary(trace, sheet, facility, likelihood):
    diag = az.summary(trace, round_to=4)
    diag = diag.reset_index().rename(columns={"index": "Parameter"})

    diag["Sheet"] = sheet
    diag["Facility"] = facility
    diag["Likelihood"] = likelihood
    diag["Chains"] = chains
    diag["Tune per chain"] = tune
    diag["Draws per chain"] = draws
    diag["Retained posterior draws"] = retained_draws_total

    if "ess_bulk" in diag.columns:
        diag["ess_bulk_ratio"] = diag["ess_bulk"] / retained_draws_total

    if "ess_tail" in diag.columns:
        diag["ess_tail_ratio"] = diag["ess_tail"] / retained_draws_total

    return diag


def save_trace_plot(trace, sheet, facility, likelihood):
    az.plot_trace(trace)
    fig = plt.gcf()
    fig.suptitle(f"{sheet} - {facility} - {likelihood}: trace plot", y=1.02)

    path = trace_dir / f"trace_{safe_name(sheet)}_{safe_name(facility)}_{safe_name(likelihood)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_autocorr_plot(trace, sheet, facility, likelihood):
    az.plot_autocorr(trace, max_lag=100)
    fig = plt.gcf()
    fig.suptitle(f"{sheet} - {facility} - {likelihood}: autocorrelation", y=1.02)

    path = autocorr_dir / f"autocorr_{safe_name(sheet)}_{safe_name(facility)}_{safe_name(likelihood)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_posterior_predictive_cdf_check(data, pred_samples, sheet, facility, likelihood):
    data = np.sort(np.asarray(data, dtype=float))
    pred_samples = np.sort(np.asarray(pred_samples, dtype=float))

    y_data = np.arange(1, len(data) + 1) / len(data)
    y_pred = np.arange(1, len(pred_samples) + 1) / len(pred_samples)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(data, y_data, color="black", lw=1.8, label="Empirical CDF")
    ax.plot(pred_samples, y_pred, color="#D55E00", lw=1.6, label="Posterior predictive CDF")

    ax.set_xlabel("Inspection time (days)")
    ax.set_ylabel("Probability of non-exceedance")
    ax.set_title(f"{sheet} - {facility} - {likelihood}: posterior predictive check")
    ax.grid(True, color="#D9D9D9", linestyle=":", linewidth=0.6)
    ax.legend(frameon=False)

    path = ppc_dir / f"ppc_cdf_{safe_name(sheet)}_{safe_name(facility)}_{safe_name(likelihood)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ==================================================
# Main workflow
# ==================================================
def main():
    posterior_rows = []
    diagnostic_rows = []
    plot_rows = []

    settings_df = pd.DataFrame([{
        "sampler": "PyMC NUTS",
        "chains": chains,
        "cores": cores,
        "tune_per_chain": tune,
        "draws_per_chain": draws,
        "retained_posterior_draws_total": retained_draws_total,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "parameter_prior_sd": parameter_prior_sd,
        "note": (
            "MCMC diagnostics evaluate convergence only. "
            "Likelihood adequacy is assessed separately using posterior predictive checks."
        ),
    }])

    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)

        for facility in ["Essential", "Non-Essential"]:
            key = (sheet, facility)
            if key not in prior_moments:
                continue

            prior_mean = prior_moments[key]["mean"]
            prior_sd = prior_moments[key]["sd"]
            likelihood = prior_moments[key]["likelihood"]

            data = get_group_data(df, facility)

            print("\n" + "=" * 72)
            print(f"Fitting {sheet} - {facility}")
            print(f"Likelihood: {likelihood}")
            print(f"N = {len(data)}")
            print(f"Observed mean = {np.mean(data):.3f}")
            print(f"Observed SD = {np.std(data, ddof=1):.3f}")
            print(f"Prior mean = {prior_mean:.3f}")
            print(f"Prior SD = {prior_sd:.3f}")

            trace = fit_mcmc_model(
                data=data,
                likelihood=likelihood,
                prior_mean=prior_mean,
                prior_sd=prior_sd,
            )

            moment_summary = summarize_distribution_moments(trace, likelihood)

            posterior_rows.append({
                "Sheet": sheet,
                "Facility": facility,
                "Likelihood": likelihood,
                "N": len(data),
                "Observed mean": np.mean(data),
                "Observed SD": np.std(data, ddof=1),
                "Prior mean": prior_mean,
                "Prior SD": prior_sd,
                "Prior COV": prior_sd / prior_mean,
                **moment_summary,
            })

            diag = make_diagnostic_summary(trace, sheet, facility, likelihood)
            diagnostic_rows.append(diag)

            rng = np.random.default_rng(random_seed)
            pred_samples = posterior_predictive_samples_from_trace(trace, likelihood, rng)

            trace_path = save_trace_plot(trace, sheet, facility, likelihood)
            autocorr_path = save_autocorr_plot(trace, sheet, facility, likelihood)
            ppc_path = save_posterior_predictive_cdf_check(
                data=data,
                pred_samples=pred_samples,
                sheet=sheet,
                facility=facility,
                likelihood=likelihood,
            )

            plot_rows.append({
                "Sheet": sheet,
                "Facility": facility,
                "Likelihood": likelihood,
                "Trace plot": str(trace_path),
                "Autocorrelation plot": str(autocorr_path),
                "Posterior predictive CDF check": str(ppc_path),
            })

    posterior_df = pd.DataFrame(posterior_rows)
    diagnostics_df = pd.concat(diagnostic_rows, ignore_index=True)
    plots_df = pd.DataFrame(plot_rows)

    with pd.ExcelWriter(output_file_path) as writer:
        settings_df.to_excel(writer, sheet_name="MCMC settings", index=False)
        posterior_df.to_excel(writer, sheet_name="Posterior summaries", index=False)
        diagnostics_df.to_excel(writer, sheet_name="MCMC diagnostics", index=False)
        plots_df.to_excel(writer, sheet_name="Diagnostic plot paths", index=False)

    print("\nSaved Excel results to:")
    print(output_file_path.resolve())

    print("\nSaved diagnostic plots to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

"""Sensitivity analysis comparing NZ empirical and REDi prior specifications."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_func
from scipy.optimize import brentq
from scipy.stats import gamma as scipy_gamma

# --------------------------------------------------
# User settings
# --------------------------------------------------
file_path = "Paper 6-data.xlsx"
output_file_path = "Prior_Sensitivity_Analysis.xlsx"

sheet_names = ["NHC", "CCC", "CERA"]
downtime_col = "Inspection Downtime"
facility_col = "Facility"

draws = 3000
tune = 2000
chains = 4
cores = 1
target_accept = 0.90
random_seed = 123
retained_draws_total = draws * chains

# Prior spread around parameter centres.
# Smaller = stronger prior, larger = weaker prior.
log_parameter_prior_sd = 0.75

# --------------------------------------------------
# Selected likelihoods
# --------------------------------------------------
selected_likelihoods = {
    ("NHC", "Essential"): "Weibull",
    ("NHC", "Non-Essential"): "Weibull",
    ("CCC", "Essential"): "Lognormal",
    ("CCC", "Non-Essential"): "Lognormal",
    ("CERA", "Essential"): "Weibull",
    ("CERA", "Non-Essential"): "Gamma",
}

assessment_type = {
    "NHC": "DDE",
    "CCC": "RBA",
    "CERA": "DDE",
}

# --------------------------------------------------
# Prior set 1: NZ empirical-Bayes reference moments
# --------------------------------------------------
nz_priors = {
    ("NHC", "Essential"): {"type": "mean_sd", "mean": 709.39, "sd": 405.46},
    ("NHC", "Non-Essential"): {"type": "mean_sd", "mean": 657.23, "sd": 367.00},
    ("CCC", "Essential"): {"type": "mean_sd", "mean": 16.36, "sd": 32.43},
    ("CCC", "Non-Essential"): {"type": "mean_sd", "mean": 30.11, "sd": 48.96},
    ("CERA", "Essential"): {"type": "mean_sd", "mean": 735.53, "sd": 238.88},
    ("CERA", "Non-Essential"): {"type": "mean_sd", "mean": 764.80, "sd": 241.79},
}

# --------------------------------------------------
# Prior set 2: established REDi reference information
# --------------------------------------------------
redi_priors = {
    ("NHC", "Essential"): {"type": "median_cov", "median": 294.0, "cov": 0.30},
    ("NHC", "Non-Essential"): {"type": "median_cov", "median": 28.0, "cov": 0.30},
    ("CCC", "Essential"): {"type": "median_cov", "median": 2.0, "cov": 0.30},
    ("CCC", "Non-Essential"): {"type": "median_cov", "median": 5.0, "cov": 0.30},
    ("CERA", "Essential"): {"type": "median_cov", "median": 350.0, "cov": 0.30},
    ("CERA", "Non-Essential"): {"type": "median_cov", "median": 84.0, "cov": 0.30},
}

prior_sets = {
    "NZ_empirical_prior": nz_priors,
    "REDi_prior": redi_priors,
}


# --------------------------------------------------
# Parameter conversion functions
# --------------------------------------------------
def lognormal_params_from_mean_sd(mean, sd):
    sigma = np.sqrt(np.log(1.0 + (sd / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def lognormal_params_from_median_cov(median, cov):
    mu = np.log(median)
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    return mu, sigma


def gamma_params_from_mean_sd(mean, sd):
    alpha = (mean / sd) ** 2
    beta = mean / (sd ** 2)  # rate
    return alpha, beta


def gamma_params_from_median_cov(median, cov):
    alpha = 1.0 / (cov ** 2)

    # scipy gamma median with scale = 1.0
    unit_median = scipy_gamma.ppf(0.5, a=alpha, scale=1.0)

    # median = unit_median / beta, because scale = 1 / beta
    beta = unit_median / median
    return alpha, beta


def weibull_cv(shape):
    g1 = gamma_func(1.0 + 1.0 / shape)
    g2 = gamma_func(1.0 + 2.0 / shape)
    return np.sqrt(g2 / (g1 ** 2) - 1.0)


def weibull_shape_from_cov(cov):
    def objective(shape):
        return weibull_cv(shape) - cov

    return brentq(objective, 0.1, 100.0)


def weibull_params_from_mean_sd(mean, sd):
    cov = sd / mean
    shape = weibull_shape_from_cov(cov)
    scale = mean / gamma_func(1.0 + 1.0 / shape)
    return shape, scale


def weibull_params_from_median_cov(median, cov):
    shape = weibull_shape_from_cov(cov)
    scale = median / (np.log(2.0) ** (1.0 / shape))
    return shape, scale


def get_prior_parameter_centres(prior_spec, likelihood):
    if prior_spec["type"] == "mean_sd":
        mean = prior_spec["mean"]
        sd = prior_spec["sd"]

        if likelihood == "Lognormal":
            return lognormal_params_from_mean_sd(mean, sd)
        if likelihood == "Gamma":
            return gamma_params_from_mean_sd(mean, sd)
        if likelihood == "Weibull":
            return weibull_params_from_mean_sd(mean, sd)

    if prior_spec["type"] == "median_cov":
        median = prior_spec["median"]
        cov = prior_spec["cov"]

        if likelihood == "Lognormal":
            return lognormal_params_from_median_cov(median, cov)
        if likelihood == "Gamma":
            return gamma_params_from_median_cov(median, cov)
        if likelihood == "Weibull":
            return weibull_params_from_median_cov(median, cov)

    raise ValueError(f"Unsupported prior specification or likelihood: {prior_spec}, {likelihood}")


def reference_distribution_summary(prior_spec, likelihood):
    """Return moments of the distribution at the parameter centers.

    These values summarize the reference specification; they are not the
    moments of the induced hierarchical prior predictive distribution.
    """
    p1, p2 = get_prior_parameter_centres(prior_spec, likelihood)

    if likelihood == "Lognormal":
        mu, sigma = p1, p2
        mean = np.exp(mu + 0.5 * sigma ** 2)
        var = (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)

    elif likelihood == "Gamma":
        alpha, beta = p1, p2
        mean = alpha / beta
        var = alpha / (beta ** 2)

    elif likelihood == "Weibull":
        shape, scale = p1, p2
        g1 = gamma_func(1.0 + 1.0 / shape)
        g2 = gamma_func(1.0 + 2.0 / shape)
        mean = scale * g1
        var = scale ** 2 * (g2 - g1 ** 2)

    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    sd = np.sqrt(var)
    return mean, sd, sd / mean


# --------------------------------------------------
# Data helpers
# --------------------------------------------------
def standardize_facility(value):
    text = str(value).strip().lower()

    if "non" in text:
        return "Non-Essential"
    if "ess" in text:
        return "Essential"

    return str(value).strip()


def get_group_data(df, facility):
    df = df.copy()

    required_columns = {downtime_col, facility_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Required columns not found: {sorted(missing_columns)}")

    df[downtime_col] = pd.to_numeric(df[downtime_col], errors="coerce")
    df = df.dropna(subset=[downtime_col])
    df[downtime_col] = df[downtime_col].astype(float)
    df = df[df[downtime_col] > 0]

    df["_facility_clean"] = df[facility_col].apply(standardize_facility)

    valid_facilities = {"Essential", "Non-Essential"}
    unknown_facilities = set(df["_facility_clean"].dropna()) - valid_facilities
    if unknown_facilities:
        raise ValueError(
            f"Unrecognized facility labels: {sorted(unknown_facilities)}"
        )

    data = df.loc[df["_facility_clean"] == facility, downtime_col].values

    return np.asarray(data, dtype=float)


# --------------------------------------------------
# MCMC model fitting
# --------------------------------------------------
def fit_mcmc_model(data, likelihood, prior_spec):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < 5:
        raise ValueError("Too few observations for MCMC fitting.")

    p1_0, p2_0 = get_prior_parameter_centres(prior_spec, likelihood)

    with pm.Model() as model:

        if likelihood == "Lognormal":
            mu0, sigma0 = p1_0, p2_0

            mu = pm.Normal(
                "mu",
                mu=mu0,
                sigma=log_parameter_prior_sd
            )

            # Centred positive prior for lognormal sigma.
            # This is more controlled than a very diffuse half-normal.
            sigma = pm.LogNormal(
                "sigma",
                mu=np.log(sigma0),
                sigma=log_parameter_prior_sd
            )

            pm.LogNormal(
                "y",
                mu=mu,
                sigma=sigma,
                observed=data
            )

        elif likelihood == "Gamma":
            alpha0, beta0 = p1_0, p2_0

            alpha = pm.LogNormal(
                "alpha",
                mu=np.log(alpha0),
                sigma=log_parameter_prior_sd
            )

            beta = pm.LogNormal(
                "beta",
                mu=np.log(beta0),
                sigma=log_parameter_prior_sd
            )

            pm.Gamma(
                "y",
                alpha=alpha,
                beta=beta,
                observed=data
            )

        elif likelihood == "Weibull":
            shape0, scale0 = p1_0, p2_0

            shape = pm.LogNormal(
                "shape",
                mu=np.log(shape0),
                sigma=log_parameter_prior_sd
            )

            scale = pm.LogNormal(
                "scale",
                mu=np.log(scale0),
                sigma=log_parameter_prior_sd
            )

            pm.Weibull(
                "y",
                alpha=shape,
                beta=scale,
                observed=data
            )

        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True
        )

    return trace


# --------------------------------------------------
# Posterior summaries
# --------------------------------------------------
def posterior_predictive_samples_from_trace(trace, likelihood, rng):
    posterior = trace.posterior

    if likelihood == "Lognormal":
        mu = posterior["mu"].values.ravel()
        sigma = posterior["sigma"].values.ravel()
        samples = rng.lognormal(mean=mu, sigma=sigma)

    elif likelihood == "Gamma":
        alpha = posterior["alpha"].values.ravel()
        beta = posterior["beta"].values.ravel()
        samples = rng.gamma(shape=alpha, scale=1.0 / beta)

    elif likelihood == "Weibull":
        shape = posterior["shape"].values.ravel()
        scale = posterior["scale"].values.ravel()
        samples = scale * rng.weibull(a=shape, size=len(shape))

    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    return samples


def summarize_distribution_moments(trace, likelihood):
    posterior = trace.posterior

    if likelihood == "Lognormal":
        mu = posterior["mu"].values.ravel()
        sigma = posterior["sigma"].values.ravel()

        mean_samples = np.exp(mu + 0.5 * sigma ** 2)
        var_samples = (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)

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
        var_samples = alpha / (beta ** 2)

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
        var_samples = scale ** 2 * (g2 - g1 ** 2)

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
        "posterior_distribution_mean_cov": (
            np.std(mean_samples, ddof=1) / np.mean(mean_samples)
        ),
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


# --------------------------------------------------
# Main sensitivity analysis
# --------------------------------------------------
def main():
    posterior_rows = []
    diagnostic_rows = []

    settings_df = pd.DataFrame([{
        "sampler": "PyMC NUTS",
        "chains": chains,
        "cores": cores,
        "tune_per_chain": tune,
        "draws_per_chain": draws,
        "retained_posterior_draws_total": retained_draws_total,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "log_parameter_prior_sd": log_parameter_prior_sd,
        "prior_framework": "Empirical Bayes with sensitivity analysis",
        "primary_prior": "NZ empirical prior reference moments",
        "sensitivity_prior": "Established REDi reference information",
        "prior_interpretation": (
            "Reference summaries are converted to likelihood-parameter "
            "centers; uncertainty is specified on transformed scales."
        ),
    }])

    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)

        for facility in ["Essential", "Non-Essential"]:
            key = (sheet, facility)

            if key not in selected_likelihoods:
                continue

            likelihood = selected_likelihoods[key]
            data = get_group_data(df, facility)

            for prior_set_name, prior_dict in prior_sets.items():
                prior_spec = prior_dict[key]
                prior_mean, prior_sd, prior_cov = reference_distribution_summary(
                    prior_spec,
                    likelihood,
                )

                print("\n" + "=" * 80)
                print(f"Fitting {sheet} - {facility}")
                print(f"Prior set: {prior_set_name}")
                print(f"Likelihood: {likelihood}")
                print(f"N = {len(data)}")
                print(f"Observed mean = {np.mean(data):.3f}, observed SD = {np.std(data, ddof=1):.3f}")
                print(
                    "Reference mean = "
                    f"{prior_mean:.3f}, reference SD = {prior_sd:.3f}, "
                    f"reference COV = {prior_cov:.3f}"
                )

                trace = fit_mcmc_model(
                    data=data,
                    likelihood=likelihood,
                    prior_spec=prior_spec
                )

                moment_summary = summarize_distribution_moments(trace, likelihood)

                posterior_rows.append({
                    "Prior set": prior_set_name,
                    "Agency": sheet,
                    "Assessment type": assessment_type[sheet],
                    "Facility": facility,
                    "Likelihood": likelihood,
                    "N": len(data),

                    "Observed mean": np.mean(data),
                    "Observed SD": np.std(data, ddof=1),

                    "Reference prior mean": prior_mean,
                    "Reference prior SD": prior_sd,
                    "Reference prior COV": prior_cov,
                    "Prior implementation": (
                        "Reference summary converted to likelihood-parameter "
                        "centers; uncertainty specified on transformed scales."
                    ),
                    **moment_summary
                })

                diag = az.summary(trace, hdi_prob=0.95, round_to=4)
                diag = diag.reset_index().rename(columns={"index": "Parameter"})
                divergences = int(trace.sample_stats["diverging"].sum().item())
                bfmi = np.asarray(az.bfmi(trace), dtype=float)
                diag["Prior set"] = prior_set_name
                diag["Agency"] = sheet
                diag["Assessment type"] = assessment_type[sheet]
                diag["Facility"] = facility
                diag["Likelihood"] = likelihood
                diag["Divergences"] = divergences
                diag["Minimum BFMI"] = float(np.min(bfmi))
                diag["Chains"] = chains
                diag["Tune per chain"] = tune
                diag["Draws per chain"] = draws
                diag["Retained posterior draws"] = retained_draws_total
                diagnostic_rows.append(diag)

                az.plot_trace(trace)
                fig = plt.gcf()
                fig.suptitle(
                    f"{sheet} - {facility} - {likelihood} - {prior_set_name}",
                    y=1.02
                )
                fig.savefig(
                    f"trace_{sheet}_{facility.replace('-', '_')}_{likelihood}_{prior_set_name}.png",
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(fig)

    posterior_df = pd.DataFrame(posterior_rows)
    diagnostics_df = pd.concat(diagnostic_rows, ignore_index=True)

    # --------------------------------------------------
    # Compare posterior sensitivity across prior sets
    # --------------------------------------------------
    comparison_rows = []

    for key_cols, group in posterior_df.groupby(["Agency", "Facility", "Likelihood"]):
        if set(group["Prior set"]) != set(prior_sets.keys()):
            continue

        nz = group[group["Prior set"] == "NZ_empirical_prior"].iloc[0]
        redi = group[group["Prior set"] == "REDi_prior"].iloc[0]

        comparison_rows.append({
            "Agency": key_cols[0],
            "Facility": key_cols[1],
            "Likelihood": key_cols[2],

            "NZ posterior mean": nz["posterior_distribution_mean"],
            "REDi posterior mean": redi["posterior_distribution_mean"],
            "Absolute difference in posterior mean": (
                redi["posterior_distribution_mean"] - nz["posterior_distribution_mean"]
            ),
            "Percent difference in posterior mean": (
                100.0
                * (redi["posterior_distribution_mean"] - nz["posterior_distribution_mean"])
                / nz["posterior_distribution_mean"]
            ),

            "NZ posterior predictive mean": nz["posterior_predictive_mean"],
            "REDi posterior predictive mean": redi["posterior_predictive_mean"],
            "Absolute difference in predictive mean": (
                redi["posterior_predictive_mean"] - nz["posterior_predictive_mean"]
            ),
            "Percent difference in predictive mean": (
                100.0
                * (redi["posterior_predictive_mean"] - nz["posterior_predictive_mean"])
                / nz["posterior_predictive_mean"]
            ),

            "NZ posterior predictive COV": nz["posterior_predictive_cov"],
            "REDi posterior predictive COV": redi["posterior_predictive_cov"],
            "Absolute difference in predictive COV": (
                redi["posterior_predictive_cov"] - nz["posterior_predictive_cov"]
            ),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    with pd.ExcelWriter(output_file_path) as writer:
        settings_df.to_excel(writer, sheet_name="MCMC settings", index=False)
        posterior_df.to_excel(writer, sheet_name="Posterior by prior", index=False)
        comparison_df.to_excel(writer, sheet_name="Prior sensitivity", index=False)
        diagnostics_df.to_excel(writer, sheet_name="MCMC diagnostics", index=False)

    print("\nPrior sensitivity analysis saved to:")
    print(output_file_path)

    print("\nPrior sensitivity summary:")
    print(comparison_df)


if __name__ == "__main__":
    main()

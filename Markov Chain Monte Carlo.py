import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm, halfnorm
from joblib import Parallel, delayed

# -----------------------------
# Part 1: MH‐MCMC Parameter Update
# -----------------------------

def calculate_sd(mean, cov):
    """Calculate standard deviation from mean and Coefficient of Variation (COV)."""
    return mean * cov

def plot_lognormal_distribution(ax, mean, sd, label, color):
    """Plot a lognormal distribution on the given axis."""
    sigma = np.sqrt(np.log(1 + (sd / mean) ** 2))
    scale = mean / np.exp(sigma ** 2 / 2)
    x = np.linspace(0.001, mean + 4 * sd, 1000)
    y = lognorm.pdf(x, s=sigma, scale=scale)
    ax.plot(x, y, label=label, color=color)

def mh_mcmc(prior_mu, prior_sigma, data, iterations=10000, burn_in=2000):
    """
    Perform Metropolis–Hastings MCMC to estimate lognormal parameters.
    
    Parameters:
        prior_mu (float): Prior estimate of the log-mean.
        prior_sigma (float): Prior estimate of the log-standard deviation.
        data (array-like): Observed inspection downtime data.
        iterations (int): Total number of iterations.
        burn_in (int): Number of initial samples to discard.
        
    Returns:
        final_mu, final_sigma: Posterior estimates after burn-in.
    """
    samples_mu = []
    samples_sigma = []
    current_mu = prior_mu
    current_sigma = prior_sigma

    for _ in range(iterations):
        # Propose new values for mu and sigma
        proposed_mu = np.random.normal(current_mu, 0.05)
        proposed_sigma = np.random.lognormal(mean=np.log(current_sigma), sigma=0.05)

        # Compute log-likelihoods under current and proposed parameters
        log_likelihood_current = -0.5 * np.sum(((np.log(data) - current_mu) / current_sigma) ** 2) - len(data) * np.log(current_sigma)
        log_likelihood_proposed = -0.5 * np.sum(((np.log(data) - proposed_mu) / proposed_sigma) ** 2) - len(data) * np.log(proposed_sigma)

        delta = log_likelihood_proposed - log_likelihood_current
        acceptance_ratio = np.exp(np.clip(delta, -700, 700))

        # Accept proposed parameters with appropriate probability
        if np.random.rand() < acceptance_ratio:
            current_mu = proposed_mu
            current_sigma = proposed_sigma

        samples_mu.append(current_mu)
        samples_sigma.append(current_sigma)

    # Discard burn-in samples and compute the posterior estimates
    final_mu = np.mean(samples_mu[burn_in:])
    final_sigma = np.mean(samples_sigma[burn_in:])
    return final_mu, final_sigma

def process_inspection_data(file_path):
    """
    Process inspection downtime data for multiple agencies using MH‐MCMC.
    
    Reads data from an Excel file with separate sheets for each agency,
    groups the data by facility type, updates lognormal parameters using MCMC,
    and saves the results to an Excel file.
    """
    # Define sheet names corresponding to each agency
    agencies = ['CERA', 'CCC', 'NHC']

    # Define prior parameters for each agency and facility type (placeholder values for CCC and NHC)
    prior_parameters = {
        'CERA': {
            'Essential': {'mu': 6.55045275664473, 'sigma': 0.316665663260573},
            'Non-Essential': {'mu': 6.59198219697841, 'sigma': 0.308649201388166}
        },
        'CCC': {
            'Essential': {'mu': 6.55045275664473, 'sigma': 0.316665663260573},
            'Non-Essential': {'mu': 6.59198219697841, 'sigma': 0.308649201388166}
        },
        'NHC': {
            'Essential': {'mu': 6.55045275664473, 'sigma': 0.316665663260573},
            'Non-Essential': {'mu': 6.59198219697841, 'sigma': 0.308649201388166}
        }
    }

    results = []

    for agency in agencies:
        df = pd.read_excel(file_path, sheet_name=agency)
        df.columns = df.columns.str.strip()

        if "Facility" not in df.columns or "Inspection Downtime" not in df.columns:
            raise ValueError(f"Missing required columns 'Facility' or 'Inspection Downtime' in sheet {agency}")

        grouped_data = df.groupby("Facility")
        for facility_type in ["Essential", "Non-Essential"]:
            if facility_type in grouped_data.groups:
                group = grouped_data.get_group(facility_type)
                sample_values = group["Inspection Downtime"].values

                prior_mu = prior_parameters[agency][facility_type]['mu']
                prior_sigma = prior_parameters[agency][facility_type]['sigma']

                posterior_mu, posterior_sigma = mh_mcmc(prior_mu, prior_sigma, sample_values)
                results.append({
                    "Agency": agency,
                    "Facility": facility_type,
                    "Prior Mu (log)": prior_mu,
                    "Prior Sigma (log)": prior_sigma,
                    "Posterior Mu (log)": posterior_mu,
                    "Posterior Sigma (log)": posterior_sigma
                })

    results_df = pd.DataFrame(results)
    output_file_path = "MCMC_Lognormal_InspectionDowntime_AllAgencies.xlsx"
    results_df.to_excel(output_file_path, index=False)
    print(f"MH-MCMC updated parameters saved to '{output_file_path}'")
    return results_df

# -----------------------------
# Part 2: Marginal Posterior Analysis
# -----------------------------

def marginal_distribution_lognormal_optimised(mu_posterior, sigma_posterior, duration_min, duration_max, num_durations,
                                              num_mus, num_sigmas, range_factor=2):
    """
    Compute the marginal distribution of a lognormal duration variable.
    
    Parameters:
        mu_posterior (float): Posterior log-mean.
        sigma_posterior (float): Posterior log-standard deviation.
        duration_min (float): Minimum duration value.
        duration_max (float): Maximum duration value.
        num_durations (int): Number of points in the duration grid.
        num_mus (int): Number of points in the mu grid.
        num_sigmas (int): Number of points in the sigma grid.
        range_factor (float): Factor to define the range for mu and sigma grids.
        
    Returns:
        Dictionary with marginal mean and standard deviation.
    """
    duration_vector = np.linspace(duration_min, duration_max, num_durations)
    mu_grid = np.linspace(mu_posterior - range_factor * sigma_posterior,
                          mu_posterior + range_factor * sigma_posterior, num_mus)
    sigma_grid = np.linspace(max(0.01, sigma_posterior - range_factor * sigma_posterior),
                             sigma_posterior + range_factor * sigma_posterior, num_sigmas)

    mu_prior_pdf = norm.pdf(mu_grid, loc=mu_posterior, scale=sigma_posterior)
    sigma_prior_pdf = halfnorm.pdf(sigma_grid, scale=sigma_posterior)
    joint_prior = np.outer(mu_prior_pdf, sigma_prior_pdf)

    duration_matrix, mu_matrix, sigma_matrix = np.meshgrid(duration_vector, mu_grid, sigma_grid, indexing='ij')
    conditional_pdf = lognorm.pdf(duration_matrix, s=sigma_matrix, scale=np.exp(mu_matrix))

    marginal_pdf = np.tensordot(conditional_pdf, joint_prior, axes=([1, 2], [0, 1]))
    marginal_pdf /= np.trapz(marginal_pdf, duration_vector)

    mean_duration = np.trapz(duration_vector * marginal_pdf, duration_vector)
    variance_duration = np.trapz((duration_vector - mean_duration) ** 2 * marginal_pdf, duration_vector)
    std_duration = np.sqrt(variance_duration)

    return {"mean_duration": mean_duration, "std_duration": std_duration}

def process_posterior(variable, facility, params, duration_min, duration_max, num_durations, num_mus, num_sigmas, range_factor):
    """
    Process the posterior data for a given agency (variable) and facility.
    
    Returns a dictionary with the posterior parameters and the computed marginal distribution statistics.
    """
    mu_posterior = params["Mu"]
    sigma_posterior = params["Sigma"]

    marginal_results = marginal_distribution_lognormal_optimised(
        mu_posterior=mu_posterior,
        sigma_posterior=sigma_posterior,
        duration_min=duration_min,
        duration_max=duration_max,
        num_durations=num_durations,
        num_mus=num_mus,
        num_sigmas=num_sigmas,
        range_factor=range_factor,
    )

    return {
        "Variable": variable,
        "Facility": facility,
        "Posterior Mu (log)": mu_posterior,
        "Posterior Sigma (log)": sigma_posterior,
        "Marginal Mean": marginal_results["mean_duration"],
        "Marginal SD": marginal_results["std_duration"],
    }

def process_posterior_data(posterior_data):
    """
    Process the posterior data for all inspection agencies using parallel computation.
    
    Exports the optimised marginal posterior results to an Excel file.
    """
    # Parameters for marginal analysis
    duration_min = 0.01
    duration_max = 5000
    num_durations = 1000
    num_mus = 200
    num_sigmas = 200
    range_factor = 2

    results = Parallel(n_jobs=-1)(
        delayed(process_posterior)(
            variable, facility, params, duration_min, duration_max, num_durations, num_mus, num_sigmas, range_factor
        )
        for variable, facilities in posterior_data.items()
        for facility, params in facilities.items()
    )

    results_df = pd.DataFrame(results)
    output_file = "Marginal_Posterior_Results_Lognormal_Optimised.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"Optimised marginal posterior results saved to '{output_file}'")
    return results_df

def build_posterior_data(mh_results):
    """
    Build a posterior data dictionary for marginal analysis based on the MH-MCMC results.
    
    The resulting dictionary has the structure:
        { "Agency-Inspection": { "Facility": {"Mu": <posterior mu>, "Sigma": <posterior sigma>} } }
    """
    posterior_data = {}
    for _, row in mh_results.iterrows():
        agency_key = f"{row['Agency']}-Inspection"
        facility = row["Facility"]
        if agency_key not in posterior_data:
            posterior_data[agency_key] = {}
        posterior_data[agency_key][facility] = {
            "Mu": row["Posterior Mu (log)"],
            "Sigma": row["Posterior Sigma (log)"]
        }
    return posterior_data

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    # Update the Excel file path as needed
    excel_file_path = 'Paper 6-data.xlsx'
    
    # Part 1: Process inspection data using MH-MCMC
    mh_results = process_inspection_data(excel_file_path)
    
    # Build the posterior_data dictionary from the MH-MCMC results
    posterior_data = build_posterior_data(mh_results)
    
    # Part 2: Process posterior data for marginal distributions using the updated parameters
    posterior_results = process_posterior_data(posterior_data)

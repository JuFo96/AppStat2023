import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# Function to generate random numbers from Rayleigh distribution


def generate_rayleigh(sigma, size):
    return sigma * np.sqrt(-2 * np.log(1 - np.random.rand(size)))


def rayleigh(x, sigma):
    return x/sigma**2 * np.exp(-0.5*x**2/sigma**2)

# Define the negative log-likelihood function for Minuit


def neg_log_likelihood(sigma):
    return -np.sum(np.log(rayleigh(random_numbers, sigma)))


# Parameters
sigma = 2
N = 1000


N_values = np.arange(50, 5001)
sigma_uncertainties_minuit = []

for n in N_values:
    # Generate random numbers
    random_numbers = generate_rayleigh(sigma, n)

    # Create a Minuit object
    minuit = Minuit(neg_log_likelihood, sigma=2.0)

    # Perform the fit
    minuit.migrad()

    # Get the fitted parameters
    sigma_fit_minuit = minuit.values['sigma']
    sigma_uncertainty_minuit = minuit.errors['sigma']

    sigma_uncertainties_minuit.append(sigma_uncertainty_minuit)

# Plot histogram
plt.hist(random_numbers, bins=30, label='Generated Data, bins = 30')
plt.title('Generated Data with transformation method')
plt.ylabel("Count")
plt.legend()
plt.savefig("plots/generated_hist_rayleigh.png", bbox_inches="tight")
plt.show()

# Plot the true Rayleigh distribution
x = np.linspace(0, 10, 1000)

# Plot the results
plt.plot(N_values, sigma_uncertainties_minuit, 'b--',
         label='Fitted Sigma Uncertainty')
plt.plot(N_values, sigma / np.sqrt(N_values),
         'r--', label='$1/\sqrt{N}$ Scaling')

plt.title('Uncertainty scaling')
plt.xlabel('Sample Size')
plt.ylabel('Sigma Fit Uncertainty')
plt.legend()
plt.savefig("plots/sigma_scaling.png", bbox_inches="tight")
plt.show()

# Initial guess for the parameter
initial_sigma_guess = 2.0

# Create a Minuit object
minuit = Minuit(neg_log_likelihood, sigma=initial_sigma_guess)

# Perform the fit
minuit.migrad()

# Get the fitted parameters
sigma_fit_minuit = minuit.values['sigma']

# Plot the fitted distribution
plt.hist(random_numbers, bins=30, density=True,
         alpha=0.7, label='Generated Data')
plt.plot(x, rayleigh(x, sigma_fit_minuit), 'g',
         label='Fitted Rayleigh Distribution')
plt.title("Generated distribution and fit")
plt.ylabel("Probability")
plt.savefig("plots/rayleigh_fit.png", bbox_inches="tight")


print(f"True Sigma: {sigma}, Fitted Sigma (Minuit): {sigma_fit_minuit}")

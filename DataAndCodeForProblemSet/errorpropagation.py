# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:47:18 2024

@author: Julius
"""

from sympy import symbols, diff
import numpy as np
import scipy
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.umath as umath

x = 1.043
x_sigma = 0.014
y = 0.07
y_sigma = 0.23
rho = 0.4

z1 = x*y*np.exp(-y)
z1_diff_x = y*np.exp(-y) * x_sigma
z1_diff_y = (x*np.exp(-y) - x*y*np.exp(-y)) * y_sigma
z1_sigma_corr = np.sqrt(z1_diff_x**2 * x_sigma**2
                        + z1_diff_y**2 * y_sigma**2
                        + 2 * z1_diff_x * z1_diff_y * x_sigma * y_sigma * rho)

z2 = (y+1)**3/(x-1)
z2_diff_y = 3 * (y+1)**2 / (x-1)
z2_diff_x = -(y+1)**3 / (x-1)**2
z2_sigma_corr = np.sqrt(z2_diff_x**2 * x_sigma**2
                        + z2_diff_y**2 * y_sigma**2
                        + 2 * z2_diff_x * z2_diff_y * x_sigma * y_sigma * rho)


def z2_func(x, y):
    return (y+1)**3/(x-1)


def z1_func(x, y):
    x*y*np.exp(-y)


x_1 = np.linspace(-2, 2, 1000)
y_1 = np.linspace(-10, 90, 1000)

#plt.plot(x_1, z1_func())

############
# 2.2
############

i = 2
measurements = np.array([5.50, 5.61, 4.88, 5.07, 5.26])
uncertainties = np.array([0.10, 0.21, 0.15, 0.14, 0.13])
#measurements = np.delete(measurements, i)
#uncertainties = np.delete(uncertainties, i)

weights = 1 / uncertainties**2

# Calculate combined result and uncertainty
combined_result = np.sum(weights * measurements) / np.sum(weights)
combined_uncertainty = np.sqrt(1 / np.sum(weights))

print("Combined Result: {:.3f} g/cm^3".format(combined_result))
print("Combined Uncertainty: {:.3f} g/cm^3".format(combined_uncertainty))

# Calculate chi-square value
chi_square = np.sum(((measurements - combined_result) / uncertainties)**2)
ndof = len(measurements) - 1
chi_square_red = chi_square / (len(measurements) - 1)
prob = scipy.stats.chi2.sf(chi_square, ndof)

print(f"Probability of chi2 value from 1-CDF: {prob:.3f}")
print("Chi-square value:", chi_square)
print("Reduced Chi-square value:", chi_square_red)

# Given precise value
precise_value = 5.514

# Calculate deviation
deviation = np.abs(combined_result - precise_value) / combined_uncertainty

print("Deviation from Precise Value:", deviation)


######
# 2.3
#####

a = 1.04
e = 0.71
a_sigma = 0.27
e_sigma = 0.12
A1 = np.pi * a ** 2 * np.sqrt(1-e**2)
A1_diff_a = 2 * np.pi * a * np.sqrt(1-e**2)
A1_diff_e = -np.pi * a ** 2 * 1/np.sqrt(1-e**2) * e

A1_sigma_corr = np.sqrt(A1_diff_a**2 * a_sigma**2
                        + A1_diff_e**2 * e_sigma**2)

a1 = unc.ufloat(1.04, 0.27)
e1 = unc.ufloat(0.71, 0.12)

A1_unc = np.pi * a1**2 * umath.sqrt(1-e1**2)

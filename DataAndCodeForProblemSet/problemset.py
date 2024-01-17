# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:05:17 2024

@author: Julius
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2)) * np.exp(-0.5*((x - mu)/sigma)**2)


def binomial(N_attempts, n_success, p_winning):
    return np.math.factorial(N_attempts) / (np.math.factorial(
        n_success) * np.math.factorial(N_attempts - n_success)) * p_winning**n_success * (1-p_winning)**(N_attempts - n_success)

####################################
# 1. Distributions and probability #
####################################


mu = 50
sigma = 20
x = np.linspace(-3*sigma + mu, 3*sigma + mu, 1000)
cdf = scipy.stats.norm.cdf(x, mu, sigma)
plt.plot(x, gaussian(x, mu, sigma))

frac = cdf[640] - cdf[540]

N_attempts = 20
n_success = 15
p_winning = 12/37
x_success = np.arange(0, 21)
p_binomial = [binomial(N_attempts, n, p_winning) for n in x_success]
p_greater_than_8 = np.sum(p_binomial[8:])

plt.figure()
plt.title("Binomial distribution with N=20 attempts and p=12/37")
plt.ylabel("Probability")
plt.xlabel("Number of successes")
plt.bar(x_success, p_binomial)
plt.bar(x_success[8:], p_binomial[8:], color="orange")
plt.savefig("plots/binomial.png", bbox_inches="tight")
print(
    f"The probability of getting 8 or more successful attempts out of {N_attempts} is {np.sum(p_binomial[8:]):.2f}")

####################################
# 2. Error Propagation             #
####################################

"""
##########
SCRATCHPAD
##########

n_points = 1000
count = 0

for _ in range(n_points):
    uniform_number = np.random.uniform(-1000, 1000)
    if uniform_number > gaussian(uniform_number, mu, sigma):
        count += 1
        
area = count / n_points
print(f"area is {area}")
"""

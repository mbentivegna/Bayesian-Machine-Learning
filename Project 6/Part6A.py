#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 6
#Sampling Methods

#Part 1 - Rejection Samping

"""
    This program attempts to sample a gaussian mixture model using a basic
rejection sampling algorithm.  Each sample is chosen randomly and independently using the q(x)
equation before being placed into the p(x) function and comparing p(x) with a value chosen
uniformly between 0 and k*q(x). This comparison will determine if the sample is rejected or kept
and the entire process is repeated 10000 times.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Ground truth
mu = [3, 7, 11]
cov = [1, 1, 1]

x = np.linspace(0, 14, 1000)

#Equation 11.13
def p(x):
    return (stats.norm.pdf(x, mu[0], cov[0]) + stats.norm.pdf(x, mu[1], cov[1]) + stats.norm.pdf(x, mu[2], cov[2]))/3

#Make q a normal curve
def q(x):
    return stats.norm.pdf(x, 7, 3)

#Ensure q is greater than p at all points
k = max(p(x)/q(x))

#Rejection sampling decision algorithm to determine whether or not to keep the sample
sample = []
for i in range(10000):
    z = np.random.normal(7, 3)
    u = np.random.uniform(0, (k*q(z)))

    #Equation 11.14
    if u <= p(z):
        sample.append(z)

#Plotting
plt.figure()
plt.plot(x, p(x), color="red", label = "p(x)")
plt.plot(x, k*q(x), color="blue", label = "k*q(x)")
plt.hist(sample, np.linspace(0,14,50), density=True, color="green", label = "Samples")
plt.legend()
plt.xlabel("Sample Values (x)")
plt.ylabel("Normalized Quantity of Samples")
plt.title("Rejection Sampling")
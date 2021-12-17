#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 7
#Gibbs Sampling

"""
This project aims to recreate the Gibbs Sampling paper,
which describes the methods taken to perform sampling on a posterior
joint distributions given a generative model set of random variables.
We use the PyMC3 package to model and perform the computations.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pymc3 as pm

#Declarations
a = 2
b = 1
N = 50
n = 26

sampleTotal = 5200
samplesTossed = 200

#Generative modeling parameters
lambda_1 = np.random.gamma(a, 1/b)
lambda_2 = np.random.gamma(a, 1/b)*5
lambda_concat = [lambda_1]*n + [lambda_2]*(N-n)
x_1 = np.random.poisson(lambda_1, n)
x_2 = np.random.poisson(lambda_2, N-n)
x_origin = np.concatenate((x_1,x_2))

#Using pymc3 to get samples
model = pm.Model()

with model:
    
    #Equation 4 Values
    lambda1 = pm.Gamma('lamb1', a, b)
    lambda2 = pm.Gamma('lamb2', a, b)
    n_value = pm.DiscreteUniform('n', 0, N)
    
    domain = np.arange(0, N)
    lambda_prime = pm.math.switch(n_value > domain, lambda1, lambda2)

    x = pm.Poisson('x1',lambda_prime, observed=x_origin)
    samples = pm.sample(sampleTotal, return_inferencedata=False)

#Throw out first 200 samples
burned_samples = samples['n', samplesTossed:]
gibbs_lambda_1 = samples['lamb1', samplesTossed:]
gibbs_lambda_2 = samples['lamb2', samplesTossed:]

#For mean line
mean_1 = np.round(sum(samples['n'])/len(samples['n']), 3)
mean_2 = np.round(sum(samples['lamb1'])/len(samples['lamb1']), 3)
mean_3 = np.round(sum(samples['lamb2'])/len(samples['lamb2']), 3)

#Plotting to match the Gibbs Sampling paper
def plot():
    fig, ax = plt.subplots(5,1, figsize=[10,10])

    ax[0].stem(range(N), x_origin, use_line_collection=True)
    ax[0].plot(range(N), lambda_concat, color='red', linestyle='dashed')
    ax[0].set_ylabel('Counts')
    ax[0].set_ylim([0, 16])

    x_space = np.linspace(0, len(gibbs_lambda_1), len(gibbs_lambda_2))
    ax[1].plot(x_space, gibbs_lambda_1, color='blue')
    ax[1].plot(x_space, gibbs_lambda_2, color='green')
    ax[1].set_ylabel("λ")

    ax[2].hist(gibbs_lambda_1, range=(0,12), color='blue', bins=200)
    ax[2].axvline(mean_2, linestyle='dashed')

    ax[3].hist(gibbs_lambda_2, range=(0,12), color='green', bins=200)
    ax[3].axvline(mean_3, linestyle='dashed')

    ax[4].hist(burned_samples, range=(0,50), color='blue', bins=200)    
    ax[4].axvline(mean_1, linestyle='dashed')
    ax[4].set_xlabel("n")
plot()
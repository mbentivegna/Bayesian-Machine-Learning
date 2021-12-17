#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 6
#Sampling Methods

#Part 2 - MCMC

"""
    This project uses the Markov Chain Monte Carlo algorithm to properly create a new sample 
distribution of two weights. The final chain has 1000 samples, but the first 100 are rejected
in order to avoid factoring in values as the chain works it's way towards the mean. The output
of this program displays two graphs showing the weight distributions and their corresponding 
averages.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Ground truth
a0 = -.3
a1 = .5
sd = .1

#Produce random samples with gaussian noise
xN = np.zeros(25)
tN = np.zeros(25)

for i in range(25):
  xN[i] = np.random.uniform(-1, 1)
  tN[i] = a0 + a1*xN[i] + np.random.normal(0, sd)

#Calculate the posterior distribution of weight vector
def posterior(weights):
    #Equation 3.10
    likelihood = np.sum(np.log(stats.norm.pdf(tN, weights[0]+weights[1]*xN, sd)))
    prior = np.log(stats.norm(-.2,0.6).pdf(weights[0])) * np.log(stats.norm(-.2,0.6).pdf(weights[1]))
    posterior = likelihood + prior
    return posterior

#Get a new zStar value
def proposal(weights):
    return [np.random.normal(weights[0], 0.15), np.random.normal(weights[1], 0.15)]

#Create z chain vector and make initial guess
zFinal = np.zeros([1000, 2])
zFinal[0, :] = [-.2, .6]

#MCMC Decision algorithm to determine whether to keep the sample
i=1
while i < 1000:
    zTau = zFinal[i-1, :]
    zStar = proposal(zTau)
    
    u2 = np.random.uniform(0, 1)

    #Equation 11.33
    prob = np.exp(posterior(zStar)-posterior(zTau))

    #Accepted path (i is only incremented in this case)
    if prob>u2:
        zFinal[i,:] = zStar
        i+=1

#Plotting
fig, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].axvline(np.mean(zFinal[100:,0]), color="green", linestyle="dashed", label = "Mean")
ax[0].hist(zFinal[100:,0], bins=50, color="gray", label = "Samples")
ax[0].legend()
ax[0].set_xlabel("Sample Values for a0")
ax[0].set_ylabel("Quantity of Samples")
ax[0].set_title("MCMC Sampling for a0")

ax[1].axvline(np.mean(zFinal[100:,1]), color="red", linestyle="dashed", label = "Mean")
ax[1].hist(zFinal[100:,1], bins=50, color="gray", label = "Samples")
ax[1].legend()
ax[1].set_xlabel("Sample Values for a1")
ax[1].set_ylabel("Quantity of Samples")
ax[1].set_title("MCMC Sampling for a1")
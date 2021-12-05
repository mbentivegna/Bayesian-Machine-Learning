#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 5
#Expectation Maximization

#Part 1 - 1D

"""
  This project uses the expectation-maximization (EM) algorithm to model a
gaussian mixture's individual components. The 1D data points are displayed in
a histogram and the estimated normal distribution of each K-mean is superimposed.
Each of the four subplots show the model after a given amount of passes through 
the algorithm.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Ground truths
mu_1 = 1
mu_2 = 7
mu_3 = 13
cov = 1
pi = 1/3

#Initial guesses
mu_init = [.5, 6, 12]
cov_init = [1.5, 1.5, 1.5]
pi_init = [0.4, 0.25, 0.35]

#Observation generation
data = np.zeros([3,100])
data[0, :] = np.random.normal(mu_1, cov, 100)
data[1, :] = np.random.normal(mu_2, cov, 100)
data[2, :] = np.random.normal(mu_3, cov, 100)
merged_data = np.concatenate((data[0,:],data[1,:],data[2,:]))

#Subplot initialization
fig, ax = plt.subplots(2, 2, figsize=(15,15))
for axis in ax.flat:
  axis.set(xlabel='μ', ylabel='normalized #')

#Function for plotting histogram and superimposed normal curve
def plot(i,j,mu,cov, iteration):
    x = np.linspace(-2, 16, 1000)
    ax[i,j].set_title(f'Expectation after #{iteration} iterations')

    ax[i,j].hist(data[0,:], color='black', density=True)
    ax[i,j].hist(data[1,:], color='black', density=True)
    ax[i,j].hist(data[2,:], color='black', density=True)
    
    ax[i,j].plot(x, stats.norm.pdf(x, mu[0], cov[0]), linewidth=2)
    ax[i,j].plot(x, stats.norm.pdf(x, mu[1], cov[1]), linewidth=2)
    ax[i,j].plot(x, stats.norm.pdf(x, mu[2], cov[2]), linewidth=2)

#EM Steps
def em(mu,cov,pi,merged_data):
    #Equation 9.23
    gam = np.zeros([3,300])
    gam[0,:] = pi[0]*stats.norm.pdf(merged_data, mu[0], cov[0])
    gam[1,:] = pi[1]*stats.norm.pdf(merged_data, mu[1], cov[1])
    gam[2,:] = pi[2]*stats.norm.pdf(merged_data, mu[2], cov[2])
    
    gam_sum = gam[0,:]+gam[1,:]+gam[2,:]

    gam[0,:] /= gam_sum
    gam[1,:] /= gam_sum
    gam[2,:] /= gam_sum

    #Loop K times to get the values for each grouping
    for i in range(3):
        mu[i] = 1/np.sum(gam[i,:]) * np.sum(gam[i,:] * merged_data) #Equation 9.24
        cov[i] = 1/np.sum(gam[i,:]) * np.sum(gam[i,:]*(merged_data-mu[i])*(merged_data-mu[i])) #Equation 9.25
        pi[i] = np.sum(gam[i,:])/300 #Equation 9.26

    return mu, cov, pi

mu_new = np.copy(mu_init)
cov_new = np.copy(cov_init)
pi_new = np.copy(pi_init)

#Mission control
def part_1(mu_new, cov_new, pi_new, merged_data):
  for i in range(16):
      if i==0: plot(0,0,mu_new,cov_new,i)
      if i==5: plot(0,1,mu_new,cov_new,i)
      if i==10: plot(1,0,mu_new,cov_new,i)
      if i==15: plot(1,1,mu_new,cov_new,i)
      mu_new, cov_new, pi_new = em(mu_new, cov_new, pi_new, merged_data) #Call function each loop with new parameters

part_1(mu_new, cov_new, pi_new, merged_data)
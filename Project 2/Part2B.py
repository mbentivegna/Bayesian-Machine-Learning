#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 2 
#Linear Regression

#Main Project
#Reproduce Figure 3.8

""" 
      Part 2 attempts to create a line of best fit for the line y = sin(2*pi*x).  Initially taking samples using the ground truth
line with additional gaussian noise, a predictive distribution was plotted using a linear combination of specific gaussian basis
functions.  The red line on the graph represents the mean of the predictive distribution and the shaded red region shows 1 SD on
either side of it.  Similar to part 1, as the number of samples increases, the predictive line becomes both more accurate and
precise.

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Base declarations
N = 25
alpha = 2.0
beta = 25
sd = (1/beta)**0.5
s_0 = alpha**-1*np.identity(9)

#Ground truth
x = np.linspace(0, 1, 1000)
y = np.sin(2*np.pi*x)
mean_0 = np.zeros([9,1])

#Generate 25 samples
x_N = [0]*25
t_N = [0]*25
for i in range(25):
  noise = np.random.normal(0, sd)
  x_N[i] = random.uniform(0,1)
  t_N[i] = np.sin(2*np.pi*x_N[i]) + noise

#Design matrix (iota)
mu_j = np.linspace(-1,1,9)
iota = np.zeros((25,9))
for i in range(iota.shape[0]):
  for j in range(iota.shape[1]):
    #Eqn 3.4 Gaussian basis function
    iota[i, j] = np.exp(-(x_N[i]-mu_j[j])**2/(2*sd**2))

#Instantiate subplot graphs
fig, ax = plt.subplots(2, 2, figsize=(15,15), constrained_layout=True)
for axis in ax.flat:
  axis.set(xlabel='x', ylabel='t')

#Determine phi(x) vector
def phi_calc(x,phi):
  for j in range(len(phi)):
    #Eqn 3.4 Gaussian basis function
    phi[j] = np.exp(-(x-mu_j[j])**2/(2*sd**2))
  return phi

#Graph predictive distribution for specified # of samples
def predictive_distribution(i, j, x_count):
  #x_count many samples to utilize
  iota_n = iota[:x_count]
  #Eqn 3.51, gives 9x9 covariance matrix
  inv_s_0 = np.linalg.inv(s_0)
  inverse_covar_n = inv_s_0 + beta*(np.matmul(np.transpose(iota_n),iota_n))
  covar_n = np.linalg.inv(inverse_covar_n)
  #Eqn 3.50, gives 9x1 mean vector
  mean_n = np.matmul(covar_n,(np.matmul(inv_s_0,mean_0) + beta*np.matmul(np.transpose(iota_n),np.transpose(np.atleast_2d(t_N[:x_count])))))

  #Find phi for 1000 values from 0 - 1
  pred_mean = np.zeros(1000)
  pred_sd = np.zeros(1000)
  phi = np.zeros(9)
  for k in range(len(x)):
    #Basis function, gives 1x9 phi vector
    phi = phi_calc(x[k], phi)
    #Eqn 3.58, predicted mean
    pred_mean[k] = np.matmul(np.transpose(mean_n),phi)
    #Eqn 3.59, predicted standard deviation
    pred_sd[k] = np.sqrt(1/beta + np.matmul(np.matmul(phi,covar_n),np.transpose(phi)))

  #Standard deviation bounds
  sd_up = pred_mean + pred_sd
  sd_down = pred_mean - pred_sd

  #Graphing 
  ax[i,j].set_title(f'Predicted Mean and SD {x_count} sample(s)')
  #Sinusoid ground truth
  ax[i,j].plot(x, y, c="g")
  #Data points
  ax[i,j].scatter(x_N[:x_count], t_N[:x_count], s=40, facecolors='none', edgecolors='b')
  #Predicted mean
  ax[i,j].plot(x, pred_mean, c="r")
  #Predicted standard deviation bounds
  ax[i,j].fill_between(x, sd_up, sd_down, alpha=0.3, interpolate = True, color = 'r')

#Call prediction function for 1, 2, 3, and 25 samples
def part2():
  x_count = [1,2,4,25]
  count = 0
  for i in range(2):
    for j in range(2):
      predictive_distribution(i,j,x_count[count])
      count+=1
    
part2()
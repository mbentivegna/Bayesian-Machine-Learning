#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 4
#Gaussian Processes and Kernel Methods

#Main Project
#Reproduce Figure 3.8

""" 
Using the same data as in project 3, this program attempts to map a line of best fit 
using kernel methods.  For 4 different number of samples, a graph was created that showcased
this line along with one standard deviation of the prediction on either side.  As the number
of samples increased, so did the accuracy of the predictions as well as the tightness of the 
standard deviation areas.

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Base declarations
N = 25
beta = 25
sd = (1/beta)**0.5

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

#Instantiate subplot graphs
fig, ax = plt.subplots(2, 2, figsize=(15,15), constrained_layout=True)
for axis in ax.flat:
  axis.set(xlabel='x', ylabel='t')

#Kernel function for cleaner calculations
def kernel(x, x_prime):
    #Equation 6.23
    temp = np.exp((-(np.absolute(x-x_prime))**2) / (2*(1/beta)))
    return temp

#Graph predictive distribution for specified # of samples
def gaussian_processes(plot_i, plot_j, x_count):

  x_spliced = np.atleast_2d(x_N[:x_count]).T
  t_spliced = np.atleast_2d(t_N[:x_count]).T
  cov_matrix = np.zeros((x_count,x_count))

  for i in range(x_count):
    for j in range(x_count):
      if i==j:
        #Equation 6.62
        cov_matrix[i,j] = kernel(x_spliced[i,0], x_spliced[j,0]) + (beta**-1)
      else:
        cov_matrix[i,j] = kernel(x_spliced[i,0], x_spliced[j,0])
  
  kernel_func = kernel(x_spliced, np.atleast_2d(np.linspace(0,1,1000)))
  #Equation 6.66
  pred_mean = kernel_func.T @ np.linalg.inv(cov_matrix) @ t_spliced

  #Equation 6.67
  c = 1+beta**-1
  pred_cov = c - np.atleast_2d(np.diag(kernel_func.T @ np.linalg.inv(cov_matrix) @ kernel_func))
  pred_sd = np.sqrt(pred_cov)

  #Standard deviation bounds
  pred_mean = pred_mean.T
  sd_up = pred_mean + pred_sd
  sd_down = pred_mean - pred_sd


  #Graphing 
  ax[plot_i,plot_j].set_title(f'Predicted Mean and SD {x_count} sample(s)')
  #Sinusoid ground truth
  ax[plot_i,plot_j].plot(x, y, c="g")
  #Data points
  ax[plot_i,plot_j].scatter(x_spliced, t_spliced, s=40, facecolors='none', edgecolors='b')
  #Predicted mean
  ax[plot_i,plot_j].plot(x, pred_mean.flatten(), c="r")
  #Predicted standard deviation bounds
  ax[plot_i,plot_j].fill_between(x, sd_up.flatten(), sd_down.flatten(), alpha=0.3, interpolate = True, color = 'r')


#Call prediction function for 1, 2, 3, and 25 samples
def main_project():
  x_count = [1,2,4,25]
  count = 0
  for i in range(2):
    for j in range(2):
      gaussian_processes(i,j,x_count[count])
      count+=1
    
main_project()
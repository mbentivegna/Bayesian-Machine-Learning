#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 2 
#Linear Regression

#Main Project
#Reproduce Figure 3.7

""" 
      Part 1 of this project illustrates Bayesian learning for a linear regression model.  Using samples from the ground truth line 
y = -.3 + .5x (with generated noise), the likelihood function, posterior distribution, and data space were plotted 4 times each with
for a differing number of trials. As the number of trials increased, the posterior distribution tightens near the correct value and
the data space line converge to the ground truth.  The likelihood function on the left only take in account the most recent sample 
and is what updates the posterior distribution and dataspace each time.

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Ground truth
a0 = -.3
a1 = .5

SD = .2
beta = (1 / SD)**2
alpha = 2

xN = [0]*20
tN = [0]*20

#Produce random samples with gaussian noise
iota = []
for i in range(20):
  xN[i] = random.uniform(-1, 1)
  tN[i] = a0 + a1*xN[i] + np.random.normal(0, SD)
  iota.append([1,xN[i]])

#Instantiating the 4 by 3 subplots
fig, ax = plt.subplots(4, 3, figsize=(15,15), constrained_layout=True)
w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((w0, w1))

#Function to calculate and graph data based on specified number of samples
def plot_parameter_dist(x_num, run_num):

  #--------Posterior Distribution Graphs----------
  #Calculate the sample mean array and covariance matrix
  if x_num == 0:
    mean_n = [0,0]
    covar_n = alpha**-1*np.identity(2)
  else:  
    sub_iota = iota[:x_num]
    #Equation 3.54
    inverse_covar_n = alpha*np.identity(2) + beta*np.matmul(np.transpose(sub_iota),sub_iota)
    covar_n = np.linalg.inv(inverse_covar_n)
    #Equation 3.53
    t = np.transpose(tN[:x_num])
    mean_n = beta*np.matmul(np.matmul(covar_n,np.transpose(sub_iota)),t)
  post_dist = multivariate_normal(mean_n, covar_n)
  
  #Plot posterior distribution
  ax[run_num,1].set_title(f'Posterior {x_num} sample(s)')
  ax[run_num,1].set(xlabel='w_0',ylabel='w_1')
  ax[run_num,1].contourf(w0,w1,post_dist.pdf(pos))
  ax[run_num,1].plot(a0, a1, marker='+', markeredgecolor='w')

  #--------Data Space Graphs----------
  #Using posterior distribution select 6 random [w0, w1] combinations and plot them
  linear_model_w = np.random.multivariate_normal(mean_n, covar_n, 6)
  for w in linear_model_w:
    x = np.linspace(-1,1,1000)
    y = [0]*1000
    for i in range(len(y)):
      y[i] = w[0] + w[1]*x[i]    
    #Plot the 6 lines
    ax[run_num,2].plot(x,y)

  #Plot the sample data points selected
  for j in range(x_num):
    x_point = xN[j]
    y_point = tN[j]
    ax[run_num,2].set_title(f'Data Space {x_num} sample(s)')
    ax[run_num,2].set(xlabel='x',ylabel='y')
    ax[run_num,2].plot(x_point, y_point, marker='o', markeredgecolor='b', markerfacecolor='None')

  #--------Likelihood Graphs----------
  if run_num != 0:
    wT = [w0, w1]
    iotaL = [1, xN[x_num-1]]
    z = (np.sqrt(2*np.pi*SD**2))**-1*np.exp(-(tN[x_num-1] - np.matmul(np.transpose(wT),iotaL))**2 / (2*SD**2))
    ax[run_num,0].set_title(f'Likelihood {x_num} sample(s)')
    ax[run_num,0].set(xlabel='w_0',ylabel='w_1')
    ax[run_num,0].contourf(w0,w1,z)
    ax[run_num,0].plot(a0, a1, marker='+', markeredgecolor='w')

#Calls previous function for 0, 1, 2, and 20 samples
def part1():
  pos = np.dstack((w0, w1))
  x_count = [0, 1, 2, 20]
  for i in range(len(x_count)):
    x_num = x_count[i]
    plot_parameter_dist(x_num, i)

part1()
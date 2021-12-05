#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 1 
#Gaussian with Known Variance (i.e. estimate the mean)

#Imports
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

#Declarations
n = 101
mu = np.linspace(-1.0,1.0,num=n)

#Ground Truth
mean = 0
var = 0.2

#Hyperparameters
mean_o1 = 0.7
var_o1 = 0.05

mean_o2 = 0.1
var_o2 = 0.01

mean_o3 = 5
var_o3 = 1

ml_error = np.zeros(n)
conj_error1 = np.zeros(n)
conj_error2 = np.zeros(n)
conj_error3 = np.zeros(n)

#Creat n samples using actual mean and variance
sample = np.random.normal(mean,np.sqrt(var),n)

#Get updated values and graph new posterior plot
def gaussian_posterior(i):
  #Equation 2.141
  mu_n = (var)/(i*var_o1+var)*mean_o1 + (i*var_o1)/(i*var_o1+var)*(np.sum(sample[:i])/(i))
  #Equation 2.142
  var_n = ((1/var_o1)+(i/var))**-1
  #Equation 2.140
  post_density = norm.pdf(mu, mu_n, np.sqrt(var_n))
  
  plt.figure()
  plt.plot(mu, post_density, color="blue")
  plt.xlabel("μ")
  plt.ylabel("Magnitude")
  plt.title(f"Posterior Density of Known Variance for {i} Observations(s)")
  plt.show()

#Get mu_n for each guess parameters after a given number of samples
#Then find MSE for graphing
def gaussian_error(i):
  ml_sq_e = mu_sq_e1 = mu_sq_e2 = mu_sq_e3 = 0  

  for j in range(1000):
    e_sample = np.random.normal(mean,np.sqrt(var),n)

    #Equation 2.143
    ml = (np.sum(e_sample[:i])/i)
    #Equation 2.141
    mu_1 = (var)/(i*var_o1+var)*mean_o1 + (i*var_o1)/(i*var_o1+var)*ml
    mu_2 = (var)/(i*var_o2+var)*mean_o2 + (i*var_o2)/(i*var_o3+var)*ml
    mu_3 = (var)/(i*var_o3+var)*mean_o3 + (i*var_o2)/(i*var_o3+var)*ml

    ml_sq_e += (ml - mean)**2
    mu_sq_e1 += (mu_1 - mean)**2
    mu_sq_e2 += (mu_2 - mean)**2
    mu_sq_e3 += (mu_3 - mean)**2
  
  ml_error[i-1] = ml_sq_e/1000
  conj_error1[i-1] = mu_sq_e1/1000
  conj_error2[i-1] = mu_sq_e2/1000
  conj_error3[i-1] = mu_sq_e3/1000

#Loop to call functions accordingly
for i in range(1,n+1):
  gaussian_error(i)
  if i%25 == 1:
    gaussian_posterior(i)

#Mean squared error plot for ML and conjugates
observations = np.linspace(1, n, num=n)
plt.figure(figsize=(10, 10))
plt.plot(observations, ml_error, color = 'black', label = "ML Error")
plt.plot(observations, conj_error1, color = 'red', label = f"Conj Error mean_o = {mean_o1} var_o = {var_o1}")
plt.plot(observations, conj_error2, color = 'green', label = f"Conj Error mean_o = {mean_o2} var_o = {var_o2}")
plt.plot(observations, conj_error3, color = 'blue', label = f"Conj Error mean_o = {mean_o3} var_o = {var_o3}")
plt.xlabel("# of trials")
plt.ylabel("MSE")
plt.title("Mean Squared Error for Known Variance")
plt.legend(loc="upper right")
plt.show()

"""
Summary: Given the ground truth of mean = 0 and variance = 0.2, we can take a 
sample from a gaussian distribution and estimate the mean with a Gaussian 
Posterior Density Function given the variance. It shows that the density which
was initially centered around mean_o = 0.7 and estimated mean's variance = 0.05
shifts over to the true mean = 0 over 101 observations. The density also becomes
narrower as the confidence in the mean's value increases.

The MSE plot shows that as the mean_o's guess is further away from the ground 
truth mean, the MSE is large, but over many observations, it asymptotically 
reaches 0. Similarly, the ML MSE also reaches 0 which makes sense since it's an 
average of all thesample values of a gaussian distribution that is centered 
around mean = 0.
"""
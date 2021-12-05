#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 1 
#Extra Credit - Gaussian with unkown mean and unknown variance

from scipy.stats import norm
from scipy.stats import invgamma 
import numpy as np
import matplotlib.pyplot as plt

n = 101

#Ground Truth
mean = 0
var = 1

#Estimates
mean_o = 0
beta = 2
a = 5
b = 6

sample = np.random.normal(mean, np.sqrt(var), n)
#3-D Grid Initialization
mu =  np.linspace(-2.0,2.0,n)
lmb = np.linspace(0,2.0,n)
M, L = np.meshgrid(mu, lmb, indexing='ij')
Z = np.zeros_like(M)

#Updates hyperparameters using proper equations
#Posterior graph plotting
def gaussian_gamma_posterior(i):
  #hyperparameter update equations derived from
  #https://en.wikipedia.org/wiki/Normal_distribution#With_unknown_mean_and_unknown_variance
  #and
  #http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-4/04-normal-gamma.pdf
  #slides 3-10. Used because it uses the inv-gamma for update like the textbook.

  #gaussian hyperparameters
  x_n = (np.sum(sample[:i]))/i
  beta_n = beta+i
  mean_n = (beta*mean_o + i*x_n) / (beta_n)

  #inv-gamma hyperparameters
  a_n = a + (i/2)
  sqrd_dev = 0
  for j in range(i):
    sqrd_dev += (sample[i-1]-x_n)**2
  b_n = b + 0.5*sqrd_dev + ((i*beta)/(2*beta_n))*((x_n-mean_o)**2)

  #gaussian posterior
  var_n = var/beta
  gaussian = norm.pdf(mu, mean_n, np.sqrt(var_n))

  #inv-gamma posterior
  gam = invgamma.pdf(lmb, a_n, 0, b_n)

  gaussian = np.atleast_2d(gaussian).T
  gam = np.atleast_2d(gam)
  #Equation 2.154
  gauss_gam = gaussian@gam
  
  #plotting
  plt.figure()
  plt.contour(M, L, gauss_gam)
  plt.xlabel("μ")
  plt.ylabel("λ")
  plt.title(f"Heatmap of Posterior DF of Unknown Mean and Variance over {i} Observation(s)")
  plt.show()
  
for i in range(1,n+1):
  if i%25==1:
    gaussian_gamma_posterior(i)

"""
Summary: Given the ground truths at mean = 0 and variance = 0, we can see that
the estimated mean and variance is represented by the heatmap. As the number of
observations increases, it sometimes overshoots but eventually returns and
tightens the boundaries of the shown heatmap towards the desired values.
"""
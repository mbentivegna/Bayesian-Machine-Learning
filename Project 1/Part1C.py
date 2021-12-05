#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 1 
#Gaussian with Known Mean (i.e. estimate the variance)

#Imports
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from scipy.stats import gamma

#Declarations
lamda = np.linspace(-1, 3, num=1001)

#Ground Truth
n = 101
mu = 0
var = 2

#Estimates
a0 = 1
b0 = 1

a1 = 20
b1 = 30

a2 = 5
b2 = 1

MLerror = np.zeros(n)
conjError1 = np.zeros(n)
conjError2 = np.zeros(n)
conjError3 = np.zeros(n)

#Find n number of samples from actual mean and variance
sample = np.random.normal(mu,np.sqrt(var),n)
numSamples = np.linspace(1, n, num = n)

#Updates hyperparameters using proper equations
#Posterior graph plotting
def gaussianUnknownVariance(i):
  #Equation 2.150
  aN = a0 + i/2
  
  #Equation 2.151
  total = 0
  for j in range(0,i):
    total = total + (sample[j] - mu)**2

  bN = b0 + (.5)*total
  #Equation 2.146
  y = gamma.pdf(lamda, aN, loc = 0, scale = 1 / bN)

  plt.figure()
  plt.plot(lamda, y, color = 'black')
  plt.xlabel("Precision (λ)")
  plt.ylabel("Magnitude")
  plt.title("Posterior Density Function for " + str(i) + " samples")
  plt.show()

#Gets variance for error function
def varForError(a, b, i, e_sample):
  #Equation 2.150
  aF = a + i/2
  
  #Equation 2.151
  total = 0
  for j in range(i):
    total = total + (e_sample[j] - mu)**2
  bF = b + (.5)*total
  varr = bF/aF

  return varr

#Error function for MSE graphs
def error(i):
  varML_sq_e = var_sq_e1 = var_sq_e2 = var_sq_e3 = 0

  for j in range(1000):
    e_sample = np.random.normal(mu,np.sqrt(var),n)
    varML = 0
    for k in range(i):
      varML += (e_sample[k] - mu)**2
    varML = varML/i

    varML_sq_e += (varML - var)**2
    var_sq_e1 += (varForError(a0, b0, i, e_sample) - var)**2
    var_sq_e2 += (varForError(a1, b1, i, e_sample) - var)**2
    var_sq_e3 += (varForError(a2, b2, i, e_sample) - var)**2

  MLerror[i-1] = varML_sq_e/1000
  conjError1[i-1] = var_sq_e1/1000
  conjError2[i-1] = var_sq_e2/1000
  conjError3[i-1] = var_sq_e3/1000

#Main control loop
for i in range(1,n):
  if i % 25 == 1:
    gaussianUnknownVariance(i)

  error(i)

#MSE plot creation
plt.figure(figsize=(10, 10))
plt.plot(numSamples, MLerror, color = 'black', label = "ML Error")
plt.plot(numSamples, conjError1, color = 'red', label = "Conj Error a = 1 b = 1")
plt.plot(numSamples, conjError2, color = 'blue', label = "Conj Error a = 20 b = 30")
plt.plot(numSamples, conjError3, color = 'green', label = "Conj Error a = 5 b = 1")
plt.xlabel("# of trials")
plt.ylabel("MSE")
plt.title("Mean Squared Error for Different Estimators")
plt.legend(loc="upper right")
plt.show()

"""
Summary: Using the ground truth mean = 0 and variance = 2, we wanted
to graphically show the estimated precision as samples increased.  Using the
gamma distribution and continutously updating hyperparameters, the posterior density
curve was graphed every 40 samples. Graphically, it can clearly be show that the
precision goes to approximately .5 which is the inverse of the variance.

The MSE plot, as in the other experiments, appeared to exponentially decay as the
number of samples increased.  Also seen in the graph is the influence of the hyperparameters
at the start of the trial.  Specifically, bad hyperparameters led to large MSE at the beginning 
but eventually leveled out with the other errors as time went on.
"""

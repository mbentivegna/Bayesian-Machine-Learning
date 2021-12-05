#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 1 
#Binomial Distribution

#Imports
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

#Declarations
x = np.linspace(0.1, 1.0, num=1000)

n = 101
m = 0

#Estimates
a1 = 12
b1 = 16

a2 = 1
b2 = 2

a3 = 300
b3 = 400

MLerror = np.zeros(n)
conjError1 = np.zeros(n)
conjError2 = np.zeros(n)
conjError3 = np.zeros(n)

tosses = np.linspace(1, n, num = n)

#Function to graph posterior density functions
def posteriorDensity(x, a1, m, b1, l):
    postDense = beta.pdf(x, a1 + m, b1 + l)

    #posterior plot
    plt.figure()
    plt.plot(x, postDense, color = 'black')
    plt.xlabel("μ")
    plt.ylabel("Magnitude")
    plt.title("Posterior Density Function for " + str(i) + " samples")
    plt.show()

#Function to fill error arrays for graphing
def error(i):
  ml_sq_e = conj_sq_e1 = conj_sq_e2 = conj_sq_e3 = 0

  for j in range(1000):
    head = 0
    for k in range(i):
      flip = random.randint(0, 1)
      if flip == 1:
        head += 1
    tail = i-head
    ml_sq_e += (head/i - .5)**2
    conj_sq_e1 += ((head + a1) / (head + a1 + tail + b1) - .5)**2
    conj_sq_e2 +=((head + a2) / (head + a2 + tail + b2) - .5)**2
    conj_sq_e3 += ((head + a3) / (head + a3 + tail + b3) - .5)**2

  MLerror[i-1] = ml_sq_e/1000
  conjError1[i-1] = conj_sq_e1/1000
  conjError2[i-1] = conj_sq_e2/1000
  conjError3[i-1] = conj_sq_e3/1000
    
#Loops through to get different number of samples and call proper functions
for i in range(1,n):
  error(i)

  k = random.randint(0, 1)
  if k == 1:
    m += 1
  l = i - m

  if i % 25 == 1:
    posteriorDensity(x, a1, m, b1, l)
    
#Mean squared error plot for ML and conjugates
plt.figure(figsize=(10, 10))
plt.plot(tosses, MLerror, color = 'black', label = "ML Error")
plt.plot(tosses, conjError1, color = 'red', label = "Conj Error a = 12 b = 16")
plt.plot(tosses, conjError2, color = 'green', label = "Conj Error a = 1 b = 2")
plt.plot(tosses, conjError3, color = 'blue', label = "Conj Error a = 300 b = 400")
plt.xlabel("# of trials")
plt.ylabel("MSE")
plt.title("Mean Squared Error for Different Estimators")
plt.legend(loc="upper right")
plt.show()

"""
Summary: This part of the experiment modelled how flipping a coin could create a
posterior distribution. With an even likelihood of 0 or 1, 200 random samples were
created.  As that was going on, every 40 samples the posterior distribution was graphed
and became more accurate as the number of samples increased.  The effect of the sub-par
hyperparameters was ultimately outweighed  by the shear volume of new samples.

The MSE showcases a similar story, as bad hyperparamters as well as the ML model were 
severly inaccurate to start with, but ultimately approached zero as the number of 
trials increased
"""
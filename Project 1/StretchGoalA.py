#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 1 
#Extra Credit Movies (Runs in VSCode -- attached video in submission folder)

import numpy as np
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
mpl.use('TkAgg')

#Binomial Movie
def movie(): 
    plt.ion()
    fig = plt.figure()
    plt.title('Binomial Distribution Movie')
    plt.xlabel('Mean (µ)')
    plt.ylabel('Magnitude')
    plt.ylim((0, 50)) 
    plt.xlim((0,1))

    x = np.linspace(0.01, 1.00,1000)
    a0 = 30
    b0 = 40

    line, = plt.plot(x, beta.pdf(x, a0, b0))

    for i in range(0, 2000, 20):
        m = 0
        for k in range(i):
            number = random.randint(0, 1)
            if number == 1:
                m += 1

        y = beta.pdf(x, a0 + m, b0 + (i-m))
        
        plt.legend(['N = ' + str(i)])
        line.set_ydata(y)
        line.set_xdata(x)
        fig.canvas.draw()
        fig.canvas.flush_events()

#movie()

#Gaussian Unknown Mean Movie
def movie2(mu0, var0, var): 

    plt.ion()
    fig = plt.figure()
    plt.title('Gaussian Unknown Mean Movie')
    plt.xlabel('Mean (µ)')
    plt.ylabel('Magnitude')
    plt.ylim((0, 200)) 
    plt.xlim((-.6,0.6))

    x = np.linspace(-.99,0.99,1000)
    line, = plt.plot(x, norm.pdf(x, mu0, np.sqrt(var0)))

    for i in range(1, 100, 2):
        data = np.random.normal(mu0, np.sqrt(var0), i)
        tmp = 0
        for k in range(0,i):
            tmp+=data[k]
        muML = tmp/i

        #Update Equations
        muN = var/(i*var0 + var)*mu0 + (i*var0)/(i*var0 + var)*muML
        varN = 1/((1/var0) + (i/var)) 
        
        y = norm.pdf(x, muN, varN)
        
        plt.legend(['N = ' + str(i)])
        line.set_ydata(y)
        line.set_xdata(x)
        fig.canvas.draw()
        fig.canvas.flush_events()


#movie2(.1, .05, .2)

#Gaussian Unknown Variance Movie
def movie3(a0, b0, mu, var): 

    plt.ion()
    fig = plt.figure()
    plt.title('Gaussian Unknown Variance Movie')
    plt.xlabel('Precision (λ)')
    plt.ylabel('Magnitude')
    plt.ylim((0, 15)) 
    plt.xlim((0,1))

    n = 500
    sample = np.random.normal(mu,np.sqrt(var),n)

    x = np.linspace(0, 5, 1000)
    line, = plt.plot(x, gamma.pdf(x, a0, 0, 1 / b0))

    for i in range(1, 500, 5):        
        aN = a0 + i/2
  
        total = 0
        for j in range(0,i):
            total = total + (sample[j] - mu)**2

        bN = b0 + (.5)*total
        y = gamma.pdf(x, aN, loc = 0, scale = 1 / bN)
        
        plt.legend(['N = ' + str(i)])
        line.set_ydata(y)
        line.set_xdata(x)
        fig.canvas.draw()
        fig.canvas.flush_events()


#movie3( 1, 1, 0, 2)
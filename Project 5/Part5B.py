#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 5
#Expectation Maximization

#Part 2 - 2D

"""
  This program implements the same model as part 1 except for 2D datasets.  Thus,
a multivariate gaussian distribution was utilized and a scatterplot was chosen to
display the data points.  The old faithful dataset was chosen for clustering purposes
where K = 2.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Plot initialization
fig_2, ax_2 = plt.subplots(2, 3, figsize=(14,12))
for axis in ax_2.flat:
  axis.set(xlabel='Eruption Time', ylabel='Waiting Time (Scaled by 1/30)')

#Process data from txt file
N = 272
full_data = np.zeros([N, 2])

fd = open('old_faithful.txt')
text = fd.readlines()
for i in range(N):
    x = text[i].split()
    full_data[i,0] = float(x[0]) #Eruption Time
    full_data[i,1] = float(x[1])/30 #Waiting Time for next Eruption

#Initial Guesses
mu2_init = np.array([[2, 3], [5, 2.5]]) #Each row is it's own guess
cov2_init1 = np.array([[1, 0],[0, 1]])
cov2_init2 = np.array([[1, 0],[0, 1]])
pi2_init = [.5, .5]

#Plotting function
def plot_2_d(i,j, mu, cov1, cov2, gamma, iteration):
    #Get proper color for each data point and plot it
    for k in range(len(full_data)):
        if iteration != 0:
            class_choice = np.argmax(gamma[k,:])
            if class_choice == 1:
                ax_2[i,j].plot(full_data[k,0], full_data[k,1], marker='o', color = 'blue', zorder = 0)
            else:
                ax_2[i,j].plot(full_data[k,0], full_data[k,1], marker='o', color = 'red', zorder = 0)
        else:
            ax_2[i,j].plot(full_data[k,0], full_data[k,1], marker='o', color = 'black', zorder = 0)

    #Plot estimated distribution
    xy = np.mgrid[1:6:0.01, 1:6:0.01]
    x = xy[0]
    y = xy[1]
    xy = np.dstack((x,y))
    class_1 = stats.multivariate_normal.pdf(xy, mu[0,:], cov1)
    class_2 = stats.multivariate_normal.pdf(xy, mu[1,:], cov2)
    ax_2[i,j].contour(x, y, class_1, levels = 3, zorder = 3)
    ax_2[i,j].contour(x, y, class_2, levels = 3, zorder = 4)
    ax_2[i,j].set_title(f'L = {iteration} iterations')

#EM steps
def em_2_d(mu, cov1, cov2, pi, data):
    #Equation 9.23
    gam_1 = pi[0]*stats.multivariate_normal.pdf(data, mu[0], cov1)
    gam_2 = pi[1]*stats.multivariate_normal.pdf(data, mu[1], cov2) 

    gam_sum = gam_1+gam_2
    gam_1 = np.atleast_2d(gam_1/gam_sum)
    gam_2 = np.atleast_2d(gam_2/gam_sum)
    gamma = np.concatenate((gam_1.T, gam_2.T), axis=1)
    
    #Equation 9.24
    mu[0, :] = 1/np.sum(gam_1, axis = 1) * np.sum(data * gam_1.T, axis = 0)
    mu[1, :] = 1/np.sum(gam_2, axis = 1) * np.sum(data * gam_2.T, axis = 0)

    #Equation 9.25
    cov1 = (gam_1*(data-mu[0, :]).T@(data-mu[0, :]))/np.sum(gam_1, axis = 1)
    cov2 = (gam_2*(data-mu[1, :]).T@(data-mu[1, :]))/np.sum(gam_2, axis = 1)

    #Equation 9.26
    pi[0] = np.sum(gam_1, axis=1)/272
    pi[1] = np.sum(gam_2, axis=1)/272

    return mu, cov1, cov2, pi, gamma

mu2_new = np.copy(mu2_init)
cov2_new1 = np.copy(cov2_init1)
cov2_new2 = np.copy(cov2_init2)
pi2_new = np.copy(pi2_init)

#Mission control
def part_2(mu2_new, cov2_new1, cov2_new2, pi2_new, full_data):
    for i in range(21):
        if i==0: plot_2_d(0,0, mu2_init, cov2_init1, cov2_init2, None, i)
        if i==1: plot_2_d(0,1, mu2_new, cov2_new1, cov2_new2, gamma_new, i)
        if i==2: plot_2_d(0,2, mu2_new, cov2_new1, cov2_new2, gamma_new, i)
        if i==5: plot_2_d(1,0, mu2_new, cov2_new1, cov2_new2, gamma_new, i)
        if i==10: plot_2_d(1,1, mu2_new, cov2_new1, cov2_new2, gamma_new, i)
        if i==20: plot_2_d(1,2, mu2_new, cov2_new1, cov2_new2, gamma_new, i)
        mu2_new, cov2_new1, cov2_new2, pi2_new, gamma_new = em_2_d(mu2_new, cov2_new1, cov2_new2, pi2_new, full_data)

part_2(mu2_new, cov2_new1, cov2_new2, pi2_new, full_data)
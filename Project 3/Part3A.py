#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 3
#Linear Classification

#Main Project
#1. Percent Correctly Classified
#2. Plot ROC curve
#3. Plot Decision Boundary
#4. Test on Kaggle/UCI ML dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import csv

#Ground truth
n = 1000
split = 0.5

mean_0 = np.array([1,1])
mean_1 = np.array([-1,-1])
sd = np.identity(2)

#Generate each class' values
class_0_n = int(n*split)
class_1_n = int(n*(1-split))
class_0_vals = np.random.multivariate_normal(mean_0, sd, class_0_n)
class_1_vals = np.random.multivariate_normal(mean_1, sd, class_1_n)

#Data processing for the function call
class_0 = np.zeros(class_0_n)
class_1 = np.ones(class_1_n)

class_0_set = np.append(class_0_vals.transpose(), np.atleast_2d(class_0), axis=0)
class_1_set = np.append(class_1_vals.transpose(), np.atleast_2d(class_1), axis=0)

class_set = np.append(class_0_set, class_1_set, axis=1)

""" 
    This part creates 1000 total data points (500 from each class) and processes them to fit the parameters of our generated functions.  Both
functions should produce high accuracy results as the data is segregated sufficiently.  Once this data has been correctly
classified by both functions, an imported dataset will be processed and tested. 
"""

#Gaussian Generative Model

def gaussian_generative(set_n, class_0_n, class_0_set, class_1_n, class_1_set, class_set, window, run_type):
  #----Find w and w0 values----
  #eqn 4.73
  pi = class_0_n/(class_0_n+class_1_n)

  #eqn 4.75, 4.76
  mu_0_x = np.sum((1-class_set[2,:])*class_set[0,:])/class_0_n
  mu_0_y = np.sum((1-class_set[2,:])*class_set[1,:])/class_0_n
  mu_0 = np.array([mu_0_x, mu_0_y])
  mu_1_x = np.sum(class_set[2,:]*class_set[0,:])/class_0_n
  mu_1_y = np.sum(class_set[2,:]*class_set[1,:])/class_0_n
  mu_1 = np.array([mu_1_x, mu_1_y])

  #eqn 4.79 4.80
  mu_0_s1 = np.array([np.ones(class_0_n)*mu_0[0], np.ones(class_0_n)*mu_0[1]])
  mu_1_s2 = np.array([np.ones(class_1_n)*mu_1[0], np.ones(class_1_n)*mu_1[1]])
  S1 = ((class_0_set[0:2, :] - mu_0_s1) @ (class_0_set[0:2, :] - mu_0_s1).T)/class_0_n
  S2 = ((class_1_set[0:2, :] - mu_1_s2) @ (class_1_set[0:2, :] - mu_1_s2).T)/class_1_n

  #eqn 4.78
  S = (class_0_n/set_n)*S1 + (class_1_n/set_n)*S2

  #eqn 4.69
  w = np.linalg.inv(S)@(mu_0 - mu_1)
  #eqn 4.70
  w0 = (-1/2)*(mu_0)@np.linalg.inv(S)@(mu_0).T + (1/2)*(mu_1)@np.linalg.inv(S)@(mu_1).T + np.log(pi/(1-pi))

  #----Percent correct calculations----
  a = w.T@class_set[:2,:] + w0
  #eqn 4.59
  sigmoid = 1/(1+np.exp(-a))
  threshold = 0.5
  count = 0

  for i in range(set_n):
      if class_set[2,i] == 1 and sigmoid[i] < .5:
          count += 1
      if class_set[2,i] == 0 and sigmoid[i] > .5:
          count += 1 

  print(f'Percent correct = {count/set_n*100}%')

  #----Decision boundary graph----
  x = np.linspace(window[0], window[1], 1000)
  #eqn 4.4
  y = -w[0]/w[1]*x + w0

  fig, ax = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
  ax[0].set_title(f'{run_type} Gaussian Generative Decision Boundary of {set_n} Samples')
  ax[0].scatter(class_0_set[0, :], class_0_set[1, :], marker='o')
  ax[0].scatter(class_1_set[0, :], class_1_set[1, :], marker='x')
  ax[0].plot(x, y)
  ax[0].set_xlabel("x1")
  ax[0].set_ylabel("x2")

  #----ROC curve plotting----
  tp_rate, fp_rate, threshold = roc_curve(class_set[2,:],sigmoid, pos_label = 1)
  center = np.linspace(0,1,1000)
  ax[1].set_title(f'{run_type} ROC Curve of {set_n} Samples')
  ax[1].plot(fp_rate,tp_rate)
  ax[1].plot(center,center)
  ax[1].set_xlabel("False Positive Rate")
  ax[1].set_ylabel("True Positive Rate")

gaussian_generative(n, class_0_n, class_0_set, class_1_n, class_1_set, class_set, [-4,4], 'Test')

""" 
This function takes a gaussian generative approach to classifying the two datasets. Using the listed equations, the w vector and w0 value were calculated.  The decision line
was then plotted along with the data points to show the efficacy of the classification.  The percent correct was then calculated and the ROC curve was generated to quantitatively
display the success of the gaussian generative model. With the percent correct being consistently at or over 92%, the model was deemed successfully implemented.
"""

#Logistic Regression Classifier with IRLS

def logistic_regression(set_n, class_0_set, class_1_set, class_set, window, run_type):
  #Set up phi matrix
  phi = np.zeros([3, set_n])
  phi[0, :] = np.ones([1, set_n])
  phi[1, :] = class_set[0, :]
  phi[2, :] = class_set[1, :]

  #----Find w_old iteratively----
  w_old = np.array([np.ones(1),np.ones(1),np.ones(1)])

  #IRLS
  for i in range(100):
      #eqn 4.105
      a = w_old.T@phi
      a = a/100
      #eqn 4.59
      y = 1/(1+np.exp(-a))
      #eqn 4.102
      R = np.zeros((set_n,set_n))
      for j in range(set_n):
          R[j,j] = y[0, j]*(1-y[0, j])
      #eqn 4.100
      Z = (phi.T@w_old) - (np.linalg.inv(R)@(y - class_set[2,:]).T)
      #eqn 4.99
      w_old = (np.linalg.inv(phi@R@phi.T))@phi@R@Z

  #----Percent correct calculations----
  a = w_old.T @ phi / 100
  sigmoid = 1/(1+np.exp(-a))
  count = 0
  for i in range(set_n):
      if class_set[2,i] == 1 and sigmoid[0,i] > .5:
          count += 1
      if class_set[2,i] == 0 and sigmoid[0,i] < .5:
          count += 1 
  print(f'Percent correct = {count/set_n*100}%')

  #----Decision boundary plotting----
  x = np.linspace(window[0],window[1],set_n)
  #eqn 4.4
  y = -w_old[1]/w_old[2]*x - w_old[0]/w_old[2]

  fig, ax = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
  ax[0].set_title(f'{run_type} Logistic Regression Decision Boundary of {set_n} Samples')
  ax[0].scatter(class_0_set[0, :], class_0_set[1, :], marker='o')
  ax[0].scatter(class_1_set[0, :], class_1_set[1, :], marker='x')
  ax[0].plot(x, y)
  ax[0].set_xlabel("x1")
  ax[0].set_ylabel("x2")
  
  #----ROC curve----
  tp_rate, fp_rate, threshold = roc_curve(class_set[2,:], sigmoid.flatten(), pos_label = 0)
  center = np.linspace(0,1,1000)
  ax[1].set_title(f'{run_type} ROC Curve of {set_n} Samples')
  ax[1].plot(fp_rate,tp_rate)
  ax[1].plot(center,center)
  ax[1].set_xlabel("False Positive Rate")
  ax[1].set_ylabel("True Positive Rate")
  
logistic_regression(n, class_0_set, class_1_set, class_set, [-4,4], 'Test')


"""
The logistic regression approach had to be done iteratively in order to produce the correct w vector.  One hundred iterations were done
to optimize for time complexity and decision line accuracy.  Similar to the gaussian generative function, this also produces the correctness
rate and ROC curve.  The percent correct ended up being very similar to that of the previous model. 
"""

#Open csv and get data into a 3 by 100 matrix
file = open('test_data.csv')
csvreader = csv.reader(file)
n2 = 100

rows = []
for row in csvreader:
  rows.append(row)
  
rows.pop(0)
data = np.array(rows)

data = data.T.astype(float)
data[[0, 2]] = data[[2, 0]]

#Get the classification of each data point
class0 = np.array([[],[],[]])
class0_n = 0
class1 = np.array([[],[],[]])
class1_n = 0
for i in range(n2):
  if data[2, i] == 0:
    class0 = np.append(class0,np.atleast_2d(data[:, i]).T, axis=1)
    class0_n += 1
  if data[2, i] == 1:
    class1 = np.append(class1,np.atleast_2d(data[:, i]).T, axis=1)
    class1_n += 1

#Call each of the functions
gaussian_generative(n2, class0_n, class0, class1_n, class1, data, [-.5, .5], 'Titanic Deaths')
logistic_regression(n2, class1, class0, data, [-.2, .2],'Titanic Deaths')

"""
This is a titanic data set where the x1 axis is the fare paid for the trip, the x2 value is the age of the passenger, and the binary classifier.  The data was preprocessed similarly
to the test dataset and ultimately was used by the functions to determine if age and fare can be used to correctly classify the people that survived the journey.  The correct outcome
was only able to be predicted approximately two-thirds of the time. Thus, the model was subpar but was still able to produce results that were better than simply guessing.
"""
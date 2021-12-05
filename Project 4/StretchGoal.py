#ECE-474 Bayesian ML Prof. Keene
#Michael Bentivegna
#Donghyun Park
#Bayesian Project 4
#Gaussian Processes and Kernel Methods

#Extra Credit

"""
This program implements the Practical Guide to Support Vector Classification paper using
built in python packaging.  A new dataset was chosen that was more easily classifiable than 
the one chosen for the previous project.  The model was trained using the built in data and a grid
search to choose the most optimal model for accuracy.  The percent of data that was correctly 
classified is then displayed along with other key values.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

wine = load_wine()

#The data set is presented in a dictionary form:
features = pd.DataFrame(wine['data'], columns = wine['feature_names'])

#Wine column is our target
targets = pd.DataFrame(wine['target'], columns =['Wine'])

X_train, X_test, y_train, y_test = train_test_split(features, np.ravel(targets), test_size = 0.30, random_state = 0)

#Defining the parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, scoring='accuracy')

#Fitting the model for grid search
grid.fit(X_train, y_train)

#Other C and gamma parameters
means = grid.cv_results_["mean_test_score"]
sds = grid.cv_results_["std_test_score"]
for mean, sd, params in zip(means, sds, grid.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, sd * 2, params))
print("\nBest Parameters:")
print(grid.best_params_)
print("")

grid_predictions = grid.predict(X_test)
 
#Print classification report
print(classification_report(y_test, grid_predictions))
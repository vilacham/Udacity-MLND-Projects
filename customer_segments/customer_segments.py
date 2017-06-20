# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:07:37 2017

@author: Matheus Vilachã
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allows the use of display() for DataFrames
from IPython.display import display

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    msg = "Wholesale customers dataset has {} samples with {} features each.\n"
    print msg.format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Display a description of the dataset
display(data.describe())

data_to_plot = [data['Frozen'], data['Milk'], data['Grocery'], data['Frozen'], 
                data['Detergents_Paper'], data['Delicatessen']]
plt.boxplot(data_to_plot, 0, '')

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [127, 298, 371]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "\nChosen samples of wholesale customers dataset:"
display(samples)

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

features = list(data.columns)

for feature in features:
    # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the
    # given feature
    new_data = data.drop([feature], axis = 1)
    
    # TODO: Split the data into training and testing sets using the given 
    # feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, 
                                                        data[feature], 
                                                        test_size = 0.25, 
                                                        random_state = 1)
    
    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state = 1)
    regressor.fit(X_train, y_train)
    
    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print 'R^2 score when removing {}: {}'.format(feature, score)

#pd.scatter_matrix(data, alpha = 0.3, figsize = (14, 8), diagonal = 'kde', 
#                  c = 'green', density_kwds = {'colormap':['green']})

import seaborn as sns
sns.pairplot(data, diag_kind = 'kde')
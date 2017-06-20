# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:57:45 2016

@author: mathe
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Supplementary code
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score

# Pretty display for notebooks
# %matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.\n".format(*data.shape)

# TODO: Minimum price of the data
minimum_price = prices.min()

# TODO: Maximum price of the data
maximum_price = prices.max()

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median()

# TODO: Standard deviation of prices of the data
std_price = prices.std()

# Show the calculated statistics
print "Statistics for Boston housing dataset:"
print "- Minimum price: ${:,.2f}".format(minimum_price)
print "- Maximum price: ${:,.2f}".format(maximum_price)
print "- Mean price: ${:,.2f}".format(mean_price)
print "- Median price: ${:,.2f}".format(median_price)
print "- Standard deviation of prices: ${:,.2f}".format(std_price)

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance between true and predict values
        based on the metric chosen. """
    
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

# Draw the graphic
plt.plot(data['RM'], data['MEDV'], 'r.')
plt.plot(data['LSTAT'], data['MEDV'], 'b.')
plt.plot(data['PTRATIO'], data['MEDV'], 'g.')
print "\nRed: RM variable"
print "Blue: LSTAT variable"
print "Green: PTRATIO variable"
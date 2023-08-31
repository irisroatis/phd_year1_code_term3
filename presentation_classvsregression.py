#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:33:38 2023

@author: roatisiris
"""

import matplotlib.pyplot as plt

import pandas as pd
from functions import *
import numpy as np
import random
import sklearn

mu1 = np.array([0])
mu2 = np.array([2])
sigma1 = np.identity(1)
sigma2 = np.identity(1)
how_many_times_repeat = 1
iterations = 100


testing_data, belonging_classes = generating_test_data(how_many_times_repeat, iterations, mu1, sigma1, mu2, sigma2)
x = testing_data[0]
y = x * 2 + 3 + np.random.normal(0,1,(iterations,1))


m1, b1 = np.polyfit(x.reshape((-1,)), y, 1)
new_point = 0.3

plt.plot(x,y,'.',color = 'tab:blue',markersize = 6, label = 'train population')
plt.plot(x, m1*x+b1, color='tab:red', linewidth = 1, label = 'regression line')
plt.plot(new_point, m1 * new_point+b1, 'P', color = 'tab:green', label = 'new value')
plt.tick_params(labelleft = False , labelbottom = False)
plt.xlabel('height')
plt.ylabel('body weight')
plt.ylim(min(y)-0.5,max(y)+0.5)
plt.legend()
plt.show()


d = {'height': x.reshape((-1,)), 'weight': y.reshape((-1,))}
df = pd.DataFrame(data=d)

# Fit the data to a logistic regression model.
clf = sklearn.linear_model.LogisticRegression()
clf.fit(df, belonging_classes[0])

# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2


see1 = np.where(belonging_classes[0]==1)[0]
see0 = np.where(belonging_classes[0]==0)[0]
x_small = np.linspace(0.4, 1.5, 20)

plt.scatter(x[see1],y[see1], label = 'adult (class 1)', s = 6)
plt.scatter(x[see0],y[see0], label = 'child (class 0)',s = 6)
plt.plot(new_point, m1 * new_point+b1, 'P', color = 'tab:green', label = 'new value')
# plt.axvline(x = 0.5, color = 'tab:pink', label = 'decision boundary')
plt.plot(x_small, m*x_small+c, color = 'tab:red', label = 'decision boundary')
plt.tick_params(labelleft = False , labelbottom = False)
plt.xlabel('height')
plt.ylabel('body weight')
plt.ylim(min(y)-0.5,max(y)+0.5)
plt.legend()
plt.show()







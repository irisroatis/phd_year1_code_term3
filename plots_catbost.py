#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:55:51 2023

@author: ir318
"""
import inspect
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from feature_engine.encoding import MeanEncoder
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing
import random
import numpy as np
import seaborn as sns
from functions import *
from scipy.stats import rankdata
from kfold_code import *
from sklearn import tree
from itertools import permutations

# list_cat = ['A','C','B','A','B','A']
# list_target = np.array([1,0,1,1,0,0])
# df = pd.DataFrame(list_target,columns = ['target'])
# df['Feature'] = list_cat
# categorical_variables=['Feature']
# target_variable = 'target'

# encoder =  ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False, a = alpha)
# encoder.fit(df, df[target_variable])  
# see = encoder.transform(df)

# list_cat = ['A','A','A','A','B','B','B','C','C','D']
# list_target = np.array([0,1,0,1,0,1,0,1,0,1])
# df = pd.DataFrame(list_target,columns = ['target'])
# df['Feature'] = list_cat
# categorical_variables=['Feature']
# target_variable = 'target'

which_dataset = 'Simulated Data'
df = pd.read_csv('simulate_categories.csv')
categorical_variables = ['Feature_3'] 
target_variable = 'target'
continuous_variables = ['Feature_1','Feature_2']

bins = np.loadtxt('bins.txt', delimiter= ',')
order_cat = np.loadtxt('order_cat.txt', delimiter= ',')


df.drop(continuous_variables, axis = 1, inplace = True)


unique_values = df[categorical_variables[0]].unique()
store_everything = {}
how_many_permutations = 10

for cat in unique_values:
    pick = df[df[categorical_variables[0]] == cat]
    actual_indices = list(pick.index)
    matrix = np.zeros((len(actual_indices),how_many_permutations))


    for index in range(how_many_permutations):
        
       order = random.sample(actual_indices, len(actual_indices))
        
        
       list_encoded = []
       current_dataset = pick.loc[order].copy()
       encoder =  ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
       encoded_current = encoder.fit_transform(current_dataset, current_dataset[target_variable])
       encoded_current = encoded_current.sort_index()
       matrix[:,index] = encoded_current[categorical_variables[0]]
    store_everything[cat] = matrix


for cat in unique_values:
    plt.figure()
    this_cat = store_everything[cat]
    for i in range(this_cat.shape[0]):
        plt.hist(this_cat[i,:], alpha = 0.3, bins = 20)
        plt.title('Category:' +str(cat)+', index:'+str(i))
        plt.show()

for cat in unique_values:
    plt.figure()
    this_cat = store_everything[cat]
    plt.hist(np.matrix.flatten(this_cat), alpha = 0.3, bins = 20)
    plt.title('All Category:' +str(cat))
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:15:45 2023

@author: roatisiris
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
import math
from scipy import stats

# list_cat = ['A','C','B','A','B','A']
# list_target = np.array([1,0,1,1,0,0])
# df = pd.DataFrame(list_target,columns = ['target'])
# df['Feature'] = list_cat
# categorical_variables=['Feature']
# target_variable = 'target'

which_dataset = 'Simulated Data'
df = pd.read_csv('simulate_categories.csv')
categorical_variables = ['Feature_3'] 
target_variable = 'target'
continuous_variables = ['Feature_1','Feature_2']


unique_values = df[categorical_variables[0]].unique()
store_everything = {}
prior = df[df[target_variable] == 1].shape[0] / df.shape[0]
alpha = 1
al_pri = alpha * prior



for cat in unique_values:
    pick = df[df[categorical_variables[0]] == cat]
    if pick.shape[0] == 1:
        store_everything[cat] = np.array([prior])
    else:
        l = random.sample(list(permutations(pick.index)), 10)
        
        
        matrix = np.zeros((len(l[0]), len(l)))
        
        prior_cat = pick[pick[target_variable] == 1].shape[0] / pick.shape[0]
        
        list_order = np.arange(alpha, len(l[0]) +alpha)
        vector = (al_pri - alpha * prior_cat) / list_order
        
        for index in range(len(l)):
           list_encoded = []
           current_dataset = pick.loc[list(l[index])].copy()

           encoder =  ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False, a = alpha)
           encoded_current = encoder.fit_transform(current_dataset, current_dataset[target_variable])
           
           encoded_current[categorical_variables[0] ] += vector
           
           encoded_current = encoded_current.sort_index()
           
           matrix[:,index] = encoded_current[categorical_variables[0]]
        store_everything[cat] = matrix
        
    target_column = pick[target_variable]
    target_column.reset_index(inplace=True, drop=True)
    
    zeros = np.where(target_column == 0)[0]
    ones = np.where(target_column == 1)[0]
    
    count = 0
    
    for i in zeros:
        for j in zeros:
            see = stats.ks_2samp(matrix[i,:], matrix[j,:]).pvalue
            if see > 0.05:
                count += 1
                
    for i in ones:
        for j in ones:
            see = stats.ks_2samp(matrix[i,:], matrix[j,:]).pvalue
            if see > 0.05:
                count += 1
                





        
    
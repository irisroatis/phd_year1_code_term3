#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:44:29 2023

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

list_cat = ['A','C','B','A','B','A']
list_target = np.array([1,0,1,1,0,0])
df = pd.DataFrame(list_target,columns = ['target'])
df['Feature'] = list_cat
categorical_variables=['Feature']
target_variable = 'target'

encoder =  ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False, a = alpha)
encoder.fit(df, df[target_variable])  
see = encoder.transform(df)

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


# unique_cat = list(set(list_cat))
# cat_and_encoded = pd.DataFrame(unique_cat, columns = ['Feature'])
# cat_and_encoded['target'] = np.nan

# encoder = ce.woe.WOEEncoder(cols=categorical_variables,verbose=False)
# woe_df = encoder.fit_transform(df, df[target_variable])
# cat_and_encoded['woe'] = encoder.transform(cat_and_encoded[['Feature','target']])['Feature']


# encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_variables,verbose=False)
# leave_df = encoder.fit_transform(df, df[target_variable])
# cat_and_encoded['leave'] =encoder.transform(cat_and_encoded[['Feature','target']])['Feature']

unique_values = df[categorical_variables[0]].unique()
store_everything = {}
prior = df[df[target_variable] == 1].shape[0] / df.shape[0]
alpha = 1
al_pri = alpha * prior


for cat in unique_values:
    pick = df[df[categorical_variables[0]] == cat]
    l = list(permutations(pick.index))
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

for cat in unique_values:
    plt.figure()
    this_cat = store_everything[cat]
    for i in range(this_cat.shape[0]):
        plt.hist(this_cat[i,:], alpha = 0.3, bins = 15)
        plt.show()


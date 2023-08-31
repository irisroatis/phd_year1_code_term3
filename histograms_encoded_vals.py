#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:05:54 2023

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


def multiple_perm(df, categorical_variable, how_many_permutations):
    unique_values = df[categorical_variable].unique()
    store_everything = {}
    
    for cat in unique_values:
    
        pick = df[df[categorical_variable] == cat]
        actual_indices = list(pick.index)
        matrix = np.zeros((len(actual_indices),how_many_permutations))
    
    
        for index in range(how_many_permutations):
            
           order = random.sample(actual_indices, len(actual_indices))
            
            
           list_encoded = []
           current_dataset = pick.loc[order].copy()
           encoder =  ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
           encoded_current = encoder.fit_transform(current_dataset, current_dataset[target_variable])
           encoded_current = encoded_current.sort_index()
           matrix[:,index] = encoded_current[categorical_variable]
        store_everything[cat] = matrix
        
    return store_everything

def catboost_various_methods(df, categorical_variable, store_everything, which_method, prior):
    
    encoded_df = df.copy()
    unique_values = df[categorical_variable].unique()
    
    unique_values_test = list(set(df_test[col].unique()) - set(df_train[col].unique()))
    
    if which_method == 'mean':
        dict_for_test ={}
        for cat in unique_values:
            pick =  encoded_df[encoded_df[categorical_variable] == cat]
            vector = np.mean(store_everything[cat],axis = 1)
            encoded_df.loc[pick.index, categorical_variable] = vector
            dict_for_test[cat] = np.mean(vector)
            
    if which_method == 'median':
        dict_for_test ={}
        for cat in unique_values:
            pick =  encoded_df[encoded_df[categorical_variable] == cat]
            vector = np.median(store_everything[cat],axis = 1)
            encoded_df.loc[pick.index, categorical_variable] = vector
            dict_for_test[cat] = np.mean(vector)
            
            for uni_test in unique_values_test:
                dict_for_test[uni_test] = prior
            
    if which_method == 'mean_interq':
        dict_for_test_m ={}
        dict_for_test_lq ={}
        dict_for_test_uq ={}
        for cat in unique_values:
            pick =  encoded_df[encoded_df[categorical_variable] == cat]
            vector_m = np.mean(store_everything[cat],axis = 1)
            vector_l1qr = np.percentile(store_everything[cat], q = 25, axis = 1)
            vector_u1qr = np.percentile(store_everything[cat], q = 75, axis = 1)
            encoded_df.loc[pick.index, categorical_variable] = vector_m
            encoded_df.loc[pick.index, categorical_variable+'_lower'] = vector_l1qr
            encoded_df.loc[pick.index, categorical_variable+'_upper'] = vector_u1qr
            dict_for_test_m[cat] = np.mean(vector_m)
            dict_for_test_lq[cat] = np.mean(vector_l1qr) # np.percentile(vector_l1qr, q = 25)
            dict_for_test_uq[cat] = np.mean(vector_u1qr) # np.percentile(vector_u1qr, q = 75)
            
            
            for uni_test in unique_values_test:
                dict_for_test_m[uni_test] = prior
                dict_for_test_lq[uni_test] = prior
                dict_for_test_uq[uni_test] = prior
            
            
        dict_for_test={'m':dict_for_test_m, 'l': dict_for_test_lq, 'u': dict_for_test_uq}
        
    if which_method == 'mean_minmax':
        dict_for_test_m ={}
        dict_for_test_min ={}
        dict_for_test_max ={}
        for cat in unique_values:
            pick =  encoded_df[encoded_df[categorical_variable] == cat]
            vector_m = np.mean(store_everything[cat],axis = 1)
            vector_min = np.min(store_everything[cat], axis = 1)
            vector_max = np.max(store_everything[cat], axis = 1)
            encoded_df.loc[pick.index, categorical_variable] = vector_m
            encoded_df.loc[pick.index, categorical_variable+'_min'] = vector_min
            encoded_df.loc[pick.index, categorical_variable+'_max'] = vector_max
            dict_for_test_m[cat] = np.mean(vector_m)
            dict_for_test_min[cat] = min(vector_min)
            dict_for_test_max[cat] = max(vector_max)
        
       
            
            
            
            
        dict_for_test={'m':dict_for_test_m, 'min': dict_for_test_min, 'max': dict_for_test_max}
          
    return encoded_df, dict_for_test



##### ## Adult (income >=50k or <50k)

which_dataset = 'Income Prediction'
df = pd.read_csv('ada_prior.csv')
df.reset_index(inplace=True, drop = True)


df = df.drop(['educationNum','fnlwgt'],axis = 1)

categorical_variables = ['workclass','education',
                          'maritalStatus','occupation','relationship','race','nativeCountry'] 
binary_cols = ['sex']
target_variable = 'label'
continuous_variables = ['age','capitalGain','capitalLoss','hoursPerWeek']
df[binary_cols] = df[binary_cols].replace(['Male', 'Female'], [1, 0])
df[target_variable] = df[target_variable].replace([-1], [0])

encoded_df = df.copy()

store_everything = multiple_perm(encoded_df, categorical_variables[5], 30)

for key in store_everything.keys():
    plt.hist(store_everything[key].flatten(), bins = 100)
    plt.title('Category:' +str(key))
    plt.show()


for key in categorical_variables:
    print(df[key].unique().size)
    

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
df_train = pd.read_csv('simulate_categories_train.csv',index_col=False)
df_test = pd.read_csv('simulate_categories_train.csv',index_col=False)
categorical_variables = ['Feature_3'] 
target_variable = 'target'
continuous_variables = ['Feature_1','Feature_2']

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

def catboost_various_methods(df, categorical_variable, store_everything, which_method):
    
    encoded_df = df.copy()
    unique_values = df[categorical_variable].unique()
    
    if which_method == 'mean':
        dict_for_test ={}
        for cat in unique_values:
            pick =  encoded_df[encoded_df[categorical_variable] == cat]
            vector = np.mean(store_everything[cat],axis = 1)
            encoded_df.loc[pick.index, categorical_variable] = vector
            dict_for_test[cat] = np.mean(vector)
            
            
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
            dict_for_test_lq[cat] = np.percentile(vector_l1qr, q = 25)
            dict_for_test_uq[cat] = np.percentile(vector_u1qr, q = 75)
        dict_for_test={'m':dict_for_test_m, 'l': dict_for_test_lq, 'u': dict_for_test_uq}
    return encoded_df, dict_for_test


how_many_permutations = 20
categorical_variable = 'Feature_3'
store_everything = multiple_perm(df_train, categorical_variable , how_many_permutations)
encoded_df, dict_for_test =  catboost_various_methods(df_train, categorical_variable, store_everything, 'mean')


encoded_df_3, dict_for_test_3 =  catboost_various_methods(df_train, categorical_variable, store_everything, 'mean_interq')


encoded_df_test = df_test.copy()
encoded_df_test_3 = df_test.copy()
encoded_df_test_3[categorical_variable+'_lower'] = encoded_df_test_3[categorical_variable].copy()
encoded_df_test_3[categorical_variable+'_upper'] = encoded_df_test_3[categorical_variable].copy()

encoded_df_test[categorical_variable].replace(list(dict_for_test.keys()),list(dict_for_test.values()), inplace = True)
encoded_df_test_3[categorical_variable].replace(list(dict_for_test_3['m'].keys()),list(dict_for_test_3['m'].values()), inplace = True)
encoded_df_test_3[categorical_variable+'_lower'].replace(list(dict_for_test_3['l'].keys()),list(dict_for_test_3['l'].values()), inplace = True)
encoded_df_test_3[categorical_variable+'_upper'].replace(list(dict_for_test_3['u'].keys()),list(dict_for_test_3['u'].values()), inplace = True)


X_catboost_mean, y_train =  dataset_to_Xandy(encoded_df, target_variable, only_X = False)
X_catboost_mean_test, y_test =  dataset_to_Xandy(encoded_df_test, target_variable, only_X = False)

X_catboost_meaninterq, y_train =  dataset_to_Xandy(encoded_df_3, target_variable, only_X = False)
X_catboost_meaninterq_test, y_test =  dataset_to_Xandy(encoded_df_test_3, target_variable, only_X = False)

classifiers = ['logistic','kNN','dec_tree','rand_for','grad_boost']
aucmean = []
aucmeaninterq = []
for classifier in classifiers:
    aucmean.append(calc_conf_matrix(X_catboost_mean,y_train,X_catboost_mean_test,y_test, classifier))
    aucmeaninterq.append(calc_conf_matrix(X_catboost_meaninterq,y_train,X_catboost_meaninterq_test,y_test, classifier))




##### if plots for cat are wanted
# for cat in unique_values:
#     plt.figure()
#     this_cat = store_everything[cat]
#     plt.hist(np.matrix.flatten(this_cat), alpha = 0.3, bins = 20)
#     plt.title('All Category:' +str(cat))
#     plt.show()


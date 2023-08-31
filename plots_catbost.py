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



##### SIMULATED DATASET



# which_dataset = 'Simulated Data'
# df_train = pd.read_csv('simulate_categories_train.csv',index_col=False)
# df_test = pd.read_csv('simulate_categories_train.csv',index_col=False)
# categorical_variables = ['Feature_3'] 
# target_variable = 'target'
# continuous_variables = ['Feature_1','Feature_2']



# how_many_permutations = 100
# categorical_variable = 'Feature_3'
# store_everything = multiple_perm(df_train, categorical_variable , how_many_permutations)
# encoded_df, dict_for_test =  catboost_various_methods(df_train, categorical_variable, store_everything, 'mean')


# encoded_df_3, dict_for_test_3 =  catboost_various_methods(df_train, categorical_variable, store_everything, 'mean_interq')

# encoded_df_minmax, dict_for_test_minmax =  catboost_various_methods(df_train, categorical_variable, store_everything, 'mean_minmax')


# #### initialise all test datasets
# encoded_df_test = df_test.copy()
# encoded_df_test_3 = df_test.copy()
# encoded_df_test_minmax = df_test.copy()

# encoded_df_test_3[categorical_variable+'_lower'] = encoded_df_test_3[categorical_variable].copy()
# encoded_df_test_3[categorical_variable+'_upper'] = encoded_df_test_3[categorical_variable].copy()

# encoded_df_test_minmax[categorical_variable+'_min'] = encoded_df_test_minmax[categorical_variable].copy()
# encoded_df_test_minmax[categorical_variable+'_max'] = encoded_df_test_minmax[categorical_variable].copy()


# #### encode test set 

# encoded_df_test[categorical_variable].replace(list(dict_for_test.keys()),list(dict_for_test.values()), inplace = True)
# encoded_df_test_3[categorical_variable].replace(list(dict_for_test_3['m'].keys()),list(dict_for_test_3['m'].values()), inplace = True)
# encoded_df_test_3[categorical_variable+'_lower'].replace(list(dict_for_test_3['l'].keys()),list(dict_for_test_3['l'].values()), inplace = True)
# encoded_df_test_3[categorical_variable+'_upper'].replace(list(dict_for_test_3['u'].keys()),list(dict_for_test_3['u'].values()), inplace = True)
# encoded_df_test_minmax[categorical_variable].replace(list(dict_for_test_minmax['m'].keys()),list(dict_for_test_minmax['m'].values()), inplace = True)
# encoded_df_test_minmax[categorical_variable+'_min'].replace(list(dict_for_test_minmax['min'].keys()),list(dict_for_test_minmax['min'].values()), inplace = True)
# encoded_df_test_minmax[categorical_variable+'_max'].replace(list(dict_for_test_minmax['max'].keys()),list(dict_for_test_minmax['max'].values()), inplace = True)


# X_catboost_mean, y_train =  dataset_to_Xandy(encoded_df, target_variable, only_X = False)
# X_catboost_mean_test, y_test =  dataset_to_Xandy(encoded_df_test, target_variable, only_X = False)

# X_catboost_meaninterq, y_train =  dataset_to_Xandy(encoded_df_3, target_variable, only_X = False)
# X_catboost_meaninterq_test, y_test =  dataset_to_Xandy(encoded_df_test_3, target_variable, only_X = False)

# X_catboost_minmax, y_train =  dataset_to_Xandy(encoded_df_minmax, target_variable, only_X = False)
# X_catboost_minmax_test, y_test =  dataset_to_Xandy(encoded_df_test_minmax, target_variable, only_X = False)



##### if plots for cat are wanted
# for cat in unique_values:
#     plt.figure()
#     this_cat = store_everything[cat]
#     plt.hist(np.matrix.flatten(this_cat), alpha = 0.3, bins = 20)
#     plt.title('All Category:' +str(cat))
#     plt.show()









###### ## Adult (income >=50k or <50k)

# which_dataset = 'Income Prediction'
# df = pd.read_csv('ada_prior.csv')
# df.reset_index(inplace=True, drop = True)


# df = df.drop(['educationNum','fnlwgt'],axis = 1)

# categorical_variables = ['workclass','education',
#                           'maritalStatus','occupation','relationship','race','nativeCountry'] 
# binary_cols = ['sex']
# target_variable = 'label'
# continuous_variables = ['age','capitalGain','capitalLoss','hoursPerWeek']
# df[binary_cols] = df[binary_cols].replace(['Male', 'Female'], [1, 0])
# df[target_variable] = df[target_variable].replace([-1], [0])



####### AUSTRALIAN CREDIT
which_dataset = 'Australian Credit Approval'
df = pd.read_csv('australian.csv')
df.columns = df.columns.str.replace("'","")


categorical_variables = ['A4','A5','A6','A12'] 
binary_cols = ['A1','A8', 'A9', 'A11']
target_variable = 'A15'
continuous_variables = ['A2','A3','A7','A10','A13', 'A14']









#####################

how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s
prior = how_many_1s / size


how_many_cv = 5
which_method = 'mean_interq'
classifiers = ['logistic','kNN','dec_tree','rand_for','grad_boost']

aucmean = np.zeros((len(classifiers),how_many_cv))
aucmeaninterq = np.zeros((len(classifiers),how_many_cv))
aucmedian = np.zeros((len(classifiers),how_many_cv))

for index in range(how_many_cv):

    randomlist = random.sample(list(df[df[target_variable]==0].index.values), 4 * how_many_0s // 5) + random.sample(list(df[df[target_variable]==1].index.values), 4 * how_many_1s // 5)
    not_in_randomlist = list(set(range(0,size)) - set(randomlist))
    df_test = df.iloc[not_in_randomlist,:]
    df_train = df.iloc[randomlist,:]
    df_train.sort_index(inplace=True)
    df_train.reset_index(inplace=True, drop = True)
    df_test.reset_index(inplace=True, drop = True)
    
    encoded_df = df_train.copy()
    encoded_df_test = df_test.copy()
    
    encoded_df_median = df_train.copy()
    encoded_df_median_test = df_test.copy()
    
    
    for col in categorical_variables:
        
        store_everything = multiple_perm(encoded_df, col, 30)
    
        encoded_df, dict_test = catboost_various_methods(encoded_df, col, store_everything, which_method, prior)
    
        encoded_df_test[col] = np.nan
        encoded_df_test[col+'_lower'] = np.nan
        encoded_df_test[col+'_upper'] = np.nan
        
        encoded_df_test[col] = df_test[col].replace(list(dict_test['m'].keys()),list(dict_test['m'].values()))
        encoded_df_test[col+'_lower'] = df_test[col].replace(list(dict_test['l'].keys()),list(dict_test['l'].values()))
        encoded_df_test[col+'_upper'] = df_test[col].replace(list(dict_test['u'].keys()),list(dict_test['u'].values()))
        
        encoded_df_median, dict_test_median = catboost_various_methods(encoded_df_median, col, store_everything, 'median', prior)
        encoded_df_median_test[col] = np.nan
        encoded_df_median_test[col] = df_test[col].replace(list(dict_test_median.keys()),list(dict_test_median.values()))
    
    encoded_df_meanonly = encoded_df.copy()
    encoded_df_meanonly.drop(encoded_df_meanonly.columns[encoded_df_meanonly.columns.str.contains('_lower')], axis=1, inplace=True)
    encoded_df_meanonly.drop(encoded_df_meanonly.columns[encoded_df_meanonly.columns.str.contains('_upper')], axis=1, inplace=True)
    
       
    encoded_df_test_meanonly = encoded_df_test.copy()
    encoded_df_test_meanonly.drop(encoded_df_test_meanonly.columns[encoded_df_test_meanonly.columns.str.contains('_lower')], axis=1, inplace=True)
    encoded_df_test_meanonly.drop(encoded_df_test_meanonly.columns[encoded_df_test_meanonly.columns.str.contains('_upper')], axis=1, inplace=True)
    
    
    
    
    
    X_catboost_meaninterq, y_train =  dataset_to_Xandy(encoded_df, target_variable, only_X = False)
    X_catboost_meaninterq_test, y_test =  dataset_to_Xandy(encoded_df_test, target_variable, only_X = False)
    
    X_catboost_mean =  dataset_to_Xandy(encoded_df_meanonly, target_variable, only_X = True)
    X_catboost_mean_test =  dataset_to_Xandy(encoded_df_test_meanonly, target_variable, only_X = True)
    
    
    X_catboost_median =  dataset_to_Xandy(encoded_df_median, target_variable, only_X = True)
    X_catboost_median_test =  dataset_to_Xandy(encoded_df_median_test, target_variable, only_X = True)
    



    for c in range(len(classifiers)):
        aucmeaninterq[c, index] = calc_conf_matrix(X_catboost_meaninterq,y_train,X_catboost_meaninterq_test,y_test, classifiers[c])
        aucmean[c, index] = calc_conf_matrix(X_catboost_mean,y_train,X_catboost_mean_test,y_test, classifiers[c])
        aucmedian[c, index] = calc_conf_matrix(X_catboost_median,y_train,X_catboost_median_test,y_test, classifiers[c])

include_means_conf = np.zeros((how_many_cv+1, len(classifiers)+1))
include_means_conf[:-1,:-1] = aucmean
include_means_conf[:-1,-1] = np.mean(aucmean, axis = 1)
include_means_conf[-1,:-1] = np.mean(aucmean, axis = 0)
include_means_conf[-1,-1]  =  np.nan

plt.figure(figsize=(10,7))
g = sns.heatmap(include_means_conf, annot=True, fmt=".5f")
g.set_xticklabels(classifiers + ['MEAN'], rotation = 45)
g.set_yticklabels(list(np.arange(1, 6) )+['MEAN'], rotation = 45)
plt.title('Plot ROC MEAN + LQ + UQ, Dataset: ' + str(which_dataset) )
plt.show()



include_means_conf2 = np.zeros((how_many_cv+1, len(classifiers)+1))
include_means_conf2[:-1,:-1] = aucmeaninterq
include_means_conf2[:-1,-1] = np.mean(aucmeaninterq, axis = 1)
include_means_conf2[-1,:-1] = np.mean(aucmeaninterq, axis = 0)
include_means_conf2[-1,-1]  =  np.nan

plt.figure(figsize=(10,7))
g = sns.heatmap(include_means_conf2, annot=True, fmt=".5f")
g.set_xticklabels(classifiers + ['MEAN'], rotation = 45)
g.set_yticklabels(list(np.arange(1, 6) )+['MEAN'], rotation = 45)
plt.title('Plot ROC MEAN only, Dataset: ' + str(which_dataset) )
plt.show()


include_means_conf3 = np.zeros((how_many_cv+1, len(classifiers)+1))
include_means_conf3[:-1,:-1] = aucmedian
include_means_conf3[:-1,-1] = np.mean(aucmedian, axis = 1)
include_means_conf3[-1,:-1] = np.mean(aucmedian, axis = 0)
include_means_conf3[-1,-1]  =  np.nan

plt.figure(figsize=(10,7))
g = sns.heatmap(include_means_conf3, annot=True, fmt=".5f")
g.set_xticklabels(classifiers + ['MEAN'], rotation = 45)
g.set_yticklabels(list(np.arange(1, 6) )+['MEAN'], rotation = 45)
plt.title('Plot ROC MEDIAN only, Dataset: ' + str(which_dataset) )
plt.show()
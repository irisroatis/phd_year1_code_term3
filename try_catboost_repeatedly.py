#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:28:48 2023

@author: ir318
"""

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

def catboost(df_train, cat_df, df_test, categorical_variables, alpha, prior):
    dictionary_cat = {}
    cat_df_test_difftest = df_test.copy()

    for col in categorical_variables:
        unique_cat = list(set(df_train[col]))

        for category in unique_cat:     
            indices = np.where(df_train[col] == category)[0]
            part_dataset_encoded = cat_df.iloc[indices]
            get_numer = np.sum(part_dataset_encoded[col]) + alpha * prior
            dictionary_cat[category] = get_numer / (len(indices) + alpha)

        cat_df_test_difftest[col] = cat_df_test_difftest[col].replace(list(dictionary_cat.keys()), list(dictionary_cat.values()))
       
        values_in_test_not_train =set(df_train[col]) - set()
    return cat_df_test_difftest


def calc_conf_matrix(X_train,y_train,X_test, y_test,classifier):
    
    # depending on what the user inputs
    if classifier == 'logistic':
        model = LogisticRegression(penalty = 'none')  
    elif classifier == 'kNN':
        model = KNeighborsClassifier(n_neighbors = 15)  
    elif classifier == 'dec_tree':
        model = DecisionTreeClassifier()
    elif classifier == 'rand_for':
        model = RandomForestClassifier(n_estimators = 500)
    elif  classifier == 'grad_boost':
        model = GradientBoostingClassifier(learning_rate = 0.01)
    elif classifier == 'naive':
        model= GaussianNB()
    elif classifier == 'lasso':
        model = LogisticRegression(penalty = 'l1', C = 1/5, solver = 'saga') 
    else:
        assert('Classifier unknown')
    
    # perform fitting of the model 
    model.fit(X_train, y_train)   #.reshape(-1,)
    
    y_predicted = model.predict(X_test) 

    return y_predicted




which_dataset = 'Simulated Data'
df = pd.read_csv('simulate_categories.csv')
categorical_variables = ['Feature_3'] 
target_variable = 'target'
continuous_variables = ['Feature_1','Feature_2']

bins = np.loadtxt('bins.txt', delimiter= ',')
order_cat = np.loadtxt('order_cat.txt', delimiter= ',')

# list_cat = ['A','A','A','A','B','B','B','C','C','D']
# list_target = np.array([0,1,0,1,0,1,0,1,0,1])
# df = pd.DataFrame(list_target,columns = ['target'])
# df['Feature'] = list_cat
# categorical_variables=['Feature']
# target_variable = 'target'

how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s


randomlist = random.sample(list(df[df[target_variable]==0].index.values), 4 * how_many_0s // 5) + random.sample(list(df[df[target_variable]==1].index.values), 4 * how_many_1s // 5)
not_in_randomlist = list(set(range(0,size)) - set(randomlist))




df_test = df.iloc[not_in_randomlist,:]
df_train = df.iloc[randomlist,:]
df_train.sort_index(inplace=True)
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)
alpha = 1
prior = how_many_1s / df_train.shape[0]




how_many_permutations = 100



##### the CatBoost encoded dataset 
encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
cat_df = encoder.fit_transform(df_train, df_train[target_variable])

X_cat,y_train=  dataset_to_Xandy(cat_df, target_variable, only_X = False)


classifier = 'kNN'

cat_df_test_difftest = catboost(df_train, cat_df, df_test, categorical_variables,alpha, prior)
X_cat_test_difftest, y_test =  dataset_to_Xandy(cat_df_test_difftest, target_variable, only_X = False)
y_predict = calc_conf_matrix(X_cat,y_train,X_cat_test_difftest,y_test, classifier)
    
all_predictions_test = np.zeros((len(y_predict), how_many_permutations+1))
all_predictions_test[:,0] = y_predict



for perm in range(how_many_permutations):
    index_dataset = np.arange(0, df_train.shape[0])
    np.random.shuffle(index_dataset)

    encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
    cat_df_shuffled = encoder.fit_transform(df_train.iloc[index_dataset], df_train.iloc[index_dataset][target_variable])
    cat_df_shuffled.sort_index(inplace = True)
    
    cat_df_test_difftest = catboost(df_train, cat_df_shuffled, categorical_variables,alpha, prior)
    X_cat_test_difftest, y_test =  dataset_to_Xandy(cat_df_test_difftest, target_variable, only_X = False)
    y_predict = calc_conf_matrix(X_cat,y_train,X_cat_test_difftest,y_test, classifier)
    
    all_predictions_test[:,perm+1] = y_predict
    
probabilities_of_one = np.mean(all_predictions_test , axis = 1)
    
    
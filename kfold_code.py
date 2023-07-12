#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:29:59 2023

@author: roatisiris
"""

import pandas as pd
from sklearn.model_selection import KFold
from functions import *


which_dataset = 'Heart Ilness'
df = pd.read_csv('heart.csv')
categorical_variables = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
target_variable = 'target' # Making sure the name of the target variable is known
continuous_variables = ['age','trestbps','chol','thalach','oldpeak']
binary_variables = ['sex','fbs','exang']


# df = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')
# fold_target_df = df.copy()
# target_variable = 'Target'
# categorical_variables = ['Feature']


for categorical_variable in categorical_variables:

    new_column = categorical_variable + '_encoded'
    df[new_column] = df[categorical_variable].copy()
    # df_test[new_column] = df_test[categorical_variable].copy()
    
    
    kf = KFold(n_splits=5, shuffle=False)
    categories = df[categorical_variable].unique()
        
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        df_train = df.iloc[train_index,:]
        _ ,dictionary =  target_encoding(categorical_variable, target_variable, df_train)
        df[new_column].iloc[test_index] =   df[new_column].iloc[test_index].replace(list(dictionary.keys()), list(dictionary.values()))
    
    
    # for cat in categories:
    #     which_cat = df[df[categorical_variable] == cat]
    #     avg_value = which_cat[new_column].mean()
    #     df_test[new_column] =  df_test[new_column].replace([cat], avg_value)
    
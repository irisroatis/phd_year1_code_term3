#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:29:59 2023

@author: roatisiris
"""

import pandas as pd
from sklearn.model_selection import KFold
from functions import *
import category_encoders as ce


# which_dataset = 'Heart Ilness'
# df = pd.read_csv('heart.csv')
# categorical_variables = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
# target_variable = 'target' # Making sure the name of the target variable is known
# continuous_variables = ['age','trestbps','chol','thalach','oldpeak']
# binary_variables = ['sex','fbs','exang']


# df = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')
# fold_target_df = df.copy()
# target_variable = 'Target'
# categorical_variables = ['Feature']

def k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds, which_encoder):
    
    modified_df = df.copy()
    modified_df_test = df_test.copy()
    for categorical_variable in categorical_variables:
    
        new_column = categorical_variable + '_encoded'
        modified_df[new_column] = modified_df[categorical_variable].copy()
        modified_df_test[new_column] = modified_df_test[categorical_variable].copy()
        
        
        kf = KFold(n_splits=how_many_folds, shuffle=True)
        categories = modified_df[categorical_variable].unique()
            
        for i, (train_index, test_index) in enumerate(kf.split(modified_df)):
            df_train = modified_df.iloc[train_index,:]
            if which_encoder =='target':
                _ ,dictionary =  target_encoding(categorical_variable, target_variable, df_train)
                modified_df[new_column].iloc[test_index] =   modified_df[new_column].iloc[test_index].replace(list(dictionary.keys()), list(dictionary.values()))
            elif which_encoder =='glmm':
               df_train_test = modified_df.iloc[test_index,:]
               encoder = ce.glmm.GLMMEncoder(cols=[categorical_variable],verbose=False)
               encoder.fit(df_train, df_train[target_variable])
               modified_df[new_column].iloc[test_index] = encoder.transform( df_train_test)[categorical_variable]

                
        for cat in categories:
            which_cat = modified_df[modified_df[categorical_variable] == cat]
            avg_value = which_cat[new_column].mean()
            modified_df_test[new_column] =  modified_df_test[new_column].replace([cat], avg_value)
    
    modified_df.drop(columns = categorical_variables, inplace=True)
    modified_df_test.drop(columns = categorical_variables, inplace=True)
    
    modified_df.columns = modified_df.columns.str.replace('_encoded', '')
    modified_df_test.columns = modified_df_test.columns.str.replace('_encoded', '')
    return modified_df, modified_df_test

    
# modified_df, modified_df_test = k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds=5)


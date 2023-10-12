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
            
        values_in_test_not_train =set(df_test[col]) - set(df_train[col])
        for val in values_in_test_not_train:
            dictionary_cat[val] = prior

        cat_df_test_difftest[col] = cat_df_test_difftest[col].replace(list(dictionary_cat.keys()), list(dictionary_cat.values()))
       
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
categorical_variables = ['Feature_1'] 
target_variable = 'target'
continuous_variables = []

bins = np.loadtxt('bins.txt', delimiter= ',')
order_cat = np.loadtxt('order_cat.txt', delimiter= ',')



########## Income Prediction
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
# which_dataset = 'Australian Credit Approval'
# df = pd.read_csv('australian.csv')
# df.columns = df.columns.str.replace("'","")


# categorical_variables = ['A4','A5','A6','A12'] 
# binary_cols = ['A1','A8', 'A9', 'A11']
# target_variable = 'A15'
# continuous_variables = ['A2','A3','A7','A10','A13', 'A14']




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
df_train.reset_index(inplace=True, drop  = True)
df_test.reset_index(inplace=True, drop  = True)
alpha = 1
prior = how_many_1s / df_train.shape[0]




how_many_permutations = 200
classifier = 'logistic'


##### the CatBoost encoded dataset 
encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
cat_df = encoder.fit_transform(df_train, df_train[target_variable])
X_cat,y_train=  dataset_to_Xandy(cat_df, target_variable, only_X = False)



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
    
    cat_df_test_difftest = catboost(df_train, cat_df_shuffled,df_test, categorical_variables,alpha, prior)
    X_cat_test_difftest, y_test =  dataset_to_Xandy(cat_df_test_difftest, target_variable, only_X = False)
    y_predict = calc_conf_matrix(X_cat,y_train,X_cat_test_difftest,y_test, classifier)
    
    all_predictions_test[:,perm+1] = y_predict
    
    
    
    
# see = np.arange(0, 2000, 10)
# accuracies = []

# for i in see:
    
    
#     probabilities_of_one = np.mean(all_predictions_test[:,:i] , axis = 1)
    
#     classes_assigned = probabilities_of_one * 1.0
    
#     for entry in range(len(classes_assigned)):
#         if classes_assigned[entry] > 0.5:
#             classes_assigned[entry] = 1
#         else: 
#             classes_assigned[entry] = 0
    
#     correct = (classes_assigned == df_test[target_variable])
#     accuracies.append(  correct.sum() / correct.size )
    
# plt.plot(see, accuracies)
# plt.title('Dataset:' + which_dataset)
# plt.show()
    
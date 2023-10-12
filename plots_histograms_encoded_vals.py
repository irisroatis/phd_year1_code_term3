#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:06:01 2023

@author: roatisiris
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
from sklearn import tree
from dealing_datasets import *
from tabulate import tabulate





def plot_encoded_diff_colours(X, which_category, enc_method, ind_0, ind_1):
    X0 = X[which_category].iloc[ind_0] 
    X1 = X[which_category].iloc[ind_1] 
    plt.hist(X0, alpha = 0.5, bins = 50, label = 'Distr 1', density = True)
    plt.hist(X1, alpha = 0.5, bins = 50, label = 'Distr 2', density = True)
    plt.legend()
    plt.title('Histograms of '+enc_method+' Encoded Values')
    
def decboundary_tprfpr(encoded_df_test, which_category, target_variable):
    db = np.linspace(min(encoded_df_test[which_category]), max(encoded_df_test[which_category]), 50)
    # db = [0.5]
    tpr = []
    fpr = []
    for c in db:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        size = encoded_df_test.shape[0]
        for i in range(size):
            row  = encoded_df_test.iloc[i]
            if row[target_variable] == 0:
                if row[which_category] > c:
                    fp += 1
                else:
                    tn += 1
            elif row[target_variable] == 1:
                if row[which_category] > c:
                    tp += 1
                else:
                    fn += 1
        tpr.append(tp / (tp+fn))
        fpr.append(fp / (tn+fp))
    return tpr, fpr


def miss_classification(encoded_df_test, value_decision, which_category, target_variable):
    how_many_misclassified_toC1 = 0
    how_many_misclassified_toC0 = 0
    size = encoded_df_test.shape[0]
    pi0 = len(encoded_df_test[encoded_df_test[target_variable] == 0]) / size
    for i in range(size):
        row  = encoded_df_test.iloc[i]
        if row[target_variable] == 0 and row[which_category] > value_decision:
            how_many_misclassified_toC1 += 1
        elif row[target_variable] == 1 and row[which_category] < value_decision:
            how_many_misclassified_toC0 += 1
            
    return (how_many_misclassified_toC1 * (pi0) + how_many_misclassified_toC0 * (1 - pi0))/size







# def miss_classification_uni_beta(encoded_df_test, val1, val2, which_category, target_variable):
#     how_many_misclassified_toC1 = 0
#     how_many_misclassified_toC0 = 0
#     size = encoded_df_test.shape[0]
#     pi0 = len(encoded_df_test[encoded_df_test[target_variable] == 0]) / size
#     for i in range(size):
#         row  = encoded_df_test.iloc[i]
#         if row[target_variable] == 1 and row[which_category] > val1 and row[which_category] < val2:
#             how_many_misclassified_toC += 1
#         elif row[target_variable] == 1 and row[which_category] < value_decision:
#             how_many_misclassified_toC1 += 0
            
#     return (how_many_misclassified_toC1 * (pi0) + how_many_misclassified_toC0 * (1 - pi0))/size


#### SIMULATED DATASET

which_dataset = 'Simulated Data One Dimension'
df_train = pd.read_csv('simulate_categories_train_two_normals.csv')
df_test = pd.read_csv('simulate_categories_test_two_normals.csv')
categorical_variables = ['Feature_1'] 
target_variable = 'target'
continuous_variables = []
binary_cols = []

original_testdata = pd.read_csv('initial_testdata_two_normals.csv')
# which_dataset = 'Uniform and Beta Distribution'


# value_decision = 0.025
# value_decision = 0.5
# value_decision = 0.25
# changed_value_decision = value_decision/8 +0.5

dict_tpr = {}
dict_fpr = {}
which_category = 'Feature_1'
see =  pd.DataFrame(df_train[which_category].value_counts())
table = tabulate(see, headers= ['Category','Count'])

dict_tpr['CONT'],  dict_fpr['CONT']= decboundary_tprfpr(original_testdata, which_category, target_variable)


# bins = np.loadtxt('bins.txt', delimiter= ',')
# order_cat = np.loadtxt('order_cat.txt', delimiter= ',')

# bins = np.loadtxt('bins_unif_beta.txt', delimiter= ',')
# order_cat = np.loadtxt('order_cat_unif_beta.txt', delimiter= ',')
bins = np.loadtxt('bins_two_normals.txt', delimiter= ',')
order_cat = np.loadtxt('order_cat_two_normals.txt', delimiter= ',')
plt.figure(figsize=(30,40))


ind_0 = np.where(df_train[target_variable] == 0)[0]
ind_1 = np.where(df_train[target_variable] == 1)[0]



method = 'ORD'


encoder = ce.OrdinalEncoder(cols=categorical_variables,verbose=False)
simple_df = encoder.fit_transform(df_train)
X_simple =  dataset_to_Xandy(simple_df, target_variable, only_X = True)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)




plt.subplot(6, 2, 1)
plot_encoded_diff_colours(X_simple, which_category, method, ind_0, ind_1)


method = 'WOE'
encoder = ce.woe.WOEEncoder(cols=categorical_variables,verbose=False)
woe_df = encoder.fit_transform(df_train, df_train[target_variable])
X_woe =  dataset_to_Xandy(woe_df, target_variable, only_X = True)
plt.subplot(6, 2, 2)
plot_encoded_diff_colours(X_woe, which_category, method, ind_0, ind_1)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)


alpha = 1
how_many_1s = len(df_train[df_train[target_variable] == 1])
prior = how_many_1s / df_train.shape[0]



##### the target encoded dataset 

method = 'TAR'
target_df = df_train.copy()
target_df_test = df_test.copy()
for col in categorical_variables:
    dict_target = {}
    target_df, dict_target =  target_encoding(col, target_variable, target_df)  
    target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
X_target =  dataset_to_Xandy(target_df, target_variable, only_X = True)
plt.subplot(6, 2, 3)
plot_encoded_diff_colours(X_target, which_category, method, ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(target_df_test, which_category, target_variable)


##### the target encoded dataset 

method = 'TAR_W'
encoder = ce.target_encoder.TargetEncoder(cols = categorical_variables, verbose=False)
target_w_df = encoder.fit_transform(df_train, df_train[target_variable])
X_target_w =  dataset_to_Xandy(target_w_df, target_variable, only_X = True)
plt.subplot(6, 2, 4)
plot_encoded_diff_colours(X_target_w, which_category, method, ind_0, ind_1)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)

 

##### the GLMM encoded dataset 
method = 'GLMM'
encoder = ce.glmm.GLMMEncoder(cols=categorical_variables,verbose=False)
glmm_df = encoder.fit_transform(df_train, df_train[target_variable])
X_glmm =  dataset_to_Xandy(glmm_df, target_variable, only_X = True)
plt.subplot(6, 2, 5)
plot_encoded_diff_colours(X_glmm, which_category, method, ind_0, ind_1)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)


##### the leave-one-out encoded dataset 
method = 'LOO'
encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_variables,verbose=False)
leave_df = encoder.fit_transform(df_train, df_train[target_variable])
X_leave=  dataset_to_Xandy(leave_df, target_variable, only_X = True)
plt.subplot(6, 2, 6)
plot_encoded_diff_colours(X_leave, which_category, method, ind_0, ind_1)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)

    
##### the CatBoost encoded dataset 
method = 'CAT'
encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
cat_df = encoder.fit_transform(df_train, df_train[target_variable])
X_cat=  dataset_to_Xandy(cat_df, target_variable, only_X = True)
encoded_df_test = encoder.transform(df_test)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(encoded_df_test, which_category, target_variable)


final_cat_df = cat_df.copy()
how_many_permutations = 5

for s in range(how_many_permutations):
    index_dataset = np.arange(0, df_train.shape[0])
    np.random.shuffle(index_dataset)
    encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)

    cat_df_shuffled = encoder.fit_transform(df_train.iloc[index_dataset], df_train.iloc[index_dataset][target_variable])
    back_cat_df = cat_df_shuffled.sort_index()
    final_cat_df[categorical_variables] +=  back_cat_df[categorical_variables]
    
final_cat_df[categorical_variables] /= how_many_permutations


cat_df_test_difftest = df_test.copy()
cat_df_test_shuffle = df_test.copy()


for col in categorical_variables:
    dictionary_cat={}
    dictionary_cat_shuffle = {}
    unique_cat = list(set(df_train[col]))

    for category in unique_cat:
        indices = np.where(df_train[col] == category)[0]
        part_dataset_encoded = cat_df.iloc[indices]
        get_numer = np.sum(part_dataset_encoded[col]) + alpha * prior
        dictionary_cat[category] =  get_numer / (len(indices) + alpha)
        
        part_dataset_encoded_final = final_cat_df.iloc[indices]
        get_numer = np.sum(part_dataset_encoded_final[col]) + alpha * prior
        dictionary_cat_shuffle[category] =  get_numer / (len(indices) + alpha)     
        
        cat_df_test_difftest[col] = cat_df_test_difftest[col].replace(list(dictionary_cat.keys()), list(dictionary_cat.values()))
        cat_df_test_shuffle[col] = cat_df_test_shuffle[col].replace(list(dictionary_cat_shuffle.keys()), list(dictionary_cat_shuffle.values()))
   
    unique_test_no_train = list(set(df_test[col]) - set(df_train[col]))
    
    for uni in unique_test_no_train:
        cat_df_test_difftest[cat_df_test_difftest[col] == uni] = cat_df_test_difftest[cat_df_test_difftest[col] == uni].replace(uni, prior)
        cat_df_test_shuffle[cat_df_test_shuffle[col] == uni] = cat_df_test_shuffle[cat_df_test_shuffle[col] == uni].replace(uni, prior)
        


X_cat_shuffle=  dataset_to_Xandy(final_cat_df, target_variable, only_X = True)
plt.subplot(6, 2, 7)
plot_encoded_diff_colours(X_cat, which_category, method, ind_0, ind_1)

method = 'CAT_T'
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(cat_df_test_difftest, which_category, target_variable)

method ='CAT_S_5'
plt.subplot(6, 2, 8)
plot_encoded_diff_colours(X_cat_shuffle, which_category, method , ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(cat_df_test_shuffle, which_category, target_variable)


#### the 10-Fold Target Encoding
method = 'TAR_10'
modified_df10, modified_df_test10 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='target')
X_target10 =  dataset_to_Xandy(modified_df10,  target_variable, only_X = True)
plt.subplot(6, 2, 9)
plot_encoded_diff_colours(X_target10, which_category, method, ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(modified_df_test10, which_category, target_variable)


#### the 5-Fold Target Encoding
method = 'TAR_5'
modified_df5, modified_df_test5 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='target')
X_target5 =  dataset_to_Xandy(modified_df5, target_variable, only_X = True)
plt.subplot(6, 2, 10)
plot_encoded_diff_colours(X_target5, which_category, method, ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(modified_df_test5, which_category, target_variable)


#### the 10-Fold GLMM Encoding
method = 'GLMM_10'
glmm_modified_df10,  glmm_modified_df_test10 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='glmm')
X_glmm10 =  dataset_to_Xandy(glmm_modified_df10, target_variable, only_X = True)
plt.subplot(6, 2, 11)
plot_encoded_diff_colours(X_glmm10, which_category, 'GLMM_10', ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(glmm_modified_df_test10, which_category, target_variable)



#### the 5-Fold GLMM Encoding
method = 'GLMM_5'
glmm_modified_df5,  glmm_modified_df_test5 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='glmm')
X_glmm5 =  dataset_to_Xandy(glmm_modified_df5, target_variable, only_X = True)
plt.subplot(6, 2, 12)
plot_encoded_diff_colours(X_glmm5, which_category, method , ind_0, ind_1)
dict_tpr[method],  dict_fpr[method]= decboundary_tprfpr(glmm_modified_df_test5, which_category, target_variable)

plt.show()


colours = ['black','tab:blue','tab:orange','tab:green','tab:red', 'tab:pink','tab:brown','tab:purple','tab:cyan', 'tab:olive','tab:gray','blue','gold','purple','magenta','green']
x = 0
for key in dict_fpr.keys():
    plt.plot(dict_fpr[key], dict_tpr[key],'.', label = key, color = colours[x])
    x += 1
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
    







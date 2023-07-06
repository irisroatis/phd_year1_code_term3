#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:43:00 2023

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

def whole_process(df, df_test, continuous_variables, target_variable, which_dataset, how_to_bin, nr_bins):
    
    how_many_rows = df.shape[0]
    how_many_rows_test = df_test.shape[0]
    binned_df, binned_df_test = create_binned_dataset(df, df_test, continuous_variables, how_many_rows, how_many_rows_test, nr_bins, how_to_bin)

    # We standardise original dataset
    for cont_col in continuous_variables:
        df[cont_col] = standardise(df[cont_col])
        df_test[cont_col] = standardise(df_test[cont_col])
    
    
    X_train, y_train =  dataset_to_Xandy(df, target_variable, only_X = False) ###### the original dataset
    X_bin =  dataset_to_Xandy(binned_df, target_variable, only_X = True) ###### the binned dataset
    
    X_test, y_test =  dataset_to_Xandy(df_test, target_variable, only_X = False) ###### the original dataset
    X_bin_test=  dataset_to_Xandy(binned_df_test, target_variable, only_X = True) ###### the binned dataset
    
    
    ##### the simple encoded dataset (after binned)
    labelencoder = ce.OrdinalEncoder(cols=continuous_variables)
    simple_df = labelencoder.fit_transform(binned_df)
    simple_df_test =  labelencoder.transform(binned_df_test)
    X_simple =  dataset_to_Xandy(simple_df, target_variable, only_X = True)
    X_simple_test=  dataset_to_Xandy(simple_df_test, target_variable, only_X = True)
    
    ##### the one hot encoded dataset (after binned)
    encoder = ce.OneHotEncoder(cols=continuous_variables,use_cat_names=True)
    onehot_df = encoder.fit_transform(binned_df)
    onehot_df_test = encoder.transform(binned_df_test)
    X_onehot =  dataset_to_Xandy(onehot_df, target_variable, only_X = True)
    X_onehot_test =  dataset_to_Xandy(onehot_df_test, target_variable, only_X = True)
    
    ##### the effect encoded dataset (after binned)
    encoder = ce.sum_coding.SumEncoder(cols=continuous_variables,verbose=False)
    effect_df = encoder.fit_transform(binned_df)
    effect_df_test = encoder.transform(binned_df_test)
    X_effect =  dataset_to_Xandy(effect_df, target_variable, only_X = True)
    X_effect_test =  dataset_to_Xandy(effect_df_test, target_variable, only_X = True)
    
    
    ##### the target encoded dataset (after binned)
    
    target_df = binned_df.copy()
    target_df_test = binned_df_test.copy()
    
    for col in continuous_variables:
        dict_target = {}
        target_df, dict_target =  target_encoding(col, target_variable, target_df)
        target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
        
        
    X_target =  dataset_to_Xandy(target_df, target_variable, only_X = True)
    X_target_test =  dataset_to_Xandy(target_df_test, target_variable, only_X = True)
    
    ##### the GLMM encoded dataset (after binned)
    encoder = ce.glmm.GLMMEncoder(cols=continuous_variables,verbose=False)
    glmm_df = encoder.fit_transform(binned_df, binned_df[target_variable])
    glmm_df_test = encoder.transform(binned_df_test)
    X_glmm =  dataset_to_Xandy(glmm_df, target_variable, only_X = True)
    X_glmm_test =  dataset_to_Xandy(glmm_df_test, target_variable, only_X = True)
    
    
    ##### the GLMM encoded dataset (after binned)
    encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=continuous_variables,verbose=False)
    leave_df = encoder.fit_transform(binned_df, binned_df[target_variable])
    leave_df_test = encoder.transform(binned_df_test)
    X_leave=  dataset_to_Xandy(leave_df, target_variable, only_X = True)
    X_leave_test =  dataset_to_Xandy(leave_df_test, target_variable, only_X = True)
    
    
    confusion_matrix = []
    

    for method in methods:
        
        # initialising confusion matrices
        
        this = []
        
        for classifier in classifiers:
        
            if method == 'not binned':
                auc = calc_conf_matrix(X_train,y_train,X_test,y_test,classifier)
            elif method == 'binned':
                auc = calc_conf_matrix(X_bin,y_train,X_bin_test,y_test, classifier)
            elif method == 'bin_test':
                auc = calc_conf_matrix(X_train,y_train,X_bin_test, y_test, classifier)        
            elif method == 'simple':
                auc =  calc_conf_matrix(X_simple,y_train,X_simple_test, y_test, classifier)
            elif method == 'onehot':
                auc = calc_conf_matrix(X_onehot,y_train,X_onehot_test,y_test, classifier)
            elif method == 'effect':
                auc = calc_conf_matrix(X_effect,y_train,X_effect_test,y_test, classifier)
            elif method == 'target':
                auc = calc_conf_matrix(X_target,y_train,X_target_test,y_test, classifier)
            elif method == 'glmm':
                auc = calc_conf_matrix(X_glmm,y_train,X_glmm_test,y_test, classifier)
            elif method == 'leave':
                auc = calc_conf_matrix(X_leave,y_train,X_leave_test,y_test, classifier)
            
            this.append(np.round(auc,3))
            
        confusion_matrix.append(this)
    
    return confusion_matrix
    # plot_boxplots_confusion(confusion_matrix, 'accuracy', which_dataset, classifier, how_to_bin, nr_bins)

    


methods = ['not binned','bin_test','binned','simple','onehot','target','effect', 'glmm','leave']
classifiers = ['logistic','kNN','dec_tree','rand_for','grad_boost','naive']


# which_dataset = 'Simulated Dataset'
# how_many_rows = 500
# how_many_rows_test = 10000


# # e1 = [0,0]
# # e2 = [1,1]
# # std1 = np.array(([1,0],[0,1]))
# # std2 =  np.array(([1,0],[0,1]))


# # testing_data, belonging_classes = generating_test_data(1, how_many_rows, e1, std1,e2, std2)
# # d = {'feature1':testing_data[0][:,0], 'feature2':testing_data[0][:,1],'target':belonging_classes[0]}
# # df = pd.DataFrame(data=d)
# # df.to_csv('simulated_dataset.csv', index=False)
# # testing_data_test, belonging_classes_test = generating_test_data(1, how_many_rows_test, e1, std1,e2, std2)
# # d_test = {'feature1':testing_data_test[0][:,0], 'feature2':testing_data_test[0][:,1],'target':belonging_classes_test[0]}
# # df_test = pd.DataFrame(data=d_test)
# # df_test.to_csv('simulated_dataset_test.csv', index=False)



# df = pd.read_csv('simulated_dataset.csv')
# df_test = pd.read_csv('simulated_dataset_test.csv')


# continuous_variables=['feature1', 'feature2']
# target_variable = 'target'
# categorical_variables = []

# how_to_bin ='fixed_number'
# nr_bins = 40
# confusion_matrix = whole_process(df, df_test, continuous_variables, target_variable, which_dataset, how_to_bin, nr_bins)
# array_confusion_matrix = np.array(confusion_matrix)


################################################################################################################################################
########################################################################
########### WINE QUALITY

# which_dataset = 'Wine Quality'
# df = pd.read_csv('wine_dataset.csv')
# target_variable = 'quality' # Making sure the name of the target variable is known
# df[target_variable] = df[target_variable].replace(['bad'], 0)
# df[target_variable] = df[target_variable].replace(['good'], 1)

########################################################################




########################################################################
########### BODY SIGNAL SMOKING
which_dataset = ' BODY SIGNAL SMOKING'
df = pd.read_csv('bodysignal_smoking.csv')
df = df.drop(['ID','oral'],axis = 1)


target_variable = 'smoking' # Making sure the name of the target variable is known
binary_cols = ['tartar','dental caries','hearing(right)','hearing(left)','gender']

df = pick_only_some(df, target_variable, 4000)
df = df.reset_index(drop=True)

### make sure binary variables are 0 and 1
labelencoder = ce.OrdinalEncoder(cols=binary_cols)
df = labelencoder.fit_transform(df)
########################################################################




how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s
continuous_variables = set(df.columns)
continuous_variables.remove(target_variable)


how_many_cv = 5
how_to_bin ='fixed_number'
nr_bins = 10

conf_matrix_list = []

array_confusion_matrix = np.zeros((len(methods),len(classifiers)))
for index in range(how_many_cv):

    randomlist = random.sample(list(df[df[target_variable]==0].index.values), 4 * how_many_0s // 5) + random.sample(list(df[df[target_variable]==1].index.values), 4 * how_many_1s // 5)
    not_in_randomlist = list(set(range(0,size)) - set(randomlist))
    
    
    df_test = df.iloc[not_in_randomlist,:]
    df_train = df.iloc[randomlist,:]
    confusion_matrix = whole_process(df_train, df_test, continuous_variables, target_variable, which_dataset, how_to_bin, nr_bins)
    array_confusion_matrix += np.array(confusion_matrix)
    conf_matrix_list.append(np.array(confusion_matrix))


array_confusion_matrix /= how_many_cv
###################

# if score is wanted, probably not the goal of our project


score_matrix = np.zeros((len(methods), len(classifiers)+1))
min_overall = np.min(np.min(array_confusion_matrix, axis = 1), axis = 0)
q3, q1 = np.percentile(array_confusion_matrix, [75 ,25])
iqr = q3 - q1

score_matrix[:,:-1]  =( array_confusion_matrix - min_overall ) / iqr
score_matrix[:,-1] = np.mean(score_matrix, axis = 1)


################################################################################################################################################

mi_array = np.zeros_like(array_confusion_matrix)
ma_array = np.zeros_like( array_confusion_matrix)
for i1 in range(len(methods)):
    for i2 in range(len(classifiers)):
        mi = 1
        ma = 0
        for cv in range(how_many_cv):
            value = conf_matrix_list[cv][i1,i2]
            if ma < value:
                ma = value
            if mi > value:
                mi = value
        mi_array[i1,i2] = mi
        ma_array[i1,i2] = ma

from prettytable import PrettyTable
 
# Specify the Column Names while initializing the Table
myTable = PrettyTable([which_dataset]+classifiers+['MEAN SCORE'])

colours = ['tab:blue','tab:orange','tab:green','tab:red', 'tab:pink','tab:brown','tab:purple','tab:cyan', 'tab:olive','tab:gray']

plt.figure(figsize=(10,7))
for method_index in range(len(methods)):
    myTable.add_row([methods[method_index]]+ list(np.round(score_matrix[method_index,:],5)))
    plt.plot(np.arange(method_index,78,13),array_confusion_matrix[method_index,:],'.',color = colours[method_index], label = str(methods[method_index]))
    plt.plot(np.arange(method_index,78,13),mi_array[method_index,:],'_', color = colours[method_index])
    plt.plot(np.arange(method_index,78,13),ma_array[method_index,:],'_', color = colours[method_index])
print(myTable)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.arange(4,78,13),labels = classifiers)
plt.title('Dataset: ' + str(which_dataset) +' \n h = ' +str(nr_bins))
plt.show()


import seaborn as sns
plt.figure(figsize=(10,7))
g = sns.heatmap(score_matrix, annot=True, fmt=".5f")
g.set_xticklabels(classifiers+['MEAN SCORE'], rotation = 45)
g.set_yticklabels(methods, rotation = 45)
plt.title('Dataset: ' + str(which_dataset) +' \n h = ' +str(nr_bins))
plt.show()




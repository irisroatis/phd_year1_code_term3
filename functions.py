#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:32:26 2023

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

def dataset_to_Xandy(dataset, target_variable, only_X = True):
    X = dataset.loc[:, dataset.columns != target_variable]
    y = dataset.loc[:, dataset.columns == target_variable]
    if only_X:
        return X
    else:
        return X, y

def standardise(X):
    return (X - np.mean(X)) / np.std(X)

def split_dataset(X,y, randomlist, not_in_randomlist):
    X_train = X.iloc[randomlist,:]
    y_train = y.iloc[randomlist,:]
    X_test = X.iloc[not_in_randomlist,:]
    y_test = y.iloc[not_in_randomlist,:]
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


def create_bins(data, data_test, how_many_bins, how_to_bin):
    mi = min(min(data), min(data_test))
    ma = max(max(data), max(data_test))
    if how_to_bin =='cons_std':
        bin_size = how_many_bins * np.std(data)
        start = (mi + ma)/2

        bins_right = [start]
        current_right = 1.0 * start
        while current_right  < ma:
            current_right += bin_size
            bins_right.append(current_right)

        bins_left = []
        current_left = 1.0 * start
        while current_left  > mi:
            current_left -= bin_size
            bins_left.append(current_left)

        bins = np.concatenate((bins_left[::-1],bins_right))
    elif how_to_bin =='fixed_number':
        bins = np.linspace(mi-0.00000001, ma+0.00000001,how_many_bins)
    else:
        assert('Way of binning unknown')
    return bins
    
def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data
 

def target_encoding(feature, target,df):
    dictionary_target_encoding = {}
    categories = df[feature].unique()
    changed_df = df.copy()
    for cat in categories:
        which_cat = df[df[feature] == cat]
        avg_value = which_cat[target].mean()
        changed_df[feature] = changed_df[feature].replace([cat], avg_value)
        dictionary_target_encoding[cat] = avg_value
    return changed_df, dictionary_target_encoding
   
def generating_test_data(how_many_times_repeat, iterations, mu1, sigma1, mu2, 
                         sigma2, plot_classes = False):

    dim = len(mu1)
    testing_data=[]
    belonging_classes=[]

    for repeat in range(how_many_times_repeat):

        random_simulation = np.zeros((iterations,dim))
        which_class_list = np.zeros((iterations,))
        
        for itera in range(iterations):

            which_normal = random.randint(0,1)
            if dim == 1:
                if which_normal == 0:
                    random_simulation[itera,] = np.random.normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.normal(mu2, sigma2)
            else:
                if which_normal == 0:
                    random_simulation[itera,] = np.random.multivariate_normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.multivariate_normal(mu2, sigma2)
            which_class_list[itera,] = which_normal
        
        testing_data.append(random_simulation)
        belonging_classes.append(which_class_list)
      
    
    return testing_data, belonging_classes

def pick_only_some(df, target_variable, number):
    df0 = df.loc[df[target_variable] ==0 ]
    df1 = df.loc[df[target_variable] ==1 ]
    how_many_0 = df0.shape[0]
    how_many_1 = df1.shape[0]
    random_indices = random.sample(range(0, how_many_0), how_many_0 - number)
    df0 = df0.drop(df0.index[random_indices])
    random_indices = random.sample(range(0, how_many_1), how_many_1 - number)
    df1 = df1.drop(df1.index[random_indices])
    df = pd.concat([df0, df1])
    return df

def calc_conf_matrix(X_train,y_train,X_test, y_test,classifier):
    
    # depending on what the user inputs
    if classifier == 'logistic':
        model = LogisticRegression(penalty = 'none')  
    elif classifier == 'kNN':
        model = KNeighborsClassifier()  
    elif classifier == 'dec_tree':
        model = DecisionTreeClassifier()
    elif classifier == 'rand_for':
        model = RandomForestClassifier()
    elif  classifier == 'grad_boost':
        model = GradientBoostingClassifier()
    elif classifier == 'naive':
        model= GaussianNB()
    else:
        assert('Classifier unknown')
    
    # perform fitting of the model 
    model.fit(X_train, y_train)   #.reshape(-1,)
    
    y_predicted = model.predict(X_test) 
        
    
    # computing confusion matrix, fpr, tpr, auc
    # matrix = metrics.confusion_matrix(y_test, y_predicted)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
    area_roc = metrics.auc(fpr, tpr)
    # return matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1], area_roc
    return area_roc

def plot_boxplots_confusion(confusion_matrix,entry,which_dataset,classifier, how_to_bin, nr_bins):
    dictionary = {}
    for key in confusion_matrix:
        dictionary[key] = confusion_matrix[key][entry]
    fig, ax = plt.subplots()
    ax.boxplot(dictionary.values())
    ax.set_xticklabels(dictionary.keys())
    if entry == '00':
        name = 'true negative'
    elif entry == '01':
        name = 'false positive'
    elif entry == '10':
        name = 'false negative'
    elif entry == '11':
        name = 'true positive'
    elif entry == 'auc':
        name = 'area under ROC curve'
    plt.title('Boxplots of ' + name +' \n Test Size '+ str(how_many_rows_test)+'\n Dataset:'+which_dataset+'\n Classifier:'+classifier+'\n Binning:'+str(how_to_bin)+', h = '+str(nr_bins))
    plt.show()    
    
def create_binned_dataset(df, df_test, continuous_variables, how_many_rows,  how_many_rows_test, nr_bins, how_to_bin):
    bins = {}
    for fea in continuous_variables:
        bins[fea] =  create_bins(df[fea], df_test[fea], nr_bins, how_to_bin)
    
    result = pd.concat([df,df_test])
    for fea in continuous_variables:
        result[fea] = standardise(put_in_bins(result[fea],bins[fea]))

    binned_df = result.head(how_many_rows)
    binned_df_test = result.tail(how_many_rows_test) 
    
    return binned_df, binned_df_test
   
    
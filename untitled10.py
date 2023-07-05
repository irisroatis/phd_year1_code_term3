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
# nr_bins = 10
# confusion_matrix = whole_process(df, df_test, continuous_variables, target_variable, which_dataset, how_to_bin, nr_bins)
# array_confusion_matrix = np.array(confusion_matrix)


################################################################################################################################################
########################################################################
########### WINE QUALITY

which_dataset = 'Wine Quality'
df = pd.read_csv('wine_dataset.csv')
target_variable = 'quality' # Making sure the name of the target variable is known
df[target_variable] = df[target_variable].replace(['bad'], 0)
df[target_variable] = df[target_variable].replace(['good'], 1)

########################################################################




########################################################################
########### BODY SIGNAL SMOKING
# which_dataset = ' BODY SIGNAL SMOKING'
# df = pd.read_csv('bodysignal_smoking.csv')
# df = df.drop(['ID','oral'],axis = 1)


# target_variable = 'smoking' # Making sure the name of the target variable is known
# binary_cols = ['tartar','dental caries','hearing(right)','hearing(left)','gender']

# df = pick_only_some(df, target_variable, 4000)
# df = df.reset_index(drop=True)

# ### make sure binary variables are 0 and 1
# labelencoder = ce.OrdinalEncoder(cols=binary_cols)
# df = labelencoder.fit_transform(df)
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
myTable = PrettyTable([which_dataset]+classifiers)

colours = ['tab:blue','tab:orange','tab:green','tab:red', 'tab:pink','tab:brown','tab:purple','tab:cyan', 'tab:olive','tab:gray']

plt.figure(figsize=(10,7))
for method_index in range(len(methods)):
    myTable.add_row([methods[method_index]]+ list(np.round(array_confusion_matrix[method_index,:],5)))
    plt.plot(np.arange(method_index,78,13),array_confusion_matrix[method_index,:],'.',color = colours[method_index], label = str(methods[method_index]))
    plt.plot(np.arange(method_index,78,13),mi_array[method_index,:],'_', color = colours[method_index])
    plt.plot(np.arange(method_index,78,13),ma_array[method_index,:],'_', color = colours[method_index])
print(myTable)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.arange(4,78,13),labels = classifiers)
plt.show()


import seaborn as sns
g = sns.heatmap(array_confusion_matrix, annot=True, fmt=".5f")
g.set_xticklabels(classifiers, rotation = 45)
g.set_yticklabels(methods, rotation = 45)
plt.show()




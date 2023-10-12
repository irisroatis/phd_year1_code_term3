#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 20:30:16 2023

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




#### SIMULATED DATASET

which_dataset = 'Simulated Data One Dimension'
# which_dataset = 'Uniform and Beta Distribution'
df, categorical_variables, continuous_variables, binary_cols, target_variable = dataset_variables(which_dataset)

which_category = 'Feature_1'
see =  pd.DataFrame(df[which_category].value_counts())
table = tabulate(see, headers= ['Category','Count'])




# bins = np.loadtxt('bins.txt', delimiter= ',')
# order_cat = np.loadtxt('order_cat.txt', delimiter= ',')

# bins = np.loadtxt('bins_unif_beta.txt', delimiter= ',')
# order_cat = np.loadtxt('order_cat_unif_beta.txt', delimiter= ',')
bins = np.loadtxt('bins_two_normals.txt', delimiter= ',')
order_cat = np.loadtxt('order_cat_two_normals.txt', delimiter= ',')




unique_cat = list(set(df[which_category]))
cat_and_encoded = pd.DataFrame(unique_cat, columns = [which_category])
cat_and_encoded['target'] = np.nan
# cat_and_encoded['Feature_2'] = np.nan
# cat_and_encoded['Feature_1'] = np.nan
intial_features = cat_and_encoded.keys()

how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s

alpha = 1
prior = how_many_1s/ size

df_train = df


##### the WOE encoded dataset 
encoder = ce.woe.WOEEncoder(cols=categorical_variables,verbose=False)
woe_df = encoder.fit_transform(df_train, df_train[target_variable])
cat_and_encoded['WOE'] = encoder.transform(cat_and_encoded[intial_features])[which_category]


target_df = df.copy()
target_df_test =  cat_and_encoded[intial_features]
for col in categorical_variables:
    dict_target = {}
    target_df, dict_target =  target_encoding(col, target_variable, target_df)
    target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
X_target_test =  dataset_to_Xandy(target_df_test, target_variable, only_X = True)
cat_and_encoded['TAR'] = X_target_test[which_category]

    

#### the target weighted encoded dataset 
encoder = ce.target_encoder.TargetEncoder(cols = categorical_variables, verbose=False)
target_w_df = encoder.fit_transform(df_train, df_train[target_variable])
cat_and_encoded['TAR_W'] = encoder.transform(cat_and_encoded[intial_features])[which_category]


##### the GLMM encoded dataset 
encoder = ce.glmm.GLMMEncoder(cols=categorical_variables,verbose=False)
glmm_df = encoder.fit_transform(df_train, df_train[target_variable])
cat_and_encoded['GLMM']= encoder.transform(cat_and_encoded[intial_features])[which_category]


##### the leave-one-out encoded dataset 
encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_variables,verbose=False)
leave_df = encoder.fit_transform(df_train, df_train[target_variable])
X_leave=  dataset_to_Xandy(leave_df, target_variable, only_X = True)

leave_df_difftest = cat_and_encoded[intial_features]
leave_df_sametest = cat_and_encoded[intial_features]

for col in categorical_variables:
    dictionary_loo = {}
    dictionary_loo_prior = {}
    unique_cat = list(set(df_train[col]))
    for category in unique_cat:
        indices = np.where(df_train[which_category] == category)[0]
        part_dataset_encoded = leave_df.iloc[indices]
        get_number = np.sum(part_dataset_encoded[col]) + alpha * prior
        dictionary_loo[category] = get_number / (alpha + len(indices)) 
        
        part_dataset = df.iloc[indices]
        get_proportion =len( np.where(part_dataset['target']==1)[0]) / len(indices)
        dictionary_loo_prior[category] = get_proportion
        
        
        leave_df_difftest[col] = leave_df_difftest[col].replace(list(dictionary_loo.keys()), list(dictionary_loo.values()))
        leave_df_sametest[col] = leave_df_sametest[col].replace(list(dictionary_loo_prior.keys()), list(dictionary_loo_prior.values()))

    cat_and_encoded['LOO_T'] = leave_df_difftest[col]
    cat_and_encoded['LOO'] = leave_df_sametest[col]
    



##### the CatBoost encoded dataset 
encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
cat_df = encoder.fit_transform(df_train, df_train[target_variable])
cat_and_encoded['CAT'] = encoder.transform(cat_and_encoded[intial_features])[which_category]

cat_df_test_difftest = cat_and_encoded[intial_features]
cat_df_test_shuffle = cat_and_encoded[intial_features]

final_cat_df = cat_df.copy()
how_many_permutations = 10

for s in range(how_many_permutations):
    index_dataset = np.arange(0, df.shape[0])
    np.random.shuffle(index_dataset)
    encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)

    cat_df_shuffled = encoder.fit_transform(df_train.iloc[index_dataset], df_train.iloc[index_dataset][target_variable])
    back_cat_df = cat_df_shuffled.sort_index()
    final_cat_df[categorical_variables] +=  back_cat_df[categorical_variables]
    
final_cat_df[categorical_variables] /= how_many_permutations



for col in categorical_variables:
    dictionary_cat={}
    dictionary_cat_shuffle = {}
    unique_cat = list(set(df[col]))

    for category in unique_cat:
        indices = np.where(df[col] == category)[0]
        part_dataset_encoded = cat_df.iloc[indices]
        get_numer = np.sum(part_dataset_encoded[col]) + alpha * prior
        dictionary_cat[category] =  get_numer / (len(indices) + alpha)
        
        part_dataset_encoded_final = final_cat_df.iloc[indices]
        get_numer = np.sum(part_dataset_encoded_final[col]) + alpha * prior
        dictionary_cat_shuffle[category] =  get_numer / (len(indices) + alpha)     
        
        cat_df_test_difftest[col] = cat_df_test_difftest[col].replace(list(dictionary_cat.keys()), list(dictionary_cat.values()))
        cat_df_test_shuffle[col] = cat_df_test_shuffle[col].replace(list(dictionary_cat_shuffle.keys()), list(dictionary_cat_shuffle.values()))

    cat_and_encoded['CAT_T'] = cat_df_test_difftest[col]
    cat_and_encoded['CAT_S_10'] = cat_df_test_shuffle[col]



#### the 10-Fold Target Encoding
modified_df10, modified_df_test10 = k_fold_target_encoding(df_train, cat_and_encoded[intial_features], categorical_variables, target_variable, how_many_folds=10, which_encoder='target')
X_target_test10 =  dataset_to_Xandy(modified_df_test10, target_variable, only_X = True)
cat_and_encoded['TAR_10']  = X_target_test10[which_category]

#### the 5-Fold Target Encoding
modified_df5, modified_df_test5 = k_fold_target_encoding(df_train, cat_and_encoded[intial_features], categorical_variables, target_variable, how_many_folds=5, which_encoder='target')
X_target_test5 =  dataset_to_Xandy(modified_df_test5, target_variable, only_X = True)
cat_and_encoded['TAR_5']  = X_target_test5[which_category]

#### the 10-Fold GLMM Encoding
glmm_modified_df10,  glmm_modified_df_test10 = k_fold_target_encoding(df_train, cat_and_encoded[intial_features], categorical_variables, target_variable, how_many_folds=10, which_encoder='glmm')
X_glmm_test10 =  dataset_to_Xandy(glmm_modified_df_test10, target_variable, only_X = True)
cat_and_encoded['GLMM_10']  = X_glmm_test10[which_category]


#### the 5-Fold GLMM Encoding
glmm_modified_df5,  glmm_modified_df_test5 = k_fold_target_encoding(df_train, cat_and_encoded[intial_features], categorical_variables, target_variable, how_many_folds=5, which_encoder='glmm')
X_glmm_test5 =  dataset_to_Xandy(glmm_modified_df_test5, target_variable, only_X = True)
cat_and_encoded['GLMM_5']  = X_glmm_test5[which_category]




methods = ['WOE','TAR','TAR_W','GLMM','LOO_T','LOO','CAT','CAT_T','CAT_S_10','TAR_10','TAR_5','GLMM_10','GLMM_5']
how_many_methods = len(methods)
orders = np.zeros((how_many_methods+1, len(unique_cat)))
orders[0,:] = order_cat



rank_categories_indices = np.argsort(orders[0,:])
diff_in_rank = []

plt.figure(figsize=(17,7))

for index in range(how_many_methods):
    method = methods[index]
    categories = cat_and_encoded[which_category] 
    # encoded_values = cat_and_encoded[method] 
    encoded_values = (cat_and_encoded[method] - np.mean(cat_and_encoded[method]) ) / np.std(cat_and_encoded[method])
    sort = np.argsort(encoded_values)
    sorted_encoded = np.array(encoded_values[sort])
    sorted_cat = np.array(categories[sort])
    orders[index+1,:] = sorted_cat
    some_zeros = np.zeros((len(categories),))
    plt.scatter(sorted_encoded,some_zeros+how_many_methods - 1 - index, label = methods[index], s = 50)
    for i in range(len(sorted_encoded)):
        plt.text(sorted_encoded[i],0+how_many_methods - 1 - index-0.01,sorted_cat[i], size='large', color='black')
        
    diff_in_rank.append(np.sum(abs(rank_categories_indices - sort))/len(unique_cat))
        
    
        
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Order of methods')
plt.yticks(np.arange(0,len(methods)), labels = methods[::-1])
plt.show()


all_plot = np.zeros((orders.shape[0],orders.shape[1]+1))
all_plot = np.zeros((orders.shape[0],orders.shape[1]))

# all_plot[:,:-1] = orders.astype(int)
all_plot[:,:] = orders.astype(int)

# all_plot[1:,-1] = diff_in_rank
# all_plot[0,-1] = np.nan







plt.figure(figsize=(15,7))
g = sns.heatmap(all_plot[1:,:], annot=True, cbar = False)
# g.set_xticklabels(list(orders[0,:])+['avg_r_diff'], rotation = 45)
g.set_xticklabels(list(orders[0,:]), rotation = 45)

g.set_yticklabels(methods, rotation = 45)
plt.title('Ranking of Encoded Values')
plt.show()

# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# fig.suptitle('Different number of folds, GLMM (no folds, 5, 10) and WOE')
# axs[0].hist(X_glmm[which_category],bins = 50)
# axs[1].hist(X_glmm5[which_category],bins = 50)
# axs[2].hist(X_glmm10[which_category],bins = 50)
# plt.show()

# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# fig.suptitle('Different number of folds, target')
# axs[0].hist(X_target[which_category],bins = 50)
# axs[1].hist(X_target5[which_category],bins = 50)
# axs[2].hist(X_target10[which_category],bins = 50)
# plt.show()

# fig, axs = plt.subplots(2, sharex=True, sharey=True)
# fig.suptitle('Leave-One-Out and Catboost encoders')
# axs[0].hist(X_leave[which_category],bins = 50)
# axs[1].hist(X_cat[which_category],bins = 50)
# plt.show()


# plt.hist(X_simple[which_category],bins = 50)
# plt.title('Simple encoding')
# plt.show()

# plt.hist(X_woe[which_category],bins = 50)
# plt.title('WOE encoding')
# plt.show()

# X_multiple_encoding = X_target.copy()
# X_multiple_encoding = X_multiple_encoding.rename({'area_cluster': 'area_cluster_target'}, axis='columns')
# X_multiple_encoding['area_cluster_target_5'] = X_target5['area_cluster']
# X_multiple_encoding['area_cluster_target_10'] = X_target10['area_cluster']
# X_multiple_encoding['area_cluster_glmm'] = X_glmm['area_cluster']
# X_multiple_encoding['area_cluster_glmm_5'] = X_glmm5['area_cluster']
# X_multiple_encoding['area_cluster_glmm_10'] = X_glmm10['area_cluster']
# model = DecisionTreeClassifier()
# model.fit(X_multiple_encoding, y_train)

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(model,
#                    feature_names=X_multiple_encoding.keys(), 
#                    class_names='target',
#                    filled=True)




# X_multiple_encoding = X_target.copy()
# X_multiple_encoding = X_multiple_encoding.rename({which_category: which_category+'_target'}, axis='columns')
# X_multiple_encoding[which_category+'_target_5'] = X_target5[which_category]
# X_multiple_encoding[which_category+'_target_10'] = X_target10[which_category]
# X_multiple_encoding[which_category+'_glmm'] = X_glmm[which_category]
# X_multiple_encoding[which_category+'_glmm5'] = X_glmm5[which_category]
# X_multiple_encoding[which_category+'_glmm10'] = X_glmm10[which_category]


# X_multiple_encoding_test = X_target_test.copy()
# X_multiple_encoding_test = X_multiple_encoding_test.rename({which_category: which_category+'_target'}, axis='columns')
# X_multiple_encoding_test[which_category+'dictionary_cat_target_5'] = X_target_test5[which_category]
# X_multiple_encoding_test[which_category+'_target_10'] = X_target_test10[which_category]
# X_multiple_encoding_test[which_category+'_glmm'] = X_glmm_test[which_category]
# X_multiple_encoding_test[which_category+'_glmm5'] = X_glmm_test5[which_category]
# X_multiple_encoding_test[which_category+'_glmm10'] = X_glmm_test10[which_category]

# model_ensemble = DecisionTreeClassifier(max_depth=5)
# model_ensemble.fit(X_multiple_encoding, y_train)
# y_predicted = model_ensemble.predict(X_multiple_encoding_test)
# fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
# area_roc_ensemble = metrics.auc(fpr, tpr)


# model_target = DecisionTreeClassifier(max_depth=5)
# model_target.fit(X_target, y_train)
# y_predicted = model_target.predict(X_target_test)
# fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
# area_roc_target = metrics.auc(fpr, tpr)



# fig = plt.figure(figsize=(50,40))dictionary_cat
# _ = tree.plot_tree(model_ensemble,
#                    feature_names=X_multiple_encoding.keys(), 
#                    class_names='target',
#                    filled=True)


# fig = plt.figure(figsize=(50,40))
# _ = tree.plot_tree(model_target,
#                    feature_names=X_multiple_encoding.keys(), 
#                    class_names='target',
#                    filled=True)






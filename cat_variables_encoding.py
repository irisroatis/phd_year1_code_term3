#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:29:19 2023

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


def whole_process_categorical(df, df_test, categorical_variables, continuous_variables, target_variable, which_dataset, how_many_permutations):
    
    how_many_rows = df.shape[0]
    how_many_rows_test = df_test.shape[0]

    # We standardise original dataset
    for cont_col in continuous_variables:
        df[cont_col] = standardise(df[cont_col])
        df_test[cont_col] = standardise(df_test[cont_col])
    
    df_wout_cat = df.drop(columns=categorical_variables)
    df_test_wout_cat = df_test.drop(columns=categorical_variables)
    
    X_train, y_train =  dataset_to_Xandy(df, target_variable, only_X = False) ###### the original dataset
    X_test, y_test =  dataset_to_Xandy(df_test, target_variable, only_X = False) ###### the original dataset
    
    
    X_nocat =  dataset_to_Xandy(df_wout_cat, target_variable, only_X = True)
    X_test_nocat =  dataset_to_Xandy(df_test_wout_cat, target_variable, only_X = True)

    ##### the simple encoded dataset 
    labelencoder = ce.OrdinalEncoder(cols=categorical_variables)
    simple_df = labelencoder.fit_transform(df)
    simple_df_test =  labelencoder.transform(df_test)
    X_simple =  dataset_to_Xandy(simple_df, target_variable, only_X = True)
    X_simple_test=  dataset_to_Xandy(simple_df_test, target_variable, only_X = True)
    
    ##### the one hot encoded dataset (after binned)
    encoder = ce.OneHotEncoder(cols=categorical_variables,use_cat_names=True)
    onehot_df = encoder.fit_transform(df)
    onehot_df_test = encoder.transform(df_test)
    X_onehot =  dataset_to_Xandy(onehot_df, target_variable, only_X = True)
    X_onehot_test =  dataset_to_Xandy(onehot_df_test, target_variable, only_X = True)
    
    ##### the effect encoded dataset (after binned)
    encoder = ce.sum_coding.SumEncoder(cols=categorical_variables,verbose=False)
    effect_df = encoder.fit_transform(df)
    effect_df_test = encoder.transform(df_test)
    X_effect =  dataset_to_Xandy(effect_df, target_variable, only_X = True)
    X_effect_test =  dataset_to_Xandy(effect_df_test, target_variable, only_X = True)
    
    ##### the WOE encoded dataset 
    encoder = ce.woe.WOEEncoder(cols=categorical_variables,verbose=False)
    woe_df = encoder.fit_transform(df, df[target_variable])
    woe_df_test = encoder.transform(df_test)
    X_woe =  dataset_to_Xandy(woe_df, target_variable, only_X = True)
    X_woe_test =  dataset_to_Xandy(woe_df_test, target_variable, only_X = True)
    
    

    ##### the target encoded dataset 
    
    target_df = df.copy()
    target_df_test = df_test.copy()
    
    for col in categorical_variables:
        dict_target = {}
        target_df, dict_target =  target_encoding(col, target_variable, target_df)
        target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
        
        
    X_target =  dataset_to_Xandy(target_df, target_variable, only_X = True)
    X_target_test =  dataset_to_Xandy(target_df_test, target_variable, only_X = True)
    
    
    
    ##### the target encoded dataset 
    

    encoder = ce.target_encoder.TargetEncoder(cols = categorical_variables, verbose=False)
    target_w_df = encoder.fit_transform(df, df[target_variable])
    target_w_test_df = encoder.transform(df_test)
    X_target_w =  dataset_to_Xandy(target_w_df, target_variable, only_X = True)
    X_target_w_test =  dataset_to_Xandy(target_w_test_df, target_variable, only_X = True)
    
 
    
    ##### the GLMM encoded dataset 
    encoder = ce.glmm.GLMMEncoder(cols=categorical_variables,verbose=False)
    glmm_df = encoder.fit_transform(df, df[target_variable])
    glmm_df_test = encoder.transform(df_test)
    X_glmm =  dataset_to_Xandy(glmm_df, target_variable, only_X = True)
    X_glmm_test =  dataset_to_Xandy(glmm_df_test, target_variable, only_X = True)
    
    

    ##### the leave-one-out encoded dataset 
    encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_variables,verbose=False)
    leave_df = encoder.fit_transform(df, df[target_variable])
    leave_df_test = encoder.transform(df_test)
    X_leave=  dataset_to_Xandy(leave_df, target_variable, only_X = True)
    X_leave_test =  dataset_to_Xandy(leave_df_test, target_variable, only_X = True)
    
        
    ##### the CatBoost encoded dataset 
    encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
    cat_df = encoder.fit_transform(df, df[target_variable])
    cat_df_test = encoder.transform(df_test)
    X_cat=  dataset_to_Xandy(cat_df, target_variable, only_X = True)
    X_cat_test =  dataset_to_Xandy(cat_df_test, target_variable, only_X = True)
    
    cat_df_test_difftest = df_test.copy()
    cat_df_test_shuffle = df_test.copy()
    dictionary_cat={}
    dictionary_cat_shuffle = {}
    
    final_cat_df = cat_df.copy()
    how_many_permutations = 5
    
    for s in range(how_many_permutations):
        index_dataset = np.arange(0, df.shape[0])
        np.random.shuffle(index_dataset)
        encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)

        cat_df_shuffled = encoder.fit_transform(df.iloc[index_dataset], df.iloc[index_dataset][target_variable])
        back_cat_df = cat_df_shuffled.sort_index()
        final_cat_df[categorical_variables] +=  back_cat_df[categorical_variables]
        
    final_cat_df[categorical_variables] /= how_many_permutations
    
    how_many_1s = len(df[df[target_variable] == 1])
    alpha = 1
    prior = how_many_1s / df.shape[0]
    
    
    for col in categorical_variables:
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
       
        unique_test_no_train = list(set(df_test[col]) - set(df[col]))
        
        for uni in unique_test_no_train:
            cat_df_test_difftest[cat_df_test_difftest[col] == uni] = cat_df_test_difftest[cat_df_test_difftest[col] == uni].replace(uni, prior)
            cat_df_test_shuffle[cat_df_test_shuffle[col] == uni] = cat_df_test_shuffle[cat_df_test_shuffle[col] == uni].replace(uni, prior)
            
    X_cat_test_difftest =  dataset_to_Xandy(cat_df_test_difftest, target_variable, only_X = True)
    X_cat_test_shuffle =  dataset_to_Xandy(cat_df_test_shuffle, target_variable, only_X = True)
    X_cat_shuffle=  dataset_to_Xandy(final_cat_df, target_variable, only_X = True)
    
    
    #### the 10-Fold Target Encoding
    modified_df10, modified_df_test10 = k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='target')
    X_target10 =  dataset_to_Xandy(modified_df10, target_variable, only_X = True)
    X_target_test10 =  dataset_to_Xandy(modified_df_test10, target_variable, only_X = True)

    #### the 5-Fold Target Encoding
    modified_df5, modified_df_test5 = k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='target')
    X_target5 =  dataset_to_Xandy(modified_df5, target_variable, only_X = True)
    X_target_test5 =  dataset_to_Xandy(modified_df_test5, target_variable, only_X = True)


    #### the 10-Fold GLMM Encoding
    glmm_modified_df10,  glmm_modified_df_test10 = k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='glmm')
    X_glmm10 =  dataset_to_Xandy(glmm_modified_df10, target_variable, only_X = True)
    X_glmm_test10 =  dataset_to_Xandy(glmm_modified_df_test10, target_variable, only_X = True)

    #### the 5-Fold GLMM Encoding
    glmm_modified_df5,  glmm_modified_df_test5 = k_fold_target_encoding(df, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='glmm')
    X_glmm5 =  dataset_to_Xandy(glmm_modified_df5, target_variable, only_X = True)
    X_glmm_test5 =  dataset_to_Xandy(glmm_modified_df_test5, target_variable, only_X = True)

    confusion_matrix = []
    

    for method in methods:
        
        # initialising confusion matrices
        
        this = []
        
        for classifier in classifiers:
            
            print(str(method) + ' '+str(classifier))
        
            if method == 'remove_cat':
                auc = calc_conf_matrix(X_nocat,y_train,X_test_nocat,y_test,classifier)
            elif method == 'simple':
                auc =  calc_conf_matrix(X_simple,y_train,X_simple_test, y_test, classifier)
            elif method == 'onehot':
                auc = calc_conf_matrix(X_onehot,y_train,X_onehot_test,y_test, classifier)
            elif method == 'effect':
                auc = calc_conf_matrix(X_effect,y_train,X_effect_test,y_test, classifier)
            elif method == 'woe':
                auc = calc_conf_matrix(X_woe,y_train,X_woe_test,y_test, classifier)
            elif method == 'target':
                auc = calc_conf_matrix(X_target,y_train,X_target_test,y_test, classifier)
            elif method == 'target_w':
                auc = calc_conf_matrix(X_target_w,y_train,X_target_w_test,y_test, classifier)
            elif method == 'glmm':
                auc = calc_conf_matrix(X_glmm,y_train,X_glmm_test,y_test, classifier)
            elif method == 'leave':
                auc = calc_conf_matrix(X_leave,y_train,X_leave_test,y_test, classifier)
            elif method == 'catboost':
                auc = calc_conf_matrix(X_cat,y_train,X_cat_test,y_test, classifier)
            elif method == 'catboost_difftest':
                auc = calc_conf_matrix(X_cat,y_train,X_cat_test_difftest,y_test, classifier)
            elif method == 'catboost_shuffle':
                auc = calc_conf_matrix(X_cat_shuffle,y_train,X_cat_test_shuffle,y_test, classifier)
            elif method =='target10fold':
                auc = calc_conf_matrix(X_target10,y_train,X_target_test10,y_test, classifier)
            elif method == 'target5fold':
                auc = calc_conf_matrix(X_target5,y_train,X_target_test5,y_test, classifier)
            elif method == 'glmm5fold':
                auc = calc_conf_matrix(X_glmm5,y_train,X_glmm_test5,y_test, classifier)
            elif method == 'glmm10fold':
                auc = calc_conf_matrix(X_glmm10,y_train,X_glmm_test10,y_test, classifier)
 
            this.append(np.round(auc,3))
            
        confusion_matrix.append(this)
    
    return confusion_matrix
    # plot_boxplots_confusion(confusion_matrix, 'accuracy', which_dataset, classifier, how_to_bin, nr_bins)



# methods = ['simple','onehot','effect','target','woe','glmm','leave','catboost']
methods = ['remove_cat','simple','onehot','effect','target','target_w','woe','glmm','leave','catboost','catboost_difftest','catboost_shuffle'] #'target5fold','target10fold','glmm5fold','glmm10fold']
# methods = ['catboost','catboost_difftest','catboost_shuffle']
# classifiers = ['logistic','kNN','dec_tree','rand_for','grad_boost','naive','lasso']
classifiers = ['logistic','kNN','dec_tree','rand_for','grad_boost']

########################################################################
######### HEART DATASET    

# which_dataset = 'Heart Ilness'
# df = pd.read_csv('heart.csv')
# categorical_variables = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
# target_variable = 'target' # Making sure the name of the target variable is known
# continuous_variables = ['age','trestbps','chol','thalach','oldpeak']
# binary_variables = ['sex','fbs','exang']

# df.head()
# #########################################################################
# ######### CHURN

# which_dataset = 'Churn'
# df = pd.read_csv('churn.csv')
# categorical_variables = ['state','area_code','number_customer_service_calls'] # Putting in this all the categorical columns
# target_variable = 'class' # Making sure the name of the target variable is known
# binary_variables = ['international_plan','voice_mail_plan']
# continuous_variables = list(set(df.keys()) - set(categorical_variables + [target_variable]))


#########################################################################
######### AMAZON

# which_dataset = 'Amazon_employee_access'
# df = pd.read_csv('amazon.csv')
# target_variable = 'target' # Making sure the name of the target variable is known
# categorical_variables = list(set(df.keys()) - set([target_variable]))
# df = pick_only_some(df, target_variable, 1000)
# df = df.reset_index(drop=True)

#########################################################################
######### MUSHROOM

# which_dataset = 'Mushroom'
# colnames = ['target','cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing','gill_size','gill_colour','stalk_shape','stalk_root','stalk_sur_ab_ring','stalk_sur_bw_ring','stalk_col_ab_ring','stalk_col_bw_ring','veil_type','veil_colour','ring_number','ring_type','spore_print_colour','population','habitat']
# df = pd.read_csv('mushrooms.data', names = colnames)
# target_variable = 'target' # Making sure the name of the target variable is known
# categorical_variables = colnames
# continuous_variables = []
# df[target_variable] = df[target_variable].replace(['e', 'p'], [1, 0])


#########################################################################
######### CLICK PREDICTION ADDS

# which_dataset = 'Click Prediction'
# df = pd.read_csv('click_prediction.csv')
# target_variable = 'click' # Making sure the name of the target variable is known
# categorical_variables = ['url_hash', 'ad_id', 'advertiser_id', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id']
# continuous_variables = list(set(df.keys()) - set(categorical_variables + [target_variable]))
# df = pick_only_some(df, target_variable, 1000)
# df = df.reset_index(drop=True)


#########################################################################
######## Internet

# which_dataset = 'Internet Usage'
# df = pd.read_csv('kdd_internet_usage.csv')
# target_variable = 'Who_Pays_for_Access_Work' # Making sure the name of the target variable is known
# categorical_variables = ['Actual_Time', 'Community_Building', 'Country', 'Education_Attainment', 'Falsification_of_Information', 'Major_Geographical_Location', 'Major_Occupation', 'Marital_Status','Most_Import_Issue_Facing_the_Internet','Opinions_on_Censorship','Primary_Computing_Platform','Primary_Language','Primary_Place_of_WWW_Access','Race','Registered_to_Vote',
#                           'Sexual_Preference','Web_Ordering','Web_Page_Creation','Age']
# continuous_variables = []
# df['Web_Ordering'] = df['Web_Ordering'].replace(['Yes', 'No'], [1, 0])
# df['Registered_to_Vote'] = df['Registered_to_Vote'].replace(['Yes', 'No'], [1, 0])

# binary_variables = list(set(df.keys()) - set(continuous_variables + categorical_variables + [target_variable]))
# df = pick_only_some(df, target_variable, 1000)
# df = df.reset_index(drop=True)



#########################################################################
######## Car Insurance

# which_dataset = 'Car Insurance'
# df = pd.read_csv('car_insurance.csv')
# df = df.drop('policy_id',axis = 1)
# categorical_variables = ['area_cluster','make', 'segment','model', 'fuel_type','max_torque','max_power','engine_type','airbags','steering_type','ncap_rating'] # Putting in this all the categorical columns
# target_variable = 'is_claim' # Making sure the name of the target variable is known

# binary_cols = ['gear_box','is_esc','is_adjustable_steering','is_tpms',
#                 'is_parking_sensors','is_parking_camera','rear_brakes_type',
#                 'cylinder','transmission_type','is_front_fog_lights'
#                 ,'is_rear_window_wiper','is_rear_window_washer'
#                 ,'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
#                 'is_central_locking','is_power_steering','is_driver_seat_height_adjustable',
#                 'is_day_night_rear_view_mirror','is_ecw','is_speed_alert']


# continuous_variables = ['policy_tenure', 'age_of_car', 'age_of_policyholder',
#         'population_density', 'displacement','turning_radius',
#         'length', 'width', 'height', 'gross_weight']

# df[binary_cols] = df[binary_cols].replace(['Yes', 'No'], [1, 0])
# df['rear_brakes_type'] = df['rear_brakes_type'].replace(['Drum', 'Disc'], [1, 0])
# df['transmission_type'] = df['transmission_type'].replace(['Automatic', 'Manual'], [1, 0])

# df = pick_only_some(df, target_variable, 1000)
# df = df.reset_index(drop=True)


##########################################################################

## Simulated Dataset

# which_dataset = 'Simulated Data'
# df = pd.read_csv('simulate_categories.csv')
# categorical_variables = ['Feature_3'] 
# target_variable = 'target'
# continuous_variables = ['Feature_1','Feature_2']



##########################################################################

## Adult (income >=50k or <50k)

which_dataset = 'Income Data'
df = pd.read_csv('ada_prior.csv')
df.reset_index(inplace=True, drop = True)


df = df.drop(['educationNum','fnlwgt'],axis = 1)

categorical_variables = ['workclass','education',
                         'maritalStatus','occupation','relationship','race','nativeCountry'] 
binary_cols = ['sex']
target_variable = 'label'
continuous_variables = ['age','capitalGain','capitalLoss','hoursPerWeek']
df[binary_cols] = df[binary_cols].replace(['Male', 'Female'], [1, 0])
df[target_variable] = df[target_variable].replace([-1], [0])


##########################################################################


plt.figure()
df[target_variable].value_counts().plot(kind='bar')
plt.title('Test how many target 0 vs 1, dataset: '+ which_dataset)
plt.show()

for key in df.keys():
    print(df[key].value_counts())
    


##########################################################################


how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s

how_many_cv = 5

conf_matrix_list = []

array_confusion_matrix = np.zeros((len(methods),len(classifiers)))

for index in range(how_many_cv):

    randomlist = random.sample(list(df[df[target_variable]==0].index.values), 4 * how_many_0s // 5) + random.sample(list(df[df[target_variable]==1].index.values), 4 * how_many_1s // 5)
    not_in_randomlist = list(set(range(0,size)) - set(randomlist))

    
    df_test = df.iloc[not_in_randomlist,:]
    df_train = df.iloc[randomlist,:]
    
    
    confusion_matrix = whole_process_categorical(df_train, df_test, categorical_variables, continuous_variables, target_variable, which_dataset, 50)
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
variance = np.zeros_like( array_confusion_matrix)
for i1 in range(len(methods)):
    for i2 in range(len(classifiers)):
        mi = 1
        ma = 0
        over_all_cv = []
        for cv in range(how_many_cv):
            value = conf_matrix_list[cv][i1,i2]
            over_all_cv.append(value)
            if ma < value:
                ma = value
            if mi > value:
                mi = value
        mi_array[i1,i2] = mi
        ma_array[i1,i2] = ma
        variance[i1,i2] = np.std(over_all_cv)

from prettytable import PrettyTable
 
# Specify the Column Names while initializing the Table
myTable = PrettyTable([which_dataset]+classifiers+['MEAN SCORE'])

colours = ['tab:blue','tab:orange','tab:green','tab:red', 'tab:pink','tab:brown','tab:purple','tab:cyan', 'tab:olive','tab:gray','blue','gold','orangered','black','purple','magenta']


how_many_methods = len(methods)
how_many_methodsplus2 = how_many_methods + 7
how_many_classifiers = len(classifiers)

plt.figure(figsize=(12,7))
for method_index in range(len(methods)):
    myTable.add_row([methods[method_index]]+ list(np.round(score_matrix[method_index,:],5)))
    plt.plot(np.arange(method_index,how_many_methodsplus2 * how_many_classifiers,how_many_methodsplus2),array_confusion_matrix[method_index,:],'.',color = colours[method_index], label = str(methods[method_index]))
    plt.plot(np.arange(method_index,how_many_methodsplus2 * how_many_classifiers,how_many_methodsplus2),mi_array[method_index,:],'_', color = colours[method_index])
    plt.plot(np.arange(method_index,how_many_methodsplus2 * how_many_classifiers,how_many_methodsplus2),ma_array[method_index,:],'_', color = colours[method_index])
print(myTable)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.arange(4,how_many_methodsplus2 * how_many_classifiers,how_many_methodsplus2),labels = classifiers)
plt.title('Dataset: ' + str(which_dataset))
plt.show() 


# import seaborn as sns
# plt.figure(figsize=(10,7))
# g = sns.heatmap(score_matrix, annot=True, fmt=".5f")
# g.set_xticklabels(classifiers+['MEAN SCORE'], rotation = 45)
# g.set_yticklabels(methods, rotation = 45)
# plt.title('Plot scores, Dataset: ' + str(which_dataset) )
# plt.show()

include_means_conf = np.zeros((len(methods)+1, len(classifiers)+1))
include_means_conf[:-1,:-1] = array_confusion_matrix
include_means_conf[:-1,-1] = np.mean(array_confusion_matrix, axis = 1)
include_means_conf[-1,:-1] = np.mean(array_confusion_matrix, axis = 0)

include_means_conf[-1,-1]  =  np.nan



plt.figure(figsize=(10,7))
g = sns.heatmap(include_means_conf, annot=True, fmt=".5f")
g.set_xticklabels(classifiers+['MEAN'], rotation = 45)
g.set_yticklabels(methods+['MEAN'], rotation = 45)
plt.title('Plot AUC, Dataset: ' + str(which_dataset) )
plt.show()

plt.figure(figsize=(10,7))
g = sns.heatmap(variance, annot=True, fmt=".5f")
g.set_xticklabels(classifiers, rotation = 45)
g.set_yticklabels(methods, rotation = 45)
plt.title('Plot Variance, Dataset: ' + str(which_dataset) )
plt.show()


# ranking = np.zeros((len(methods), len(classifiers)+1))
# for col in range(ranking.shape[1]-1):
#     ranking[:,col] = rankdata(array_confusion_matrix[:,col])
# ranking[:,-1] = np.mean(ranking, axis = 1)

# plt.figure(figsize=(10,7))
# g = sns.heatmap(ranking, annot=True, fmt=".5f")
# g.set_xticklabels(classifiers+['MEAN RANK'], rotation = 45)
# g.set_yticklabels(methods, rotation = 45)
# plt.title('Plot ranks, Dataset: ' + str(which_dataset) )
# plt.show()

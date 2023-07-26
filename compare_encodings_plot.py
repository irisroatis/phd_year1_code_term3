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

#########################################################################
######### CHURN

# which_dataset = 'Churn'
# df = pd.read_csv('churn.csv')
# categorical_variables = ['state','area_code','number_customer_service_calls','phone_number'] # Putting in this all the categorical columns
# target_variable = 'class' # Making sure the name of the target variable is known
# binary_variables = ['international_plan','voice_mail_plan']
# continuous_variables = list(set(df.keys()) - set(categorical_variables + [target_variable]))
# ########################################################################
######### HEART DATASET    

# which_dataset = 'Heart Ilness'
# df = pd.read_csv('heart.csv')
# categorical_variables = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
# target_variable = 'target' # Making sure the name of the target variable is known
# continuous_variables = ['age','trestbps','chol','thalach','oldpeak']
# binary_variables = ['sex','fbs','exang']




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


##### SIMULATED DATASET

which_dataset = 'Simulated Data'
df = pd.read_csv('simulate_categories.csv')
categorical_variables = ['Feature_3'] 
target_variable = 'target'
continuous_variables = ['Feature_1','Feature_2']



how_many_0s = len(df[df[target_variable] == 0])
how_many_1s = len(df[df[target_variable] == 1])
size = how_many_0s + how_many_1s


randomlist = random.sample(list(df[df[target_variable]==0].index.values), 4 * how_many_0s // 5) + random.sample(list(df[df[target_variable]==1].index.values), 4 * how_many_1s // 5)
not_in_randomlist = list(set(range(0,size)) - set(randomlist))

df_test = df.iloc[not_in_randomlist,:]
df_train = df.iloc[randomlist,:]


how_many_rows_train = df.shape[0]
how_many_rows_test = df_test.shape[0]

# We standardise original dataset
for cont_col in continuous_variables:
    df_train[cont_col] = standardise(df_train[cont_col])
    df_test[cont_col] = standardise(df_test[cont_col])

df_wout_cat = df_train.drop(columns=categorical_variables)
df_test_wout_cat = df_test.drop(columns=categorical_variables)

X_train, y_train =  dataset_to_Xandy(df_train, target_variable, only_X = False) ###### the original dataset
X_test, y_test =  dataset_to_Xandy(df_test, target_variable, only_X = False) ###### the original dataset


X_nocat =  dataset_to_Xandy(df_wout_cat, target_variable, only_X = True)
X_test_nocat =  dataset_to_Xandy(df_test_wout_cat, target_variable, only_X = True)

##### the simple encoded dataset 
labelencoder = ce.OrdinalEncoder(cols=categorical_variables)
simple_df = labelencoder.fit_transform(df_train)
simple_df_test =  labelencoder.transform(df_test)
X_simple =  dataset_to_Xandy(simple_df, target_variable, only_X = True)
X_simple_test=  dataset_to_Xandy(simple_df_test, target_variable, only_X = True)

##### the one hot encoded dataset (after binned)
encoder = ce.OneHotEncoder(cols=categorical_variables,use_cat_names=True)
onehot_df = encoder.fit_transform(df_train)
onehot_df_test = encoder.transform(df_test)
X_onehot =  dataset_to_Xandy(onehot_df, target_variable, only_X = True)
X_onehot_test =  dataset_to_Xandy(onehot_df_test, target_variable, only_X = True)

##### the effect encoded dataset (after binned)
encoder = ce.sum_coding.SumEncoder(cols=categorical_variables,verbose=False)
effect_df = encoder.fit_transform(df_train)
effect_df_test = encoder.transform(df_test)
X_effect =  dataset_to_Xandy(effect_df, target_variable, only_X = True)
X_effect_test =  dataset_to_Xandy(effect_df_test, target_variable, only_X = True)

##### the WOE encoded dataset 
encoder = ce.woe.WOEEncoder(cols=categorical_variables,verbose=False)
woe_df = encoder.fit_transform(df_train, df_train[target_variable])
woe_df_test = encoder.transform(df_test)
X_woe =  dataset_to_Xandy(woe_df, target_variable, only_X = True)
X_woe_test =  dataset_to_Xandy(woe_df_test, target_variable, only_X = True)



##### the target encoded dataset 

target_df = df_train.copy()
target_df_test = df_test.copy()

for col in categorical_variables:
    dict_target = {}
    target_df, dict_target =  target_encoding(col, target_variable, target_df)
    target_df_test[col] = target_df_test[col].replace(list(dict_target.keys()), list(dict_target.values()))
    
    
X_target =  dataset_to_Xandy(target_df, target_variable, only_X = True)
X_target_test =  dataset_to_Xandy(target_df_test, target_variable, only_X = True)

##### the GLMM encoded dataset 
encoder = ce.glmm.GLMMEncoder(cols=categorical_variables,verbose=False)
glmm_df = encoder.fit_transform(df_train, df_train[target_variable])
glmm_df_test = encoder.transform(df_test)
X_glmm =  dataset_to_Xandy(glmm_df, target_variable, only_X = True)
X_glmm_test =  dataset_to_Xandy(glmm_df_test, target_variable, only_X = True)

##### the leave-one-out encoded dataset 
encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_variables,verbose=False)
leave_df = encoder.fit_transform(df_train, df_train[target_variable])
leave_df_test = encoder.transform(df_test)
X_leave=  dataset_to_Xandy(leave_df, target_variable, only_X = True)
X_leave_test =  dataset_to_Xandy(leave_df_test, target_variable, only_X = True)

    
##### the CatBoost encoded dataset 
encoder = ce.cat_boost.CatBoostEncoder(cols=categorical_variables,verbose=False)
cat_df = encoder.fit_transform(df_train, df_train[target_variable])
cat_df_test = encoder.transform(df_test)
X_cat=  dataset_to_Xandy(cat_df, target_variable, only_X = True)
X_cat_test =  dataset_to_Xandy(cat_df_test, target_variable, only_X = True)


#### the 10-Fold Target Encoding
modified_df10, modified_df_test10 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='target')
X_target10 =  dataset_to_Xandy(modified_df10, target_variable, only_X = True)
X_target_test10 =  dataset_to_Xandy(modified_df_test10, target_variable, only_X = True)

#### the 5-Fold Target Encoding
modified_df5, modified_df_test5 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='target')
X_target5 =  dataset_to_Xandy(modified_df5, target_variable, only_X = True)
X_target_test5 =  dataset_to_Xandy(modified_df_test5, target_variable, only_X = True)


#### the 10-Fold GLMM Encoding
glmm_modified_df10,  glmm_modified_df_test10 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=10, which_encoder='glmm')
X_glmm10 =  dataset_to_Xandy(glmm_modified_df10, target_variable, only_X = True)
X_glmm_test10 =  dataset_to_Xandy(glmm_modified_df_test10, target_variable, only_X = True)

#### the 5-Fold GLMM Encoding
glmm_modified_df5,  glmm_modified_df_test5 = k_fold_target_encoding(df_train, df_test, categorical_variables, target_variable, how_many_folds=5, which_encoder='glmm')
X_glmm5 =  dataset_to_Xandy(glmm_modified_df5, target_variable, only_X = True)
X_glmm_test5 =  dataset_to_Xandy(glmm_modified_df_test5, target_variable, only_X = True)


# which_category = 'area_cluster'
# which_category = 'cp'
which_category = 'Feature_3'


fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Different number of folds, GLMM (no folds, 5, 10) and WOE')
axs[0].hist(X_glmm[which_category],bins = 50)
axs[1].hist(X_glmm5[which_category],bins = 50)
axs[2].hist(X_glmm10[which_category],bins = 50)
plt.show()

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Different number of folds, target')
axs[0].hist(X_target[which_category],bins = 50)
axs[1].hist(X_target5[which_category],bins = 50)
axs[2].hist(X_target10[which_category],bins = 50)
plt.show()

fig, axs = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('Leave-One-Out and Catboost encoders')
axs[0].hist(X_leave[which_category],bins = 50)
axs[1].hist(X_cat[which_category],bins = 50)
plt.show()


plt.hist(X_simple[which_category],bins = 50)
plt.title('Simple encoding')
plt.show()

plt.hist(X_woe[which_category],bins = 50)
plt.title('WOE encoding')
plt.show()

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




X_multiple_encoding = X_target.copy()
X_multiple_encoding = X_multiple_encoding.rename({which_category: which_category+'_target'}, axis='columns')
X_multiple_encoding[which_category+'_target_5'] = X_target5[which_category]
X_multiple_encoding[which_category+'_target_10'] = X_target10[which_category]
X_multiple_encoding[which_category+'_glmm'] = X_glmm[which_category]
X_multiple_encoding[which_category+'_glmm5'] = X_glmm5[which_category]
X_multiple_encoding[which_category+'_glmm10'] = X_glmm10[which_category]


X_multiple_encoding_test = X_target_test.copy()
X_multiple_encoding_test = X_multiple_encoding_test.rename({which_category: which_category+'_target'}, axis='columns')
X_multiple_encoding_test[which_category+'_target_5'] = X_target_test5[which_category]
X_multiple_encoding_test[which_category+'_target_10'] = X_target_test10[which_category]
X_multiple_encoding_test[which_category+'_glmm'] = X_glmm_test[which_category]
X_multiple_encoding_test[which_category+'_glmm5'] = X_glmm_test5[which_category]
X_multiple_encoding_test[which_category+'_glmm10'] = X_glmm_test10[which_category]

model_ensemble = DecisionTreeClassifier(max_depth=5)
model_ensemble.fit(X_multiple_encoding, y_train)
y_predicted = model_ensemble.predict(X_multiple_encoding_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
area_roc_ensemble = metrics.auc(fpr, tpr)


model_target = DecisionTreeClassifier(max_depth=5)
model_target.fit(X_target, y_train)
y_predicted = model_target.predict(X_target_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
area_roc_target = metrics.auc(fpr, tpr)



fig = plt.figure(figsize=(50,40))
_ = tree.plot_tree(model_ensemble,
                   feature_names=X_multiple_encoding.keys(), 
                   class_names='target',
                   filled=True)


fig = plt.figure(figsize=(50,40))
_ = tree.plot_tree(model_target,
                   feature_names=X_multiple_encoding.keys(), 
                   class_names='target',
                   filled=True)






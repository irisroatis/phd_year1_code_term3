#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:44:29 2023

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
from sklearn import tree

list_cat = ['A','C','B','A','B','A']
list_target = np.array([1,0,1,1,0,0])
df = pd.DataFrame(list_target,columns = ['target'])
df['Feature'] = list_cat
categorical_variables=['Feature']
target_variable = 'target'

unique_cat = list(set(list_cat))
cat_and_encoded = pd.DataFrame(unique_cat, columns = ['Feature'])
cat_and_encoded['target'] = np.nan

encoder = ce.target_encoder.TargetEncoder(cols = categorical_variables, verbose=False)
target_w_df = encoder.fit_transform(df, df[target_variable])

cat_and_encoded['target_encoded'] = encoder.transform(cat_and_encoded)['Feature']

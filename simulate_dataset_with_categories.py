#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:23:22 2023

@author: ir318
"""

import pandas as pd
from functions import *
import numpy as np
import random

mu1 = np.array([0, 0, 0])
mu2 = np.array([1, 1, 1])
sigma1 = np.identity(3)
sigma2 = np.identity(3)
how_many_times_repeat = 1
iterations = 2000

testing_data, belonging_classes = generating_test_data(how_many_times_repeat, iterations, mu1, sigma1, mu2, sigma2)

names_columns = ['Feature_'+str(i+1) for i in range(len(mu1))]
target_variable = 'target'
which_column_to_categories = 'Feature_3'

df = pd.DataFrame(testing_data[0], columns = names_columns)
df[target_variable] = belonging_classes[0]

how_many_bins = 10

bins_col3 = create_bins(df[which_column_to_categories], df[which_column_to_categories], how_many_bins+1, 'fixed_number')

categories = list(np.arange(1, how_many_bins+1))
categories_shuffled = random.sample(categories, how_many_bins)

compare = df[which_column_to_categories]

digitized = np.digitize(df[which_column_to_categories],bins_col3)
df[which_column_to_categories] = digitized
df[which_column_to_categories] = df[which_column_to_categories].replace(categories, categories_shuffled)

df_train, df_test = split_train_test(df, target_variable)

df_train.to_csv('simulate_categories_train.csv', index = None)
df_test.to_csv('simulate_categories_test.csv', index = None)
bins_col3.tofile('bins.txt',sep=',')
np.array(categories_shuffled).tofile('order_cat.txt',sep=',')


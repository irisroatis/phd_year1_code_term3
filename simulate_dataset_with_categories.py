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

what_to_generate = 'two_normals'
how_many_times_repeat = 1
iterations = 5000

if what_to_generate == 'unif_beta':
    low = 0
    high = 1
    first = 4
    second = 1
    testing_data, belonging_classes = generating_test_data_uniform(how_many_times_repeat, iterations, low, high, first, second)
    names_columns = ['Feature_1']
elif what_to_generate == 'two_normals':
    mu1 = np.array([0])
    mu2 = np.array([0.5])
    sigma1 = np.identity(len(mu1))
    sigma2 =  np.identity(len(mu1))
    testing_data, belonging_classes = generating_test_data(how_many_times_repeat, iterations, mu1, sigma1, mu2, sigma2)
    names_columns = ['Feature_'+str(i+1) for i in range(len(mu1))]
    



target_variable = 'target'
which_column_to_categories = 'Feature_1' ######## this needs to be changed

df = pd.DataFrame(testing_data[0], columns = names_columns)
df[target_variable] = belonging_classes[0]

initial_dataset = df.copy()

how_many_bins = 20

bins_col3 = create_bins(df[which_column_to_categories], df[which_column_to_categories], how_many_bins+1, 'fixed_number')
categories = list(np.arange(1, how_many_bins+1))
# categories_shuffled = random.sample(categories, how_many_bins)



plt.figure(figsize = (17,7))

if what_to_generate == 'unif_beta':
    plt.hist(df[df[target_variable] == 0][which_column_to_categories], label='U('+str(low)+','+str(high)+')',alpha = 0.5, density=True, bins = 50)
    plt.hist(df[df[target_variable] == 1][which_column_to_categories], label='B('+str(first)+','+str(second)+')',alpha = 0.5, density=True, bins = 50)
elif what_to_generate == 'two_normals':
    plt.hist(df[df[target_variable] == 0][which_column_to_categories], label='N('+str(mu1[0])+',1)',alpha = 0.5, density=True, bins = 50)
    plt.hist(df[df[target_variable] == 1][which_column_to_categories], label='N('+str(mu2[0])+',1)',alpha = 0.5, density=True, bins = 50)
plt.legend()
for index in range(len(bins_col3)):
    plt.axvline(bins_col3[index], ls='--', color = 'black')
    if index < len(bins_col3)-1:
        plt.text((bins_col3[index] + bins_col3[index+1])/2, 0.2, categories[index], size = 'large')
plt.xlabel('$x$')
plt.ylabel('$f_X(x)$')
plt.savefig(str(what_to_generate)+'.png')
plt.show()




compare = df[which_column_to_categories]

digitized = np.digitize(df[which_column_to_categories],bins_col3)
df[which_column_to_categories] = digitized
df[which_column_to_categories] = df[which_column_to_categories].replace(categories, categories)

df_train, df_test, ind_test = split_train_test(df, target_variable, indices_test=True)


df_train.to_csv('simulate_categories_train_'+str(what_to_generate)+'.csv', index = None)
df_test.to_csv('simulate_categories_test_'+str(what_to_generate)+'.csv', index = None)
bins_col3.tofile('bins_'+str(what_to_generate)+'.txt',sep=',')
np.array(categories).tofile('order_cat_'+str(what_to_generate)+'.txt',sep=',')

initial_dataset.iloc[ind_test].to_csv('initial_testdata_'+str(what_to_generate)+'.csv', index = None)




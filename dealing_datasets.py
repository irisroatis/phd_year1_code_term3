#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:39:08 2023

@author: roatisiris
"""

import pandas as pd
import random
import numpy as np
from functions import *

def dataset_variables(which_dataset):
    ######### HEART DATASET   
    if which_dataset == 'Heart Ilness':
        df = pd.read_csv('heart.csv')
        categorical_variables = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
        target_variable = 'target' # Making sure the name of the target variable is known
        continuous_variables = ['age','trestbps','chol','thalach','oldpeak']
        binary_variables = ['sex','fbs','exang']
        
    ######### CHURN
    elif which_dataset == 'Churn':
        which_dataset = 'Churn'
        df = pd.read_csv('churn.csv')
        categorical_variables = ['state','area_code','number_customer_service_calls'] # Putting in this all the categorical columns
        target_variable = 'class' # Making sure the name of the target variable is known
        binary_variables = ['international_plan','voice_mail_plan']
        continuous_variables = list(set(df.keys()) - set(categorical_variables + [target_variable]))
    
    ######### AMAZON
    elif which_dataset == 'Amazon_employee_access':
        df = pd.read_csv('amazon.csv')
        target_variable = 'target' # Making sure the name of the target variable is known
        categorical_variables = list(set(df.keys()) - set([target_variable]))
        df = pick_only_some(df, target_variable, 1000)
        df = df.reset_index(drop=True)
        
    ######### MUSHROOM
    elif which_dataset == 'Mushroom':
        colnames = ['target','cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing','gill_size','gill_colour','stalk_shape','stalk_root','stalk_sur_ab_ring','stalk_sur_bw_ring','stalk_col_ab_ring','stalk_col_bw_ring','veil_type','veil_colour','ring_number','ring_type','spore_print_colour','population','habitat']
        df = pd.read_csv('mushrooms.data', names = colnames)
        target_variable = 'target' # Making sure the name of the target variable is known
        categorical_variables = colnames
        continuous_variables = []
        df[target_variable] = df[target_variable].replace(['e', 'p'], [1, 0])

    ######### CLICK PREDICTION ADDS
    elif which_dataset == 'Click Prediction':
        df = pd.read_csv('click_prediction.csv')
        target_variable = 'click' # Making sure the name of the target variable is known
        categorical_variables = ['url_hash', 'ad_id', 'advertiser_id', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id']
        continuous_variables = list(set(df.keys()) - set(categorical_variables + [target_variable]))
        df = pick_only_some(df, target_variable, 1000)
        df = df.reset_index(drop=True)
        
    ######## Internet
    elif which_dataset == 'Internet Usage':
        df = pd.read_csv('kdd_internet_usage.csv')
        target_variable = 'Who_Pays_for_Access_Work' # Making sure the name of the target variable is known
        categorical_variables = ['Actual_Time', 'Community_Building', 'Country', 'Education_Attainment', 'Falsification_of_Information', 'Major_Geographical_Location', 'Major_Occupation', 'Marital_Status','Most_Import_Issue_Facing_the_Internet','Opinions_on_Censorship','Primary_Computing_Platform','Primary_Language','Primary_Place_of_WWW_Access','Race','Registered_to_Vote',
                                  'Sexual_Preference','Web_Ordering','Web_Page_Creation','Age']
        continuous_variables = []
        df['Web_Ordering'] = df['Web_Ordering'].replace(['Yes', 'No'], [1, 0])
        df['Registered_to_Vote'] = df['Registered_to_Vote'].replace(['Yes', 'No'], [1, 0])

        binary_variables = list(set(df.keys()) - set(continuous_variables + categorical_variables + [target_variable]))
        df = pick_only_some(df, target_variable, 1000)
        df = df.reset_index(drop=True)
        
    ######## Car Insurance
    elif which_dataset == 'Car Insurance':    
        df = pd.read_csv('car_insurance.csv')
        df = df.drop('policy_id',axis = 1)
        categorical_variables = ['area_cluster','make', 'segment','model', 'fuel_type','max_torque','max_power','engine_type','airbags','steering_type','ncap_rating'] # Putting in this all the categorical columns
        target_variable = 'is_claim' # Making sure the name of the target variable is known
        
        binary_cols = ['gear_box','is_esc','is_adjustable_steering','is_tpms',
                        'is_parking_sensors','is_parking_camera','rear_brakes_type',
                        'cylinder','transmission_type','is_front_fog_lights'
                        ,'is_rear_window_wiper','is_rear_window_washer'
                        ,'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
                        'is_central_locking','is_power_steering','is_driver_seat_height_adjustable',
                        'is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
        
        
        continuous_variables = ['policy_tenure', 'age_of_car', 'age_of_policyholder',
                'population_density', 'displacement','turning_radius',
                'length', 'width', 'height', 'gross_weight']
        
        df[binary_cols] = df[binary_cols].replace(['Yes', 'No'], [1, 0])
        df['rear_brakes_type'] = df['rear_brakes_type'].replace(['Drum', 'Disc'], [1, 0])
        df['transmission_type'] = df['transmission_type'].replace(['Automatic', 'Manual'], [1, 0])
        
        df = pick_only_some(df, target_variable, 1000)
        df = df.reset_index(drop=True)
        
    ######## Simulated Data
    elif which_dataset == 'Simulated Data':  
        df = pd.read_csv('simulate_categories.csv')
        categorical_variables = ['Feature_3'] 
        target_variable = 'target'
        continuous_variables = ['Feature_1','Feature_2']
        
        
    ########  Adult (income >=50k or <50k)
    elif which_dataset == 'Income Prediction':  
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
        
    ####### Australian credit approval 
    elif which_dataset == 'Australian Credit Approval':
        df = pd.read_csv('australian.csv')
        df.columns = df.columns.str.replace("'","")

        categorical_variables = ['A4','A5','A6','A12'] 
        binary_cols = ['A1','A8', 'A9', 'A11']
        target_variable = 'A15'
        continuous_variables = ['A2','A3','A7','A10','A13', 'A14']
        
        
    ####### Good/bad Credit risks 
    elif which_dataset == 'Good/bad Credit Risks':

        df = pd.read_csv('credit-g.csv')
        df.columns = df.columns.str.replace("'","")
        
        categorical_variables = ['checking_status','credit_history',
                                  'savings_status','employment','installment_commitment','personal_status','other_parties','residence_since',
                                  'property_magnitude','purpose','other_payment_plans','housing','job'] 
        binary_cols = ['own_telephone','foreign_worker']
        target_variable = 'class'
        continuous_variables = ['duration','credit_amount','age','existing_credits','num_dependents']
        df[binary_cols] = df[binary_cols].replace(['yes','none','no'], [1,0,0])
        df[target_variable] = df[target_variable].replace(['good','bad'], [1,0])
        
    elif which_dataset == 'Cylinder Bands':

        df = pd.read_csv('cylinder-bands.csv')
        df.columns = df.columns.str.replace("'","")
        
        categorical_variables = ['checking_status','credit_history',
                                  'savings_status','employment','installment_commitment','personal_status','other_parties','residence_since',
                                  'property_magnitude','purpose','other_payment_plans','housing','job'] 
        binary_cols = ['own_telephone','foreign_worker']
        target_variable = 'class'
        continuous_variables = ['duration','credit_amount','age','existing_credits','num_dependents']
        df[binary_cols] = df[binary_cols].replace(['yes','none','no'], [1,0,0])
        df[target_variable] = df[target_variable].replace(['good','bad'], [1,0])

    return df, categorical_variables, continuous_variables, binary_cols, target_variable


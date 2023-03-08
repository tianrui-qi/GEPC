#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:48:08 2023

@author: hklumpe
"""

import glob
import sys
import json
import numpy as np

import matplotlib.pyplot as plt

username='hklumpe'
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

# Import data
def get_fluo_and_pred(simul_id):
    """
    Given a simul_id (str), return the fluo (arr, (n_cells, len(future)))
    and fluo_pred (arr, (n_cells, len(future))) for each individual evaluation
    cell
    """

    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'
    
    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    # Load data
    eval_sets = training_params['eval_sets'][0]
    fluo = np.load(f"{training_params['datasets_folder']}/{eval_sets}/futures_fluo.npy")
    # Keep only first realization
    fluo = fluo[:,0,:]
    
    # Load predictions
    fluo_pred = 4095 * np.load(f'{simul_dir}/evaluation/predictions.npy')
    
    return fluo, fluo_pred

def get_params(simul_id):
    """
    Given a simul_id (str), return the fluo (arr, (n_cells, len(future)))
    and fluo_pred (arr, (n_cells, len(future))) for each individual evaluation
    cell
    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'

    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    with open(training_params['datasets_folder']+'/model_parameters.json', 'r') as f:
        model_params = json.load(f)
        
    return training_params, model_params

simul_id = '2023-03-08_00-17-59'
fluo, fluo_pred = get_fluo_and_pred(simul_id)
training_params, model_params = get_params(simul_id)

RMSE = np.sqrt((fluo - fluo_pred)**2)
plt.plot(np.arange(np.shape(fluo)[1]), RMSE.T, '.')



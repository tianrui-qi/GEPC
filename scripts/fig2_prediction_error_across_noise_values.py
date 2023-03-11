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
import pandas as pd

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
        
    # Load predictions
    fluo_pred = 4095 * np.load(f'{simul_dir}/evaluation/predictions.npy')

    # Load data
    eval_sets = training_params['eval_sets'][0]
    fluo = np.load(f"{training_params['datasets_folder']}/{eval_sets}/futures_fluo.npy")
    
    # Keep only first realization, and as many time point as in the prediction
    fluo = fluo[:,0,:np.shape(fluo_pred)[1]]

    
    return fluo, fluo_pred

def get_params(simul_id):
    """
    Given: 
        simul_id: str (some unique part of file name in /assets/models/) 
    Returns:
        training_params: dict
        model parameters: dict
    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'

    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    with open(training_params['datasets_folder']+'/model_parameters.json', 'r') as f:
        model_params = json.load(f)
        
    return training_params, model_params

#%% Get information about simulations

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023-03-10*')
                    
simul_id_list = [simul_dir.split('/')[-1] for simul_dir in simul_dir_list]
horizon_list = []
training_folder_list = []
past_steps_list = []
cell_class_list = []
epochs_list = []
datasets_list = []
h1_list = []
h2_list = []

for simul_id in simul_id_list:
    
    training_params, model_params = get_params(simul_id)
    
    horizon_list += [training_params['horizon']]
    past_steps_list += [training_params['past_steps']]
    training_folder_list += [training_params['training_sets'][0]]
    cell_class_list += [training_params['cell_class']]
    epochs_list += [training_params['training_parameters']['epochs']]
    datasets_list += [training_params['datasets_folder']]
    h1_list += [model_params['h1']]
    h2_list += [model_params['h2']]
    
df_meta = pd.DataFrame({'simul_id': simul_id_list,
                        'horizon': horizon_list,
                        'past_steps': past_steps_list,
                        'training_sets': training_folder_list,
                        'epochs': epochs_list,
                        'datasets_folder': datasets_list,
                        'h1': h1_list,
                        'h2': h2_list,
                        'cell_class': cell_class_list,})

# Drop repeats
df_meta = df_meta.drop_duplicates()

# Only keep fully trained models
df_meta.drop(df_meta[df_meta['epochs']<200].index, inplace=True)

# Common bools
default_horizon = df_meta['horizon']==24
default_past_steps = df_meta['past_steps']==36
default_training_size = df_meta['training_sets']=='training_set'

_h1 = 0.071/25
_h2 = 0.0303/50
_h1h2 = _h1 / _h2
default_h1 = np.abs((df_meta['h1'] - _h1) / _h1) < 0.01
default_h2 = np.abs((df_meta['h2'] - _h2) / _h2) < 0.01
default_h1h2 = np.abs(((df_meta['h1'] / df_meta['h2']) - _h1h2) / _h1h2) < 0.01

#%% Plot error as function of speed (same steady state) 
model_style_dict = {'CcaSR_gillespie_simple': '.',
                    'CcaSR_gillespie': 's'}

simul_slice = default_training_size & default_past_steps & default_horizon & default_h1h2

df_past = df_meta.loc[simul_slice].sort_values(by=['cell_class','h1','h2']).reset_index(drop=True)

plt.figure(figsize=(8,3))
for i in range(len(df_past)):
    cell_class = df_past.loc[i,'cell_class']
    h1 = df_past.loc[i,'h1']
    h2 = df_past.loc[i,'h2']
    simul_id = df_past.loc[i, 'simul_id']
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
    t = [x/12. for x in range(len(RMSE))]
    plt.plot(t, RMSE, 
             model_style_dict[cell_class], 
             label=f'{cell_class}, h1={h1:.3f}, h2={h2:.3f}')

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('time (h)')
plt.ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+'/assets/figures/effect_of_rate.png', dpi=600)


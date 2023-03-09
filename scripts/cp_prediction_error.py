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

#%% Plot error as function of horizon 
color_dict = {'CcaSR_gillespie_simple_noE': 'r',
              'CcaSR_gillespie_simple': 'g',
              'CcaSR_gillespie': 'k'}

style_dict = {12: '.', 24: 'x', 48: '_'}

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023-03-07_23*') + \
                    glob.glob(dcc_repo_path + '/assets/models/2023-03-08_00*')

simul_id_list = [simul_dir.split('/')[-1] for simul_dir in simul_dir_list]

plt.figure(figsize=(8,3))
for simul_id in simul_id_list:
    fluo, fluo_pred = get_fluo_and_pred(simul_id)
    training_params, model_params = get_params(simul_id)
    cell_class = training_params['cell_class']
    horizon = training_params['horizon']

    RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
    t = [x/12. for x in range(len(RMSE))]
    plt.plot(t, RMSE, style_dict[horizon], 
             label=f'{cell_class}, horizon={horizon}',
             color=color_dict[cell_class])

plt.legend(bbox_to_anchor=(1,1))
# plt.yscale('log')
plt.xlabel('time (h)')
plt.ylabel(f'Median RMSE\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+'/assets/figures/effect_of_horizon.png', dpi=600)

#%% Get information about simulations

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023-03-07_23*') + \
                    glob.glob(dcc_repo_path + '/assets/models/2023-03-08*') + \
                    glob.glob(dcc_repo_path + '/assets/models/2023-03-09_0*')
                    
simul_id_list = [simul_dir.split('/')[-1] for simul_dir in simul_dir_list]
horizon_list = []
training_folder_list = []
past_steps_list = []
cell_class_list = []
epochs_list = []

for simul_id in simul_id_list:
    
    training_params, model_params = get_params(simul_id)
    
    horizon_list += [training_params['horizon']]
    past_steps_list += [training_params['past_steps']]
    training_folder_list += [training_params['training_sets'][0]]
    cell_class_list += [training_params['cell_class']]
    epochs_list += [training_params['training_parameters']['epochs']]
    
df_meta = pd.DataFrame({'simul_id': simul_id_list,
                        'horizon': horizon_list,
                        'past_steps': past_steps_list,
                        'training_sets': training_folder_list,
                        'epochs': epochs_list,
                        'cell_class': cell_class_list,})

df_meta = df_meta.sort_values(by=['past_steps','training_sets','cell_class']).reset_index(drop=True)
df_meta.drop(df_meta[df_meta['epochs']<200].index, inplace=True)

#%% Plot past steps results

simul_slice = (df_meta['training_sets']=='training_set')&(df_meta['horizon']==24)

df_past = df_meta.loc[simul_slice].sort_values(by=['cell_class','past_steps']).reset_index(drop=True)

past_steps_style_dict = {12: '_', 24: 'x', 36: '.'}

plt.figure(figsize=(8,3))
for i in range(len(df_past)):
    cell_class = df_past.loc[i,'cell_class']
    past_steps = df_past.loc[i,'past_steps']
    simul_id = df_past.loc[i, 'simul_id']
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
    t = [x/12. for x in range(len(RMSE))]
    plt.plot(t, RMSE, past_steps_style_dict[past_steps], 
             label=f'{cell_class}, past_steps={past_steps}',
             color=color_dict[cell_class])

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('time (h)')
plt.ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+'/assets/figures/effect_of_past_steps.png', dpi=600)

#%% Plot error as function of training set size 

simul_slice = (df_meta['past_steps']==36)&(df_meta['horizon']==24)
df_past = df_meta.loc[simul_slice].sort_values(by=['cell_class','training_sets']).reset_index(drop=True)

training_style_dict = {100: '_', 300: 'x', 1000: '.', 10000: 's'}

plt.figure(figsize=(12,6))
for i in range(len(df_past)):
    cell_class = df_past.loc[i, 'cell_class']
    training_folder = df_past.loc[i,'training_sets']
    if training_folder == 'training_set':
        training_set_size = 10000
    else:
        training_set_size = int(training_folder.split('_')[-2])
    simul_id = df_past.loc[i, 'simul_id']
    
    if training_set_size==300:
        continue
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
    t = [x/12. for x in range(len(RMSE))]
    plt.plot(t, RMSE, 
             training_style_dict[training_set_size], 
              label=f'{cell_class}, trained on {training_set_size}cells',
              color=color_dict[cell_class])

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('time (h)')
plt.ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+'/assets/figures/effect_of_training_set_size.png',dpi=600)

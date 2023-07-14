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
fig_path = dcc_repo_path+'/assets/figures/'

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

#%% Function definitions

def get_eval_data(simul_id):
    """
    Function to get evaluation data
    
    Parameters
    ----------
    simul_id : str
        fname of model
        
    Returns
    -------
    stims : arr (n_cells, n_timepoints)
    
    past_fluo: arr (n_cells, n_past_timepoints)
    
    futures_fluo: arr (n_cells, n_futures, n_future_timepoints)

    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'
    
    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    # Load evaluation data
    eval_sets = training_params['eval_sets'][0]
    futures_fluo = np.load(f"{training_params['datasets_folder']}/{eval_sets}/futures_fluo.npy")
    past_fluo = np.load(f"{training_params['datasets_folder']}/{eval_sets}/past_fluo.npy")
    stims = np.load(f"{training_params['datasets_folder']}/{eval_sets}/stims.npy")
    
    return stims, past_fluo, futures_fluo

def get_fluo_pred(simul_id, validated=True):
    """
    Funtion to get predicted futures

    Parameters
    ----------
    simul_id : str
        fname of model
        
    Returns
    -------
    fluo_pred : arr

    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'
        
    # Load predictions
    if validated:
        fluo_pred = 4095 * np.load(f'{simul_dir}/evaluation/predictions.npy')
    else:
        fluo_pred = 4095 * np.load(f'{simul_dir}/evaluation_no_validation/predictions.npy')
    
    return fluo_pred
    
def get_fluo_and_pred(simul_id, return_all=True, validated=True):
    """
    Inputs
    ------
    simul_id: str (unique identifier of folder in `assets/models`)
    return_all: bool, optional
        Whether to return all possible futures
        Default is false
    validated: bool, optional
        Whether to return the prediction data on a model that was validated or not
        Default is True
        
    Outputs
    -------
    fluo: arr, (n_cells, len(future))
        Evaluation data, future fluorescence
        if return_all==True, then fluo has shape: (n_cells, n_futures, len(future))
    fluo_pred: arr, (n_cells, len(future))
        Predicted future fluorescence
    
    """
        
    # Load predictions
    fluo_pred = get_fluo_pred(simul_id, validated=validated)

    # Load data
    stims, past_fluo, futures_fluo = get_eval_data(simul_id)
    
    # Keep only first realization, and as many time point as in the prediction
    if return_all:
        fluo = futures_fluo[:,:,:np.shape(fluo_pred)[1]]
        fluo_pred = np.repeat(fluo_pred[:, np.newaxis], fluo.shape[0], axis=1)
    else:
        fluo = futures_fluo[:,0,:np.shape(fluo_pred)[1]]
    return fluo, fluo_pred

def get_params(simul_id):
    """
    Given: 
        simul_id: str (some unique part of file name in /assets/models/) 
    Returns:
        model parameters: dict
        training_params: dict
    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'

    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    with open(training_params['datasets_folder']+'/model_parameters.json', 'r') as f:
        model_params = json.load(f)
        
    return model_params, training_params


#%% Global plot styles

df_simul_config = pd.DataFrame({'cell_class': ['CcaSR_gillespie',
                                               'CcaSR_gillespie_simple_noE',
                                               'CcaSR_gillespie_simple_noE',
                                               'CcaSR_gillespie_simple_noE'],
                                'camera_sim': [True, True, True, False],
                                'solver': ['original','original','ode', 'ode'],
                                'color': ["#327a42",
                                            "#5a6eb8",
                                            "#57377b",
                                            "#c561b0",]}
                            )

KI_vals = [20, 40, 50, 70]
KI_colors = ["#d84d32",
             "#b27d3e",
            "#dac54b",
            "#8f8539"]
KI_color_dict = {KI_vals[i]: KI_colors[i] for i in range(len(KI_vals))}


#%% Plot responses to pure light, to show variability

n_hours = 2
n_cells = 80
alpha=0.1
lw = 1.5

#  # Constant E
cell_class = 'CcaSR_Autoactivation'
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
light_sequence = [0]*n_hours*12 + [0,0,1,1]*n_hours*12 + [1]*n_hours*6*12 + \
                    [0,0,1,1]*n_hours*12 + [0]*n_hours*8*12
x = [t/12. for t in range(len(light_sequence)+1)]


dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    
for i in range(n_cells):

    # Simulate cell
    cell = dcc.simulations.CcaSR_Autoactivation()
    
    # Simulate
    cell.set_light_events(light_sequence)     
    series = cell.run(len(light_sequence)*5, 
                           solver="original", 
                           realizations=1)
    
    plt.plot(x, [state['F'] for state in series],
                 color='b',
                 alpha=alpha, lw=lw)
    
plt.xlim([0,np.max(x)])
plt.ylim([0,150])
plt.tight_layout()
plt.savefig(f'{fig_path}/figN_{cell_class}_sample_responses.png', dpi=300)

#%% Get information about simulations

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023*')
                    
simul_id_list = [simul_dir.split('/')[-1] for simul_dir in simul_dir_list]
horizon_list = []
training_folder_list = []
solver_list = []
camera_sim_list = []
past_steps_list = []
cell_class_list = []
epochs_list = []
datasets_list = []
h1_list = []
h2_list = []
KI_list = []
KJ_list = []

for simul_id in simul_id_list:
    
    model_params, training_params = get_params(simul_id)
    
    horizon_list += [training_params['horizon']]
    past_steps_list += [training_params['past_steps']]
    training_folder_list += [training_params['training_sets'][0]]
    cell_class_list += [training_params['cell_class']]
    epochs_list += [training_params['training_parameters']['epochs']]
    datasets_list += [training_params['datasets_folder']]
    
    if 'K_I' in model_params.keys():
        KI_list += [model_params['K_I']]
    else:
        KI_list += [np.nan]
    if 'K_J' in model_params.keys():
        KJ_list += [model_params['K_J']]
    else: 
        KJ_list += [np.nan]
    if 'h1' in model_params.keys():
        h1_list += [model_params['h1']]
    else:
        h1_list += [np.nan]
    if 'h2' in model_params.keys():
        h2_list += [model_params['h2']]
    else:
        h2_list += [np.nan]
    if 'solver' in model_params.keys():
        solver_list += [model_params['solver']]
    else:
        solver_list += ['original']
    if 'camera_sim' in model_params.keys():
        camera_sim_list += [model_params['camera_sim']]
    else:
        camera_sim_list += [True]
    
df_meta = pd.DataFrame({'simul_id': simul_id_list,
                        'horizon': horizon_list,
                        'past_steps': past_steps_list,
                        'camera_sim': camera_sim_list,
                        'solver': solver_list,
                        'h1': h1_list,
                        'h2': h2_list,
                        'K_I': KI_list,
                        'K_J': KJ_list,
                        'cell_class': cell_class_list,
                        'training_sets': training_folder_list,
                        'epochs': epochs_list,
                        'datasets_folder': datasets_list,})

# Drop repeats
df_meta = df_meta.sort_values(by=['simul_id','past_steps','horizon','K_I','K_J'])
df_meta = df_meta.drop_duplicates(subset=[col for col in df_meta.columns[1:]], keep='last')

# Only keep fully trained models
df_meta.drop(df_meta[df_meta['epochs']<200].index, inplace=True)

# Common bools
default_horizon = df_meta['horizon']==24
default_past_steps = df_meta['past_steps']==36
default_training_size = df_meta['training_sets']=='training_set'

_h1 = 4e-2
_h2 = 1e-3
_h1h2 = _h1 / _h2
default_h1 = np.abs((df_meta['h1'] - _h1) / _h1) < 0.01
default_h2 = np.abs((df_meta['h2'] - _h2) / _h2) < 0.01
default_h1h2 = np.abs(((df_meta['h1'] / df_meta['h2']) - _h1h2) / _h1h2) < 0.01

default_models = default_training_size & default_past_steps & default_horizon & default_h1 & default_h2

#%% Gillespie noise correlation (tf)

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

alpha=0.3
for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]
    camera_sim = config['camera_sim']
    solver = config['solver']
    cell_class = config['cell_class']
    color = config['color']
    config_bool = (df_meta['camera_sim']==camera_sim)&(df_meta['solver']==solver)&(df_meta['cell_class']==cell_class)
    
    simul_id = df_meta.loc[default_models & config_bool, 'simul_id'].values[0]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    x = np.arange(np.shape(fluo)[-1])/12
    
    t0_spread = np.max(fluo[:,:,0], axis=1) - np.min(fluo[:,:,0], axis=1)
    tf_spread = np.max(fluo[:,:,-1], axis=1) - np.min(fluo[:,:,-1], axis=1)
        
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))

    x_list = [RMSE[:,0], RMSE[:,-1], t0_spread, tf_spread]
    x_label = ['t0 RMSE', 'tf RMSE', 't0 spread (ground truth)', 'tf spread (ground truth)']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'camera_sim{camera_sim}, solver{solver}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)

plt.tight_layout()
plt.savefig(fig_path+'/figN_fig1_models_error_correlations_t0_tf.png',dpi=300)

#%% Gillespie noise error correlations(stims)

simul_slice = default_training_size & default_horizon & default_past_steps

df_plot = df_meta.loc[simul_slice].sort_values(by=['cell_class'])

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

past_steps = 3*12
horizon = 2*12
alpha=0.3
for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]
    camera_sim = config['camera_sim']
    solver = config['solver']
    cell_class = config['cell_class']
    color = config['color']
    config_bool = (df_meta['camera_sim']==camera_sim)&(df_meta['solver']==solver)&(df_meta['cell_class']==cell_class)
    
    simul_id = df_meta.loc[default_models & config_bool, 'simul_id'].values[0]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    stims, past_fluo, futures_fluo = get_eval_data(simul_id)
    t_on_past = np.sum(stims[:,-(past_steps+horizon):-horizon], axis=1)
    t_on_future = np.sum(~stims[:,-horizon:], axis=1)
    t_switches = np.sum(np.diff(stims[:, -horizon:], axis=1), axis=1)
    
    x_list = [t_on_past, t_on_future, t_switches, np.mean(futures_fluo[:,:,-1],axis=1)]
    x_label = ['t on past','t on future', 'on/off switches', 'avg fluorescence']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'camera_sim{camera_sim}, solver{solver}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)

plt.tight_layout()
plt.savefig(fig_path+'/figN_fig1_models_error_correlations_stim.png',dpi=300)

#%% Cascade error correlations (tf)

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

alpha=0.3
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    df_index = (df_past['K_I']==K_I)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    color = KI_color_dict[K_I]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    x = np.arange(np.shape(fluo)[-1])/12
    
    t0_spread = np.max(fluo[:,:,0], axis=1) - np.min(fluo[:,:,0], axis=1)
    tf_spread = np.max(fluo[:,:,-1], axis=1) - np.min(fluo[:,:,-1], axis=1)
        
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))

    x_list = [RMSE[:,0], RMSE[:,-1], t0_spread, tf_spread]
    x_label = ['t0 RMSE', 'tf RMSE', 't0 spread (ground truth)', 'tf spread (ground truth)']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'K_I={K_I}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)

for i in range(4):
    axes[i].legend()
plt.tight_layout()
plt.savefig(fig_path+'/figN_fig3_models_error_correlations_t0_tf.png',dpi=300)

#%% Cascade error correlations (stims)

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

alpha=0.3
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    df_index = (df_past['K_I']==K_I)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    color = KI_color_dict[K_I]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    stims, past_fluo, futures_fluo = get_eval_data(simul_id)
    t_on_past = np.sum(stims[:,-(past_steps+horizon):-horizon], axis=1)
    t_on_future = np.sum(~stims[:,-horizon:], axis=1)
    t_switches = np.sum(np.diff(stims[:, -horizon:], axis=1), axis=1)
    
    x_list = [t_on_past, t_on_future, t_switches]
    x_label = ['t on past','t on future', 'on/off switches']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    x_list = [t_on_past, t_on_future, t_switches, np.mean(futures_fluo[:,:,-1],axis=1)]
    x_label = ['t on past','t on future', 'on/off switches', 'avg fluorescence']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'camera_sim{camera_sim}, solver{solver}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)
    
plt.tight_layout()
plt.savefig(fig_path+'/figN_fig3_models_error_correlations_stim.png',dpi=300)

#%% Cascade 6 past steps error correlations (tf)

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & (df_meta['past_steps']==6) & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

alpha=0.3
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    df_index = (df_past['K_I']==K_I)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    color = KI_color_dict[K_I]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    x = np.arange(np.shape(fluo)[-1])/12
    
    t0_spread = np.max(fluo[:,:,0], axis=1) - np.min(fluo[:,:,0], axis=1)
    tf_spread = np.max(fluo[:,:,-1], axis=1) - np.min(fluo[:,:,-1], axis=1)
        
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))

    x_list = [RMSE[:,0], RMSE[:,-1], t0_spread, tf_spread]
    x_label = ['t0 RMSE', 'tf RMSE', 't0 spread (ground truth)', 'tf spread (ground truth)']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'K_I={K_I}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)

for i in range(4):
    axes[i].legend()
plt.tight_layout()
plt.savefig(fig_path+'/figN_fig3_6past_models_error_correlations_t0_tf.png',dpi=300)

#%% Cascade error correlations (stims)

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & (df_meta['past_steps']==6) & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1,4, 
                         figsize=(12,3))

alpha=0.1
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    df_index = (df_past['K_I']==K_I)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    color = KI_color_dict[K_I]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    stims, past_fluo, futures_fluo = get_eval_data(simul_id)
    t_on_past = np.sum(stims[:,-(past_steps+horizon):-horizon], axis=1)
    t_on_future = np.sum(~stims[:,-horizon:], axis=1)
    t_switches = np.sum(np.diff(stims[:, -horizon:], axis=1), axis=1)
    
    x_list = [t_on_past, t_on_future, t_switches]
    x_label = ['t on past','t on future', 'on/off switches']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    
    x_list = [t_on_past, t_on_future, t_switches, np.mean(futures_fluo[:,:,-1],axis=1)]
    x_label = ['t on past','t on future', 'on/off switches', 'avg fluorescence']
    
    y = RMSE[:,-1]
    y = np.mean(RMSE, axis=1)
    y_label = 'RMSE (time average)'
    
    for i, x in enumerate(x_list):
    
        axes[i].plot(x, y, '.',
                  color=color,
                  alpha=alpha,
                  label=f'camera_sim{camera_sim}, solver{solver}')
        
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel(y_label)
    
plt.tight_layout()
plt.savefig(fig_path+'/figN_fig3_6past_models_error_correlations_stim.png',dpi=300)

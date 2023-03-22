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

def get_fluo_pred(simul_id):
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
    fluo_pred = 4095 * np.load(f'{simul_dir}/evaluation/predictions.npy')
    
    return fluo_pred
    
def get_fluo_and_pred(simul_id, return_all=False):
    """
    Inputs
    ------
    simul_id: str (unique identifier of folder in `assets/models`)
    return_all: bool, optional
        Whether to return all possible futures
        Default is false
    
    Outputs
    -------
    fluo: arr, (n_cells, len(future))
        Evaluation data, future fluorescence
        if return_all==True, then fluo has shape: (n_cells, n_futures, len(future))
    fluo_pred: arr, (n_cells, len(future))
        Predicted future fluorescence
    
    """
        
    # Load predictions
    fluo_pred = get_fluo_pred(simul_id)

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




#%% Global plot styles

horizon_color_dict = {12: 'r', 24: 'g', 48: 'b'}
past_steps_color_dict = {12: 'r', 24: 'g', 36: 'b'}

#%% Check training data

plot_list = ['CcaSR_Cascade', 'CcaSR_FeedforwardPositive']
alpha = 0.3

for c, cell_class in enumerate(plot_list):
    plt.figure()
    
    training_data_dir_list = glob.glob(dcc_repo_path + f'/assets/simulated/data/{cell_class}/2023*')#_10-55*')
    
    for t, training_dir in enumerate(np.sort(np.array(training_data_dir_list))):
        fluo_training = np.load(training_dir+'/training_set/fluo1.npy')
        
        plt.hist(fluo_training.ravel(), 
                     bins = np.linspace(0, 4095+1, 50),
                     alpha=alpha, 
                     histtype='step',
                     linewidth=2,
                     density=True, 
                     label=str(t))
        
    plt.legend()
    plt.ylim([0,0.006])
    plt.title(cell_class)
    plt.ylabel('Frequency')
    plt.xlabel('fluorescence')
    plt.tight_layout()

#%% Cascade: responses to pure light, to show delays

n_hours = 6
n_cells = 50
random_bit = dcc.utilities.random_stimulations(
                        timepoints=n_hours*12*2,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*2*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
light_sequence = [1]*12*6 + [0]*12*6
x = [t/12. for t in range(len(light_sequence)+1)]

# Find new parameters
K_I_new_list = [60/1.5, 60, 60*1.5]
# K_I_new_list = [20, 40, 80, 160]
refcell = dcc.simulations.CcaSR_Cascade()

cascade_params_list = []
for K_I_new in K_I_new_list:
    
    H_ss = refcell.params['eta'] / refcell.params['nu'] 
    E_ss = refcell.params['h1'] / refcell.params['h2'] 
    I_ss_num = refcell.params['a'] * E_ss * H_ss**refcell.params['nh']
    I_ss_denom = refcell.params['nu'] * (refcell.params['K_H']**refcell.params['nh'] + H_ss**refcell.params['nh'])
    I_ss = I_ss_num / I_ss_denom
    a_update_num = (K_I_new**refcell.params['ni'] + I_ss**refcell.params['ni'])
    a_update_denom = (refcell.params['K_I']**refcell.params['ni'] + I_ss**refcell.params['ni'])
    a_update = a_update_num / a_update_denom
    
    new_params = {'K_I': K_I_new, 'a_F': a_update}
    cascade_params_list += [new_params]

# plot
fig, axes = plt.subplots(1, len(cascade_params_list), 
                         figsize=(5*len(cascade_params_list),4))
alpha=0.2
lw = 2

for p, new_params in enumerate(cascade_params_list):
    
    # Plot results
    plt.sca(axes[p])
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=250, x=x)
    for i in range(n_cells):
        # Simulate cell
        cell = dcc.simulations.CcaSR_Cascade()
        cell.update_params(new_params)
        cell.set_light_events(light_sequence)     
        series = cell.run(len(light_sequence)*5, 
                               solver="original")
    
        # axes[p].plot(x, [state['H'] for state in series],
        #               color='r',
        #               alpha=alpha, lw=lw)
        # axes[p].plot(x, [state['E'] for state in series],
        #               color='g',
        #               alpha=alpha, lw=lw)
        axes[p].plot(x, [state['I'] for state in series],
                      color='#6d418a',
                      alpha=alpha, lw=lw)
        axes[p].plot(x, [state['F'] for state in series],
                      color='b',
                      alpha=alpha, lw=lw)
    axes[p].set_xlabel("time (hours)")
    axes[p].set_ylabel("proteins (#)")
    axes[p].set_ylim(0,150)
    axes[p].set_title(f"K_I={new_params['K_I']}, a_F={new_params['a_F']:.2e}")
    
plt.tight_layout()
plt.savefig(fig_path+'/fig3_CcaSR_Cascade_ss_responses.png', dpi=300)

#%% Feedforward: responses to pure light, to show delays


# Solve for parameters
refcell = dcc.simulations.CcaSR_FeedforwardPositive()
starting_list = [{'K_I': 120, 'K_J': 120},
                   {'K_I': 120, 'K_J': 30},
                   {'K_I': 60, 'K_J': 120},
                   {'K_I': 60, 'K_J': 30},
                    ]

new_params_list = []
for starting in starting_list:
    
    K_I_new = starting['K_I']
    K_J_new = starting['K_J']
    
    eta = refcell.params['eta']
    nu = refcell.params['nu']
    h1 = refcell.params['h1']
    h2 = refcell.params['h2']
    K_H = refcell.params['K_H']
    nh = refcell.params['nh']
    K_I = refcell.params['K_I']
    ni = refcell.params['ni']
    K_J = refcell.params['K_J']
    nj = refcell.params['nj']
    a = refcell.params['a']
    
    H_ss = eta / nu
    E_ss = h1 / h2
    I_ss = (a * E_ss * H_ss**nh) / nu / (K_H**nh + H_ss**nh)
    
    J_ss = (a * E_ss * I_ss**ni) / nu / (K_I**ni + I_ss**ni)
    fac1 = I_ss**ni / (K_I**ni + I_ss**ni)
    fac2 = J_ss**nj / (K_J**nj + J_ss**nj)
    
    J_ss_new = (a * E_ss * I_ss**ni) / nu / (K_I_new**ni + I_ss**ni)
    fac1_new = I_ss**ni / (K_I_new**ni + I_ss**ni)
    fac2_new = J_ss_new**nj / (K_J_new**nj + J_ss_new**nj)
    
    a_update = (fac1 + fac2) / (fac1_new + fac2_new)
    
    new_params = {'K_J': K_J_new, 'K_I': K_I_new, 'a_F': a_update}
    new_params_list += [new_params]

# plot
fig, axes = plt.subplots(1, len(new_params_list), 
                         figsize=(5*len(new_params_list),4))

for p, new_params in enumerate(new_params_list):
    
    
    
    # Plot results
    plt.sca(axes[p])
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=250, x=x)
    
    for i in range(n_cells):
        # Simulate cell
        cell = dcc.simulations.CcaSR_FeedforwardPositive()
        cell.update_params(new_params)
        cell.set_light_events(light_sequence)     
        series = cell.run(len(light_sequence)*5, 
                               solver="original")
    
    
        axes[p].plot(x, [state['I'] for state in series],
                      color='#6d418a',
                      alpha=alpha, lw=lw)
        axes[p].plot(x, [state['J'] for state in series],
                      color = '#9c065d',
                      alpha=alpha, lw=lw)
        axes[p].plot(x, [state['F'] for state in series],
                     color='b',
                     alpha=alpha, lw=lw)
    axes[p].set_xlabel("time (hours)")
    axes[p].set_ylabel("proteins (#)")
    axes[p].set_ylim(0,150)
    axes[p].set_title(f"K_I={new_params['K_I']}, K_J={new_params['K_J']} ,a_F={new_params['a_F']:.2e}")
    
plt.tight_layout()
plt.savefig(fig_path+'/fig3_CcaSR_FeedforwardPositive_ss_responses.png', dpi=300)

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


#%% Cascade: error in each fluorescence bin

n_bins=20
q = 0.5

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
KI_vals = [40, 60, 90]
color_list = ['r','g','b']

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I',])

fig, axes = plt.subplots(1, 2, figsize=(9,5), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    df_index = (df_past['K_I']==K_I)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
    for i in range(2):    
        # Look at one hour
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        fluo = fluo[:,:,i*12:(i+1)*12]
        fluo_pred = fluo_pred[:,:,i*12:(i+1)*12]
    
        # Calulate error
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        bins = np.linspace(0,4100, n_bins+1)
        RMSE_bins = np.zeros((n_bins, 
                                np.shape(RMSE)[0]))
        
        # Look at error in each bin
        for b in range(n_bins):
            
            RMSE_bin = np.nan*np.ones_like(RMSE)
            RMSE_bin[(fluo>bins[b])&(fluo<bins[b+1])] = RMSE[(fluo>bins[b])&(fluo<bins[b+1])]
            # Median across each cell's future and time points?
            RMSE_bins[b] = np.nanmedian(RMSE_bin, axis=[1,2])
                
        axes[i].plot(bins[:-1], 
                     np.nanmedian(RMSE_bins, axis=[1,]),
                      '.-', color=color_list[k], 
                      label=f'K_I = {K_I}')
        axes[i].fill_between(
            bins[:-1],
            np.nanquantile(RMSE_bins, axis=1, q=0.5-q/2),
            np.nanquantile(RMSE_bins, axis=1, q=0.5+q/2),
            color=color_list[k],
            alpha=.2,
            )
    
        axes[i].legend()   
        axes[i].grid(True, 'both','both')
        axes[i].set_title(f'hour {i+1}')
        axes[i].set_xlabel('fluorescence')
        axes[i].set_ylabel(f'RMSE ({np.shape(fluo)[1]} futures of {np.shape(fluo)[0]} cells)\nMiddle {q*100:.0f}%')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig3_{cell_class}_qplot_err_fluor_bin.png',dpi=300)
    

#%% Cascade x horizon

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
KI_vals = [40, 60, 90]
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1, len(KI_vals), figsize=(10,3), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE,axis=[0,1]), '.', 
                     color = horizon_color_dict[horizon],
                     alpha=alpha,
                     label=f'horizon={horizon}')
    
    axes[k].set_title(f'K_I={K_I}')
    axes[k].grid(True, 'both', 'both')
    axes[k].legend()
    axes[k].set_xlabel('time (h)')
    axes[k].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_horizon.png', dpi=600)

#%% Cascade x past steps

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
KI_vals = [40, 60, 90]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(KI_vals), figsize=(10,3), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE, axis=[0,1]), '.', 
                     color = past_steps_color_dict[past_steps],
                     alpha=alpha,
                     label=f'past steps={past_steps}')
    
    axes[k].set_title(f'K_I={K_I}')
    axes[k].legend()
    axes[k].set_xlabel('time (h)')
    axes[k].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
    axes[k].grid(True, 'both', 'both')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_past_steps.png', dpi=600)


#%% FF+ error binned by fluorescence

n_bins=20
q = 0.5

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_FeedforwardPositive'
params_list = [{'K_I': 120, 'K_J': 120},
                   {'K_I': 120, 'K_J': 30},
                   {'K_I': 60, 'K_J': 120},
                   {'K_I': 60, 'K_J': 30},
                    ]
color_list = ['b','tab:purple','r','#db6837']

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','K_J'])

fig, axes = plt.subplots(1, 2, figsize=(9,5), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for p, params in enumerate(params_list):
    
    K_I = params['K_I']
    K_J = params['K_J']
    
    df_index = (df_past['K_I']==K_I)&(df_past['K_J']==K_J)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
    for i in range(2):    
        # Look at one hour
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        fluo = fluo[:,:,i*12:(i+1)*12]
        fluo_pred = fluo_pred[:,:,i*12:(i+1)*12]
    
        # Calulate error
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        bins = np.linspace(0,4100, n_bins+1)
        RMSE_bins = np.zeros((n_bins, 
                                np.shape(RMSE)[0]))
        
        # Look at error in each bin
        for b in range(n_bins):
            
            RMSE_bin = np.nan*np.ones_like(RMSE)
            RMSE_bin[(fluo>bins[b])&(fluo<bins[b+1])] = RMSE[(fluo>bins[b])&(fluo<bins[b+1])]
            # Median across each cell's future and time points?
            RMSE_bins[b] = np.nanmedian(RMSE_bin, axis=[1,2])
         
        axes[i].plot(bins[:-1], 
                     np.nanmedian(RMSE_bins, axis=[1,]),
                      '.-', color=color_list[p], 
                      label=f'K_I = {K_I}, K_J={K_J}')
        axes[i].fill_between(
            bins[:-1],
            np.nanquantile(RMSE_bins, axis=1, q=0.5-q/2),
            np.nanquantile(RMSE_bins, axis=1, q=0.5+q/2),
            color=color_list[p],
            alpha=.2,
            )
    
        axes[i].legend()   
        axes[i].set_title(f'hour {i+1}')
        axes[i].set_xlabel('fluorescence')
        axes[i].set_ylabel(f'RMSE ({np.shape(fluo)[1]} futures of {np.shape(fluo)[0]} cells)\nMiddle {q*100:.0f}%')
        axes[i].grid(True, 'both','both')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig3_{cell_class}_qplot_err_fluor_bin.png',dpi=300)
    

#%% FF x horizon

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_FeedforwardPositive'
params_list = [{'K_I': 120, 'K_J': 120},
                   {'K_I': 120, 'K_J': 30},
                   {'K_I': 60, 'K_J': 120},
                   {'K_I': 60, 'K_J': 30},
                    ]
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','K_J','horizon'])

fig, axes = plt.subplots(1, len(params_list), figsize=(10,3), sharey=True)
alpha = 0.5
for p, params in enumerate(params_list):
    K_I = params['K_I']
    K_J = params['K_J']
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['K_J']==K_J)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[p].plot(t, np.median(RMSE, axis=[0,1]), '.', 
                      color = horizon_color_dict[horizon],
                      alpha=alpha,
                      label=f'horizon={horizon}')
    
    axes[p].set_title(f'K_I={K_I}, K_J={K_J}')
    axes[p].grid(True, 'both','both')
    axes[p].legend()
    axes[p].set_xlabel('time (h)')
    axes[p].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(f'{fig_path}/fig3_{cell_class}_effect_of_horizon.png', dpi=600)

#%% Plot MEDIAN error as function of past steps 

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_FeedforwardPositive'
params_list = [{'K_I': 120, 'K_J': 120},
                   {'K_I': 120, 'K_J': 30},
                   {'K_I': 60, 'K_J': 120},
                   {'K_I': 60, 'K_J': 30},
                    ]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','K_J','past_steps'])

fig, axes = plt.subplots(1, len(params_list), figsize=(10,3), sharey=True)
alpha = 0.5
for p, params in enumerate(params_list):
    K_I = params['K_I']
    K_J = params['K_J']
    
    for pa, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['K_J']==K_J)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[p].plot(t, np.median(RMSE, axis=[0,1]), '.', 
                      color = past_steps_color_dict[past_steps],
                      alpha=alpha,
                      label=f'past steps={past_steps}')
    
    axes[p].set_title(f'K_I={K_I}, K_J={K_J}')
    axes[p].grid(True, 'both','both')
    axes[p].legend()
    axes[p].set_xlabel('time (h)')
    axes[p].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(f'{fig_path}/fig3_{cell_class}_effect_of_past_steps.png', dpi=600)

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

horizon_color_dict = {12: 'r', 24: 'g', 48: 'b'}

past_steps_color_dict = {3: "#ce5451",
                         6: "#bc8b3d",
                         12: "#73af3d",
                         24: "#5da071", 
                         36: "#588acf"}

KI_vals = [20,70]#[20, 40, 50, 70]
KI_colors = ["#d84d32",
            #  "#b27d3e",
            # "#dac54b",
            "#8f8539"]
KI_color_dict = {KI_vals[i]: KI_colors[i] for i in range(len(KI_vals))}

#%% Check training data

plot_list = ['CcaSR_Cascade', ]
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
alpha=0.3
lw =1

random_bit = dcc.utilities.random_stimulations(
                        timepoints=n_hours*12*2,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*2*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
light_sequence = [1]*12*6 + [0]*12*6
x = [t/12. for t in range(len(light_sequence)+1)]

# Find new parameters
refcell = dcc.simulations.CcaSR_Cascade()

cascade_params_list = []
for K_I_new in KI_vals:
    
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
                         figsize=(3*len(cascade_params_list),3))

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
    # axes[p].set_xlabel("time (hours)")
    # axes[p].set_ylabel("proteins (#)")
    # axes[p].set_xticklabels([])
    axes[p].set_yticklabels([])
    axes[p].set_ylim(0,150)
    axes[p].set_xlim([0,2*n_hours])
    # axes[p].set_title(f"K_I={new_params['K_I']}, a_F={new_params['a_F']:.2e}")
    
plt.tight_layout()
plt.savefig(fig_path+'/fig3_CcaSR_Cascade_ss_responses.png', dpi=300)

#%% Feedforward: responses to pure light, to show delays


# # Solve for parameters
# refcell = dcc.simulations.CcaSR_FeedforwardPositive()
# starting_list = [{'K_I': 120, 'K_J': 120},
#                    {'K_I': 120, 'K_J': 30},
#                    {'K_I': 60, 'K_J': 120},
#                    {'K_I': 60, 'K_J': 30},
#                     ]

# new_params_list = []
# for starting in starting_list:
    
#     K_I_new = starting['K_I']
#     K_J_new = starting['K_J']
    
#     eta = refcell.params['eta']
#     nu = refcell.params['nu']
#     h1 = refcell.params['h1']
#     h2 = refcell.params['h2']
#     K_H = refcell.params['K_H']
#     nh = refcell.params['nh']
#     K_I = refcell.params['K_I']
#     ni = refcell.params['ni']
#     K_J = refcell.params['K_J']
#     nj = refcell.params['nj']
#     a = refcell.params['a']
    
#     H_ss = eta / nu
#     E_ss = h1 / h2
#     I_ss = (a * E_ss * H_ss**nh) / nu / (K_H**nh + H_ss**nh)
    
#     J_ss = (a * E_ss * I_ss**ni) / nu / (K_I**ni + I_ss**ni)
#     fac1 = I_ss**ni / (K_I**ni + I_ss**ni)
#     fac2 = J_ss**nj / (K_J**nj + J_ss**nj)
    
#     J_ss_new = (a * E_ss * I_ss**ni) / nu / (K_I_new**ni + I_ss**ni)
#     fac1_new = I_ss**ni / (K_I_new**ni + I_ss**ni)
#     fac2_new = J_ss_new**nj / (K_J_new**nj + J_ss_new**nj)
    
#     a_update = (fac1 + fac2) / (fac1_new + fac2_new)
    
#     new_params = {'K_J': K_J_new, 'K_I': K_I_new, 'a_F': a_update}
#     new_params_list += [new_params]

# # plot
# fig, axes = plt.subplots(1, len(new_params_list), 
#                          figsize=(5*len(new_params_list),4))

# for p, new_params in enumerate(new_params_list):
    
    
    
#     # Plot results
#     plt.sca(axes[p])
#     dcc.utilities.OptoPlotBackground(light_sequence, ymax=250, x=x)
    
#     for i in range(n_cells):
#         # Simulate cell
#         cell = dcc.simulations.CcaSR_FeedforwardPositive()
#         cell.update_params(new_params)
#         cell.set_light_events(light_sequence)     
#         series = cell.run(len(light_sequence)*5, 
#                                solver="original")
    
    
#         axes[p].plot(x, [state['I'] for state in series],
#                       color='#6d418a',
#                       alpha=alpha, lw=lw)
#         axes[p].plot(x, [state['J'] for state in series],
#                       color = '#9c065d',
#                       alpha=alpha, lw=lw)
#         axes[p].plot(x, [state['F'] for state in series],
#                      color='b',
#                      alpha=alpha, lw=lw)
#     axes[p].set_xlabel("time (hours)")
#     axes[p].set_ylabel("proteins (#)")
#     axes[p].set_ylim(0,150)
#     axes[p].set_title(f"K_I={new_params['K_I']}, K_J={new_params['K_J']} ,a_F={new_params['a_F']:.2e}")
    
# plt.tight_layout()
# plt.savefig(fig_path+'/fig3_CcaSR_FeedforwardPositive_ss_responses.png', dpi=300)

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
h1_list = []
h2_list = []

for simul_id in simul_id_list:
    
    model_params, training_params = get_params(simul_id)
    
    horizon_list += [training_params['horizon']]
    past_steps_list += [training_params['past_steps']]
    training_folder_list += [training_params['training_sets'][0]]
    cell_class_list += [training_params['cell_class']]
    epochs_list += [training_params['training_parameters']['epochs']]
    datasets_list += [training_params['datasets_folder']]
    
    if 'h1' in model_params.keys():
        h1_list += [model_params['h1']]
    else: h1_list += [np.nan]
    if 'h2' in model_params.keys():
        h2_list += [model_params['h2']]
    else: h2_list += [np.nan]
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
        SE = (fluo - fluo_pred)**2
        bins = np.linspace(0,4100, n_bins+1)
        RMSE_bins = np.zeros((n_bins, 
                                np.shape(SE)[0]))
        
        # Look at error in each bin
        for b in range(n_bins):
            
            SE_bin = np.nan*np.ones_like(SE)
            SE_bin[(fluo>bins[b])&(fluo<bins[b+1])] = SE[(fluo>bins[b])&(fluo<bins[b+1])]
            # Median across each cell's future and time points?
            RMSE_bins[b] = np.sqrt(np.nanmean(SE_bin, axis=[1,2]))
                
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
    


#%% Plot select predictions

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon
df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I',])
past_steps_vals = [6,12,24,36]

percentile = [50, 250, 500, 750, 950]


alpha=0.5
lw=0.5
# Rather than plotting everything, just plot one replicate of each trained model
    
for k, K_I in enumerate(KI_vals):
    
    fig, axes = plt.subplots(len(percentile), 
                             len(past_steps_vals),
                              figsize=(3*len(past_steps_vals), 2*len(plot_list)),
                              sharex=True)
    
    for p, past_steps in enumerate(past_steps_vals):
        
        df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        color = KI_color_dict[K_I]
        
        # Find best and worst error
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE_avg_across_time = np.mean(np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1)),axis=1)
        sorted_error = np.argsort(RMSE_avg_across_time)
        
        plot_list = [sorted_error[i] for i in percentile]
        
        stims, past_fluo, futures_fluo = get_eval_data(simul_id)
        model_params, training_params = get_params(simul_id)
        
        plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
        cutoff = past_fluo.shape[1]
        
        
        for i, pl in enumerate(plot_list):
                
            # Stimulations
            plt.sca(axes[i,p])
            dcc.utilities.OptoPlotBackground(
                stims[pl,cutoff-plot_past:cutoff+training_params["horizon"]],
                x=np.arange(-plot_past, training_params["horizon"])/12,
                ymax = 4095
                )
            
            # Past
            axes[i,p].plot(np.arange(-plot_past, 0)/12, 
                          past_fluo[pl, -plot_past:],
                          color, lw=3*lw)
            
            # Future
            axes[i,p].plot(np.arange(0, training_params["horizon"])/12, 
                      futures_fluo[pl, :, :training_params['horizon']].T, 
                      color=color, alpha=0.01)
            
            # Prediction
            axes[i,p].plot(np.arange(0, training_params["horizon"])/12, 
                      fluo_pred[pl, 0],
                      color='k', lw=3*lw)
            
            # Prediction starts line
            axes[i,p].plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
            
            # Limits and labels
            axes[i,p].set_ylim([0,4095])
            axes[i,p].set_yticklabels([])
            axes[i,p].set_xticklabels([])
            # plt.title(f'Cell {c}')
            # axes[i].set_xlabel("time (h)")
            # axes[i].set_ylabel("Fluorescence (a.u.)")
            # axes[i].set_title(f'{100 * percentile[i] / len(sorted_error):.0f}th percentile')
            axes[i,p].set_title(f'K_I={K_I}, {past_steps}past steps')
        axes[-1,p].set_xlim([-plot_past/12, training_params["horizon"]/12])
    plt.tight_layout()
    plt.savefig(fig_path+f'/fig3_{cell_class}_predictions_percentiles_KI_{K_I}.png', dpi=300)

#%% Cascade average error

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
cascade_slice = default_training_size & default_past_steps \
    & default_horizon & (df_meta['cell_class']=='CcaSR_Cascade') \
        & (df_meta['K_I'].isin(KI_vals))
gillespie_slice = default_training_size & default_past_steps \
    & default_horizon & default_h1 & default_h2 & \
        (df_meta['solver']=='original') & (df_meta['cell_class']=='CcaSR_gillespie')

df_plot = df_meta.loc[cascade_slice + gillespie_slice].reset_index(drop=True)

alpha=0.5
xmax = 800
n_bins = 50
lw = 1
fig, axes = plt.subplots(1,2, figsize=(7,3))

# Rather than plotting everything, just plot one replicate of each trained model
for i in range(len(df_plot)):
    
    df_i = df_plot.loc[i]
    cell_class = df_i['cell_class']
    K_I = df_i['K_I']
    simul_id = df_i['simul_id']
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    if np.isnan(K_I):
        color='k'
        label=f'{cell_class}'
        zorder = 2
        histtype='step'
    elif K_I==20:
        color='tab:blue'
        label=f'{cell_class}, K_I={K_I}'
        zorder=1
        histtype='stepfilled'
    elif K_I==70:
        color = 'tab:orange'
        label=f'{cell_class}, K_I={K_I}'
        zorder=0
        histtype='stepfilled'
        
    axes[0].hist(np.mean(RMSE, axis=1),
             bins = np.linspace(0,xmax, n_bins+1),
              label=label,
              color=color,
              histtype=histtype,
              density=True,
              lw=lw,
              zorder=zorder,
              alpha=alpha)
    axes[1].plot(np.mean(fluo, axis=(1,2)),
                 np.mean(RMSE, axis=1),
                 '.', color=color, alpha=alpha/2)
        
# for i in range(2):
#     axes[i].set_xticklabels([])
#     axes[i].set_yticklabels([])
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_Cascade_error_overall.png', dpi=600)

#%% Cascade x horizon

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))

# Keep simulations with default training size, default past, and same h1/h2 ratio
cascade_slice = default_training_size & default_past_steps \
     & (df_meta['cell_class']=='CcaSR_Cascade') \
        & (df_meta['K_I'].isin(KI_vals))
gillespie_slice = default_training_size & default_past_steps \
     & default_h1 & default_h2 & \
        (df_meta['solver']=='original') & (df_meta['cell_class']=='CcaSR_gillespie')

df_plot = df_meta.loc[cascade_slice + gillespie_slice].reset_index(drop=True)

fig, axes = plt.subplots(1, len(horizon_vals), 
                         figsize=(3*len(horizon_vals),3), sharey=True)
alpha=0.5
xmax = 800
n_bins = 50
# Rather than plotting everything, just plot one replicate of each trained model
for i in range(len(df_plot)):
            
        df_i = df_plot.loc[i]
        cell_class = df_i['cell_class']
        K_I = df_i['K_I']
        horizon = df_i['horizon']
        simul_id = df_i['simul_id']
        
        ax = axes[np.where(horizon_vals==horizon)[0][0]]
        
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
        if np.isnan(K_I):
            ax.hist(RMSE[:,-1],
                     bins = np.linspace(0,xmax, n_bins+1),
                      label=f'{cell_class}',
                      color='k',
                      histtype='step',
                      density=True,
                      lw=lw,
                      zorder=0)
        else:
            ax.hist(RMSE[:,-1],
                     bins = np.linspace(0,xmax, n_bins+1),
                      label=f'{cell_class}, K_I={K_I}',
                      alpha=alpha,
                      density=True)
            
        ax.set_title(f'horizon={horizon}')
        
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_horizon.png', dpi=600)

#%% Cascade x Horizon extra

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))

# Keep simulations with default training size, default past, and same h1/h2 ratio
cascade_slice = default_training_size & default_past_steps \
     & (df_meta['cell_class']=='CcaSR_Cascade') \
        & (df_meta['K_I'].isin(KI_vals))
gillespie_slice = default_training_size & default_past_steps \
     & default_h1 & default_h2 & \
        (df_meta['solver']=='original') & (df_meta['cell_class']=='CcaSR_gillespie')

df_plot = df_meta.loc[cascade_slice + gillespie_slice].reset_index(drop=True)

fig, axes = plt.subplots(1, len(horizon_vals), 
                         figsize=(3*len(horizon_vals),3), sharey=True)
alpha=0.2
xmax = 800
n_bins = 50
# Rather than plotting everything, just plot one replicate of each trained model
for i in range(len(df_plot)):
            
        df_i = df_plot.loc[i]
        cell_class = df_i['cell_class']
        K_I = df_i['K_I']
        horizon = df_i['horizon']
        simul_id = df_i['simul_id']
        
        ax = axes[np.where(horizon_vals==horizon)[0][0]]
        
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
        if np.isnan(K_I):
            color='k'
            label=f'{cell_class}'
            zorder = 0
        elif K_I==20:
            color='tab:blue'
            label=f'{cell_class}, K_I={K_I}'
            zorder=2
        elif K_I==70:
            color = 'tab:orange'
            label=f'{cell_class}, K_I={K_I}'
            zorder=1

        ax.plot(np.mean(fluo[:,:,-1], axis=1),
                    RMSE[:,-1],
                            '.',
                      label=label,
                      color=color,
                      alpha=alpha,
                      zorder=zorder)
            
        ax.set_title(f'horizon={horizon}')
        ax.set_ylim([0,1600])
        
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_horizon_endpoint_fluor.png', dpi=600)



# #%% Cascade x horizon

# # Which cell class, h1, and horizon values to plot
# cell_class = 'CcaSR_Cascade'
# horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# # Keep simulations with default training size, default past, and same h1/h2 ratio
# simul_slice = default_training_size & default_past_steps

# df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
# df_past = df_past.sort_values(by=['K_I','horizon'])

# fig, axes = plt.subplots(2, len(KI_vals), figsize=(10,5), sharey=True)
# alpha=0.5
# xmax = 1500
# n_bins = 100
# # Rather than plotting everything, just plot one replicate of each trained model
# for k, K_I in enumerate(KI_vals):
    
#     for ho, horizon in enumerate(horizon_vals):
#         df_index = (df_past['K_I']==K_I)&(df_past['horizon']==horizon)
#         simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
#         fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
#         RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
#         axes[0, k].hist(RMSE[:,-1],
#                  bins = np.linspace(0,xmax, n_bins+1),
#                   color = horizon_color_dict[horizon],
#                   label=f'horizon={horizon}',
#                   alpha=alpha,
#                   density=True)
#         axes[1, k].hist(RMSE[:,11],
#                  bins = np.linspace(0,xmax, n_bins+1),
#                   color = horizon_color_dict[horizon],
#                   label=f'horizon={horizon}',
#                       alpha=alpha,
#                       density=True)
    
#     axes[0,k].set_xlabel('Endpoint RMSE')
#     axes[1,k].set_xlabel('1h RMSE')
    
#     for i in range(2):
#         axes[i,k].set_xlim([0, xmax])
#         axes[i, 0].legend()
#         axes[i,k].set_title(f'K_I={K_I}')
        
# plt.tight_layout()
# plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_horizon.png', dpi=600)


#%% Cascade x horizon x time

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
df_past = df_past.sort_values(by=['K_I','horizon'])

fig, axes = plt.subplots(1, len(KI_vals), figsize=(10,3), sharey=True)
alpha=0.5
q = 0.5
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
                     color = horizon_color_dict[horizon],
                     label=f'horizon={horizon}')
        axes[k].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=horizon_color_dict[horizon],
            alpha=alpha,
            )
    
    axes[k].set_title(f'K_I={K_I}')
    axes[k].grid(True, 'both', 'both')
    axes[k].legend()
    axes[k].set_xlabel('time (h)')
    axes[k].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_horizon_with_time.png', dpi=600)
#%% Cascade x past_steps

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
past_steps_vals = np.sort(np.unique(past_steps_list))[1:]

# Keep simulations with default training size, default past, and same h1/h2 ratio
cascade_slice = default_training_size & default_horizon \
     & (df_meta['cell_class']=='CcaSR_Cascade') \
        & (df_meta['K_I'].isin(KI_vals)) & (df_meta['past_steps'].isin(past_steps_vals))
gillespie_slice = default_training_size & default_horizon \
     & default_h1 & default_h2 & (df_meta['past_steps'].isin(past_steps_vals)) \
        & (df_meta['solver']=='original') & (df_meta['cell_class']=='CcaSR_gillespie')

df_plot = df_meta.loc[cascade_slice + gillespie_slice].reset_index(drop=True)

fig, axes = plt.subplots(1, 3, 
                         figsize=(3*2,2), sharey=False)
markersize=8
lw=1
# Rather than plotting everything, just plot one replicate of each trained model
for i in range(len(df_plot)):
            
    df_i = df_plot.loc[i]
    cell_class = df_i['cell_class']
    K_I = df_i['K_I']
    past_steps = df_i['past_steps']
    simul_id = df_i['simul_id']
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
    plot_list = [np.mean(RMSE,axis=1),
                 RMSE[:,-1],
                 RMSE[:,0]]
    y_label = ['RMSE (average over time)',
               'RMSE (final timepoint)',
               'RMSE (initial timepoint)']

    if np.isnan(K_I):
        histtype = 'step'
        color = 'k'
        nudge = -1/5
        if past_steps==36:
            for j, y in enumerate(plot_list):
                axes[j].plot([-1,len(past_steps_vals)],
                         np.median(y)*np.ones(2),'--',
                         color=0.5*np.ones(3), zorder=0)
        
    else:
        histtype='stepfilled'
        if K_I==20:
            color='tab:blue'
            nudge = 0
        elif K_I==70:
            color = 'tab:orange'
            nudge = 1/5
        
    for j, y in enumerate(plot_list):
        x = np.where(past_steps_vals==past_steps)[0][0]
        axes[j].plot(x+nudge,
                     np.median(y),
                     '.', markersize=markersize,
                     color=color)
        axes[j].plot([x+nudge, x+nudge],
                     [np.quantile(y, q=0.25), np.quantile(y,q=0.75)],
                     '-',lw=lw,
                     color=color)
        # axes[j].set_ylabel(y_label[j])
        axes[j].set_ylim([0,800])
        axes[j].set_xlim([-0.5,len(past_steps_vals)-0.5])
        axes[j].set_xticks(np.arange(len(past_steps_vals)))
        
plt.tight_layout()
plt.savefig(dcc_repo_path+'/assets/figures/fig3_Cascade_effect_of_past_steps.png', dpi=600)


# # Which cell class, h1, and horizon values to plot
# cell_class = 'CcaSR_Cascade'
# past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
# past_steps_vals = [6, 12, 24, 36]

# # Keep simulations with default training size, default past, and same h1/h2 ratio
# simul_slice = default_training_size & default_horizon

# df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]
# df_past = df_past.sort_values(by=['K_I','past_steps'])

# fig, axes = plt.subplots(1, len(KI_vals), figsize=(10,3), sharey=True)
# alpha=0.5
# xmax = 50_000
# n_bins = 100

# # Rather than plotting everything, just plot one replicate of each trained model
# for k, K_I in enumerate(KI_vals):
    
#     for p, past_steps in enumerate(past_steps_vals):
#         df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
#         simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
#         fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
#         RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
#         axes[k].hist(np.sum(RMSE, axis=1),
#                  bins = np.linspace(0,xmax, n_bins+1),
#                   color = past_steps_color_dict[past_steps],
#                   label=f'past steps={past_steps}',
#                   alpha=alpha,
#                   density=True)
    
#     axes[k].set_xlabel('Total RMSE')
    
#     axes[k].set_xlim([0, xmax])
#     axes[0].legend()
#     axes[k].set_title(f'K_I={K_I}')
        
# plt.tight_layout()
# plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_past_steps.png', dpi=600)

#%% Cascade x past steps x time

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [6, 12, 24, 36]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(KI_vals), figsize=(10,3), sharey=True)
alpha=0.5
q=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
                     color = past_steps_color_dict[past_steps],
                     label=f'past steps={past_steps}')
        axes[k].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=past_steps_color_dict[past_steps],
            alpha=alpha,
            )
    
    axes[k].set_title(f'K_I={K_I}')
    # axes[k].legend()
    axes[k].set_xlabel('time (h)')
    axes[k].grid(True, 'both', 'both')
axes[0].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_effect_of_past_steps_over_time.png', dpi=600)

#%% Cascade x past steps - time 0 error

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [6, 12, 24, 36]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(2, len(KI_vals), figsize=(12,6), 
                         sharey=True, sharex=True)
alpha=0.5
q=0.5
x_max = 2000
n_bins = 100

# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[0,k].hist(RMSE[:,0],bins=np.linspace(0,x_max, n_bins+1),
                     color=past_steps_color_dict[past_steps],
                     alpha=alpha, density=True,
                     label=f'{past_steps} past steps')
        axes[1,k].hist(RMSE[:,-1],bins=np.linspace(0,x_max, n_bins+1),
                     color=past_steps_color_dict[past_steps],
                     alpha=alpha, density=True)
        # axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
        #              color = past_steps_color_dict[past_steps],
        #              label=f'past steps={past_steps}')
        # axes[k].fill_between(
        #     t,
        #     np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
        #     np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
        #     color=past_steps_color_dict[past_steps],
        #     alpha=alpha,
        #     )
    
    axes[0,k].set_title(f'K_I={K_I}, t0 error')
    axes[1,k].set_title(f'K_I={K_I}, tf error')
    axes[0,k].legend()
    # axes[k].legend()
    for i in range(2):
        axes[i,k].set_xlabel('RMSE')
        axes[i,k].grid(True, 'both', 'both')
        axes[0,i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_past_steps_t0_err.png', dpi=600)


#%% [by cell] Cascade x past steps x time

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_Cascade'
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [6, 12, 24, 36]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(len(past_steps_vals), len(KI_vals), 
                         figsize=(3*len(KI_vals),3*len(past_steps_vals)), 
                         sharey=False)
alpha=0.03
# Rather than plotting everything, just plot one replicate of each trained model
for k, K_I in enumerate(KI_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['K_I']==K_I)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k,p].plot(t, RMSE.T, '.-',
                     alpha=alpha,
                     color=past_steps_color_dict[past_steps],
                     label=f'past steps={past_steps}')
        # axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
        #              color = past_steps_color_dict[past_steps],
        #              label=f'past steps={past_steps}')
        # axes[k].fill_between(
        #     t,
        #     np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
        #     np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
        #     color=past_steps_color_dict[past_steps],
        #     alpha=alpha,
        #     )
    
        axes[k,p].set_title(f'K_I={K_I}')
        # axes[k].legend()
        axes[k,p].set_xlabel('time (h)')
        axes[k,p].grid(True, 'both', 'both')
axes[0,0].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig3_{cell_class}_past_steps_over_time_individual_cells.png', dpi=600)

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
        SE = (fluo - fluo_pred)**2
        bins = np.linspace(0,4100, n_bins+1)
        RMSE_bins = np.zeros((n_bins, 
                                np.shape(SE)[0]))
        
        # Look at error in each bin
        for b in range(n_bins):
            
            SE_bin = np.nan*np.ones_like(SE)
            SE_bin[(fluo>bins[b])&(fluo<bins[b+1])] = SE[(fluo>bins[b])&(fluo<bins[b+1])]
            # Median across each cell's future and time points?
            RMSE_bins[b] = np.sqrt(np.nanmean(SE_bin, axis=[1,2]))
         
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
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
                     color = horizon_color_dict[horizon],
                     label=f'horizon={horizon}')
        axes[k].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=horizon_color_dict[horizon],
            alpha=alpha,
            )
    
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
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[k].plot(t, np.median(RMSE,axis=[0,]), '.', 
                     color = past_steps_color_dict[past_steps],
                     label=f'past steps={past_steps}')
        axes[k].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=past_steps_color_dict[past_steps],
            alpha=alpha,
            )
    
    axes[p].set_title(f'K_I={K_I}, K_J={K_J}')
    axes[p].grid(True, 'both','both')
    axes[p].legend()
    axes[p].set_xlabel('time (h)')
    axes[p].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(f'{fig_path}/fig3_{cell_class}_effect_of_past_steps.png', dpi=600)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:48:08 2023

@author: hklumpe
"""

import glob
import sys
import json
import os

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

df_simul_config = pd.DataFrame({'cell_class': ['CcaSR_gillespie_simple_noE',
                                               'CcaSR_gillespie_simple_noE',
                                               'CcaSR_gillespie_simple_noE',
                                               'CcaSR_gillespie'],
                                'camera_sim': [False, True, True, True],
                                'solver': ['ode','ode','original','original'],
                                'color': ["#c561b0",
                                            "#57377b",
                                            "#5a6eb8",
                                            "#327a42",]}
                            )

#%% Get information about simulations

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023*')
                    
simul_id_list = []
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

for s, simul_dir in enumerate(simul_dir_list):
    
    if os.path.isfile(f'{simul_dir}/training_parameters.json'):
        
        simul_id = simul_dir.split('/')[-1]
        simul_id_list += [simul_id,]
            
        model_params, training_params = get_params(simul_id)
        
        horizon_list += [training_params['horizon']]
        past_steps_list += [training_params['past_steps']]
        training_folder_list += [training_params['training_sets'][0]]
        cell_class_list += [training_params['cell_class']]
        epochs_list += [training_params['training_parameters']['epochs']]
        datasets_list += [training_params['datasets_folder']]
        
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
                        'cell_class': cell_class_list,
                        'training_sets': training_folder_list,
                        'epochs': epochs_list,
                        'datasets_folder': datasets_list,})

# Drop repeats
df_meta = df_meta.sort_values(by=['simul_id','past_steps','horizon','h1','h2'])
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

#%% Plotting functions
def plot_err_distribution(err = 'RMSE',
        n_bins=100, x_max = 610, histtype='stepfilled',
        alpha=0.7, fig_path = fig_path,):

    plt.figure(figsize=(5,3))
    
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
        
        if err=='RMSE':
            
            x = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
        elif err=='Normalized_RMSE':
            
            x = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            x = x / fluo_pred[:,0]
            
        elif err=='Error_predicting_avg':
            
            fluo_avg = np.mean(fluo, axis=1)
                
            x = np.abs(fluo_pred[:,0] - fluo_avg) / fluo_pred[:,0]
        else:
            print('Wrong error measure (see kwarg "err")')
        
        plt.hist(np.mean(x, axis=1), 
                  bins = np.linspace(0,x_max,n_bins+1),
                  color=color,
                  alpha=alpha,
                  histtype=histtype,
                  linewidth=1.5,
                  density=True)
        
    plt.xlabel(f'{err} (time average)')
    plt.ylabel('Frequency')
    plt.xlim([0,x_max])
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_{err}_distribution.png',dpi=300)

def plot_percentile_predictions(err = 'RMSE',
                                percentile = [50, 500, 950],
                                plot_past = 3*12,
                                lw = 0.75,):
    
    fig, axes = plt.subplots(len(percentile), len(df_simul_config),
                              figsize=(3*len(df_simul_config), 2*len(percentile)),
                              sharex=True)
    
    for c in range(len(df_simul_config)):
        
        config = df_simul_config.loc[c]
        camera_sim = config['camera_sim']
        solver = config['solver']
        cell_class = config['cell_class']
        color = config['color']
        config_bool = (df_meta['camera_sim']==camera_sim)&(df_meta['solver']==solver)&(df_meta['cell_class']==cell_class)
        
        simul_id = df_meta.loc[default_models & config_bool, 'simul_id'].values[0]
        
        # Find best and worst error
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        
        if err=='RMSE':
            x = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
        elif err=='Normalized_RMSE':
            
            x = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            x = x / fluo_pred[:,0]
            
        elif err=='Error_predicting_avg':
            
            fluo_avg = np.mean(fluo, axis=1)
                
            x = np.abs(fluo_pred[:,0] - fluo_avg) / fluo_pred[:,0]
        else:
            print('Wrong error measure (see kwarg "err")')
        
        x_avg = np.mean(x, axis=1)
        sorted_error = np.argsort(x_avg)
        
        plot_list = [sorted_error[i] for i in percentile]
        
        stims, past_fluo, futures_fluo = get_eval_data(simul_id)
        model_params, training_params = get_params(simul_id)
        
        cutoff = past_fluo.shape[1]
        
        
        for i, pl in enumerate(plot_list):
            ax = axes[i,c]
                
            # Stimulations
            plt.sca(ax)
            dcc.utilities.OptoPlotBackground(
                stims[pl,cutoff-plot_past:cutoff+training_params["horizon"]],
                x=np.arange(-plot_past, training_params["horizon"])/12,
                ymax = 4095
                )
            
            # Past
            ax.plot(np.arange(-plot_past, 0)/12, 
                          past_fluo[pl, -plot_past:],
                          color, lw=3*lw)
            
            # Future
            ax.plot(np.arange(0, training_params["horizon"])/12, 
                      futures_fluo[pl, :, :training_params['horizon']].T, 
                      color=color, alpha=0.01)
            ax.plot(np.arange(0, training_params["horizon"])/12, 
                      np.mean(futures_fluo[pl, :, :training_params['horizon']], axis=0), 
                      color=0.6*np.ones(3), alpha=1, lw=3*lw)
            
            # Prediction
            ax.plot(np.arange(0, training_params["horizon"])/12, 
                      fluo_pred[pl, 0],
                      color='k', lw=3*lw)
            
            # Prediction starts line
            ax.plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
            
            # Limits and labels
            ax.set_ylim([0,4095])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # plt.title(f'Cell {c}')
            # axes[i].set_xlabel("time (h)")
            # axes[i].set_ylabel("Fluorescence (a.u.)")
            # axes[i].set_title(f'{100 * percentile[i] / len(sorted_error):.0f}th percentile')
        axes[-1,c].set_xlim([-plot_past/12, training_params["horizon"]/12])
        # plt.suptitle(f'{cell_class}, camera_sim {camera_sim}, solver:{solver}')
        plt.tight_layout()
        plt.savefig(fig_path+f'/fig1_{cell_class}_predictions_percentiles_{err}.png', dpi=300)

def plot_err_over_time(err = 'RMSE',
                       q = 0.5,
                       alpha = 0.3, 
                       y_max = 450):
    
    plt.figure(figsize=(5,3))
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
        
        if err == 'RMSE':
            
            e = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
        elif err=='Normalized_RMSE':
            
            e = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            e = e / fluo_pred[:,0]
            
        elif err=='Error_predicting_avg':
            
            fluo_avg = np.mean(fluo, axis=1)
                
            e = np.abs(fluo_pred[:,0] - fluo_avg) / fluo_pred[:,0]
            
        else:
            print('Wrong error measure (see kwarg "err")')
        
        plt.plot(x, np.median(e, axis=0),
                      '.-', color=color, )
                      # label=f'{cell_class}, camera noise {camera_sim}, solver: {solver})
        
        # Lightweight shading: 95%
        plt.fill_between(
            x,
            np.nanquantile(e, axis=0, q=0.5 - q/2),
            np.nanquantile(e, axis=0, q=0.5 + q/2),
            color=color,
            alpha=alpha,
            lw=0,
            )
        
    # plt.legend()   
    plt.xlabel('time( h)')
    plt.ylabel(f'RMSE: Middle {q*100:.0f}%')
    plt.title(f'{err}')
    plt.ylim([0,y_max])
    plt.grid(True, 'both','both')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_qplot_{err}_over_time.png',dpi=300)
    
    
def plot_individual_traces_binned(err = 'Error_predicting_avg',
                q_list = [[0.9,1.0], [0.45, 0.55], [0,0.1]],
                alpha = 0.3, lw=0.5,
                y_max = 0.5):
    

    fig, axes = plt.subplots(1, len(df_simul_config),
                             figsize=(2.5*len(df_simul_config), 4),
                             sharey=True)
    for c in range(len(df_simul_config)):
        
        ax = axes[c]
            
        config = df_simul_config.loc[c]
        camera_sim = config['camera_sim']
        solver = config['solver']
        cell_class = config['cell_class']
        color = config['color']
        config_bool = (df_meta['camera_sim']==camera_sim)&(df_meta['solver']==solver)&(df_meta['cell_class']==cell_class)
        
        simul_id = df_meta.loc[default_models & config_bool, 'simul_id'].values[0]
        
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
        x = np.arange(np.shape(fluo)[-1])/12
        
        if err == 'RMSE':
            
            e = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
        elif err=='Normalized_RMSE':
            
            e = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            e = e / fluo_pred[:,0]
            
        elif err=='Error_predicting_avg':
            
            fluo_avg = np.mean(fluo, axis=1)
                
            e = np.abs(fluo_pred[:,0] - fluo_avg) / fluo_pred[:,0]
            
        else:
            print('Wrong error measure (see kwarg "err")')
            
        q_color = ['#080708', color, '#9c9c9c']
            
        for q, q_bound in enumerate(q_list):
            
            e_ind = np.argsort(np.mean(e,axis=1))[int(q_bound[0]*1000):int(q_bound[1]*1000)]
            
            ax.plot(x, e[e_ind].T,
                          '-', 
                          color=q_color[q], 
                          alpha=alpha,
                          linewidth=lw)
        
        ax.set_xlabel('time( h)')
        ax.set_ylim([0,y_max])
        ax.grid(True, 'both','both')
    
    axes[0].set_ylabel(f'{err}')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_{err}_over_time_individual_traces.png',dpi=300)

#%% Plot responses to pure light

n_hours = 4
n_cells = 1
alpha = 0.8
lw = 2.5

# Get light
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
# light_sequence = [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24
x = [t/12. for t in range(len(light_sequence)+1)]

for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]
    
    # Plot light
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=4095, x=x)

    for i in range(n_cells):
        cell_class = config['cell_class']
        if cell_class=='CcaSR_gillespie_simple_noE':
            cell = dcc.simulations.CcaSR_gillespie_simple_noE()
        elif cell_class=='CcaSR_gillespie':
            cell = dcc.simulations.CcaSR_gillespie()
        else:
            print('Wrong class string')
        cell.set_light_events(light_sequence) 
        
        series = cell.run(len(light_sequence)*5, 
                               solver=config['solver'])
        
        fluo = [state['F'] for state in series]
            
        fluo = np.array(fluo)
        if config['camera_sim']: 
            fluo = dcc.simulations.camera_sim(fluo)
        else:
            fluo = dcc.simulations.camera_sim(fluo, noise_perc=0)
    
        # Plot results
        plt.plot(x, fluo, 
                  lw=lw, 
                  alpha=alpha,
                  color=config['color'])
        
plt.xlabel('time (h)')
plt.ylabel('Fluorescence (AU)')
plt.xlim([0,20])
plt.ylim([0,4095])
plt.tight_layout()
plt.savefig(f'{fig_path}/fig1_sample_activation_different_noise.png',
            dpi=300)

#%% Plot responses to pure light (verbose)

n_hours = 4
n_cells = 1
alpha = 0.8
lw = 2.5

H_color = '#e89f3f'
E_color = 'g'

# Get light
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
# light_sequence = [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24
x = [t/12. for t in range(len(light_sequence)+1)]

fig, axes = plt.subplots(5,1, figsize=(6,20))

for i in range(5):
    plt.sca(axes[i])
    
    # Plot light
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=4095, x=x)

for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]
    
    for i in range(n_cells):
        cell_class = config['cell_class']
        if cell_class=='CcaSR_gillespie_simple_noE':
            cell = dcc.simulations.CcaSR_gillespie_simple_noE()
        elif cell_class=='CcaSR_gillespie':
            cell = dcc.simulations.CcaSR_gillespie()
        else:
            print('Wrong class string')
        cell.set_light_events(light_sequence) 
        
        series = cell.run(len(light_sequence)*5, 
                               solver=config['solver'])
        
        fluo = [state['F'] for state in series]
            
        fluo = np.array(fluo)
        if config['camera_sim']: 
            fluo = dcc.simulations.camera_sim(fluo)
        else:
            fluo = dcc.simulations.camera_sim(fluo, noise_perc=0)
    
        # Plot all results in subpanel 1
        axes[0].plot(x, fluo, 
                  lw=lw, 
                  alpha=alpha,
                  color=config['color'])
        
        # For deterministic solution without noise
        if (config['camera_sim']==False) & (config['solver']=='ode'):
            axes[1].plot(x, [state['F'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color='b')
            axes[1].plot(x, [state['H'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color=H_color)
        # For deterministic solutions
        if config['solver']=='ode':
            axes[2].plot(x, fluo,
                         lw=lw,
                         alpha=alpha,
                         color=config['color'])
        
        # for stochastc with no E
        if (config['solver']=='original') & (config['cell_class']=='CcaSR_gillespie_simple_noE'):
            axes[3].plot(x, [state['F'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color='b')
            axes[3].plot(x, [state['H'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color=H_color)
            
        # for stochastic with  E
        if (config['solver']=='original') & (config['cell_class']=='CcaSR_gillespie'):
            axes[4].plot(x, [state['H'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color=H_color)
            axes[4].plot(x, [state['E'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color=E_color)
            axes[4].plot(x, [state['F'] for state in series],
                         lw=lw, 
                         alpha=alpha,
                         color='b')
        
    
for i in range(5):
    axes[i].set_xlabel('Time (h)')
    axes[i].set_xlim([0,20])
    if i in [0, 2]:
        axes[i].set_ylabel('Fluorescence (AU)')
        axes[i].set_ylim([0,4095])
    else:
        axes[i].set_ylabel('# proteins')
        axes[i].set_ylim([0,120])

plt.tight_layout()
for extension in ['png','svg']:
    plt.savefig(f'{fig_path}/fig1_sample_activation_different_noise_verbose.{extension}',
            dpi=300)

#%% Error distribution

plot_err_distribution(err = 'RMSE')
plot_err_distribution(err = 'Normalized_RMSE', 
                      x_max = 0.3)
plot_err_distribution(err = 'Error_predicting_avg', 
                      x_max = 0.3, histtype='step')

#%% Q-plot of error  v. time 

plot_err_over_time(err='RMSE')
plot_err_over_time(err='Normalized_RMSE',
                  y_max = 0.3)
plot_err_over_time(err='Error_predicting_avg',
                  y_max = 0.15)

#%% Plot select predictions

plot_percentile_predictions(err = 'Normalized_RMSE',
                            percentile = [50, 250, 500, 750, 950])
plot_percentile_predictions(err = 'Error_predicting_avg')

#%% Time evolution of error for worst predictions

plot_individual_traces_binned(err='Error_predicting_avg', lw=0.5)
plot_individual_traces_binned(err='Normalized_RMSE', lw=0.5)
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


#%% Global plot styles

model_color_dict = {'CcaSR_gillespie_simple_noE': 'r',
              'CcaSR_gillespie': 'k'}
model_list = [key for key in model_color_dict.keys()]



#%% Check training data

plot_list = ['CcaSR_gillespie_simple_noE', 
              'CcaSR_gillespie']
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
    plt.ylim([0,0.008])
    plt.title(cell_class)
    plt.ylabel('Frequency')
    plt.xlabel('fluorescence')
    plt.tight_layout()
    


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

df_simul_config = pd.DataFrame({'camera_sim': [True, True, False],
                                'solver': ['original','ode', 'ode'],
                                'color': ['b', 'g', 'r']}
                            )


#%% Plot responses to pure light

n_hours = 4
n_cells = 3
alpha = 0.5
lw = 2

# Get light
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
# light_sequence = [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24
x = [t/12. for t in range(len(light_sequence)+1)]

# Only for plot title, not instantiation
plt.figure()
cell_class = 'CcaSR_gillespie_simple_noE'

for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]
    
    # Plot light
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=4095, x=x)

    for i in range(n_cells):
        cell = dcc.simulations.CcaSR_gillespie_simple_noE()
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
plt.title(cell_class)
plt.tight_layout()
plt.savefig(f'{fig_path}/fig1_{cell_class}_off-on_camera_noise_stochasticity.png',
            dpi=300)

# Only for plot title, not instantiation
plt.figure()
cell_class = 'CcaSR_gillespie'

for c in range(len(df_simul_config)):
    
    config = df_simul_config.loc[c]    
    
    # Plot light
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=4095, x=x)

    for i in range(n_cells):
        cell = dcc.simulations.CcaSR_gillespie_simple_noE()
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
plt.title(cell_class)
plt.tight_layout()
plt.savefig(f'{fig_path}/fig1_{cell_class}_off-on_camera_noise_stochasticity.png',
            dpi=300)



#%% Plot select predictions

default_models = default_training_size & default_past_steps & default_horizon & default_h1 & default_h2

for cell_class in ['CcaSR_gillespie_simple_noE', 'CcaSR_gillespie']:
        
    df_cc = df_meta.loc[default_models & (df_meta['cell_class']==cell_class)].reset_index()
    
    for c in range(len(df_cc)):
        simul_id = df_cc.loc[c,'simul_id']
        camera_sim = df_cc.loc[c,'camera_sim']
        solver = df_cc.loc[c,'solver']
        color = df_simul_config.loc[(df_simul_config['camera_sim']==camera_sim)&(df_simul_config['solver']==solver),'color'].values[0]

        # Find best and worst error, arbitrarily relative to the first cell future
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE_sum_across_time = np.sum(np.sqrt((fluo - fluo_pred)**2), axis=1)
        sorted_error = np.argsort(RMSE_sum_across_time)
        
        percentile = [50, 500, 950]
        plot_list = [sorted_error[i] for i in percentile]
        
        stims, past_fluo, futures_fluo = get_eval_data(simul_id)
        model_params, training_params = get_params(simul_id)
        
        plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
        cutoff = past_fluo.shape[1]
        
        fig, axes = plt.subplots(len(plot_list), 1,
                                 figsize=(4, 3*len(plot_list)),
                                 sharex=True)
        for i, pl in enumerate(plot_list):
                
            # Stimulations
            plt.sca(axes[i])
            dcc.utilities.OptoPlotBackground(
                stims[pl,cutoff-plot_past:cutoff+training_params["horizon"]],
                x=np.arange(-plot_past, training_params["horizon"])/12,
                ymax = 4095
                )
            
            # Past
            axes[i].plot(np.arange(-plot_past, 0)/12, 
                         past_fluo[pl, -plot_past:],
                         "k")
            
            # Future
            axes[i].plot(np.arange(0, training_params["horizon"])/12, 
                     futures_fluo[pl, 0, :training_params['horizon']], 
                     "k")
            
            # Prediction
            axes[i].plot(np.arange(0, training_params["horizon"])/12, 
                     fluo_pred[pl],
                     "b")
            
            # Prediction starts line
            axes[i].plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
            
            # Limits and labels
            axes[i].set_ylim([0,4095])
            # plt.title(f'Cell {c}')
            axes[i].set_xlabel("time (h)")
            axes[i].set_ylabel("Fluorescence (a.u.)")
            axes[i].set_title(f'{100 * percentile[i] / len(sorted_error):.0f}th percentile')
        axes[-1].set_xlim([-plot_past/12, training_params["horizon"]/12])
        plt.suptitle(f'{cell_class}, camera_sim {camera_sim}, solver:{solver}')
        plt.tight_layout()
        plt.savefig(fig_path+f'/fig1_{cell_class}_predictions_percentiles_config_{color}.png', dpi=300)

#%% TO DO: fix this /Q-plot of error  v. fluorescence(better for overlay?)

n_bins=200
q = 0.5
default_models = default_training_size & default_past_steps & default_horizon & default_h1 & default_h2

for cell_class in ['CcaSR_gillespie_simple_noE', 'CcaSR_gillespie']:
    fig, axes = plt.subplots(1,2, figsize=(10,3), sharey=True)
                                           
    df_cc = df_meta.loc[default_models & (df_meta['cell_class']==cell_class)].reset_index()
    
    for c in range(len(df_cc)):
        simul_id = df_cc.loc[c,'simul_id']
        camera_sim = df_cc.loc[c,'camera_sim']
        solver = df_cc.loc[c,'solver']
        color = df_simul_config.loc[(df_simul_config['camera_sim']==camera_sim)&(df_simul_config['solver']==solver),'color'].values[0]
    
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
        
            axes[i].plot(bins[:-1], RMSE_median_bins,
                          '.-', color=color, 
                          label=f'camera noise {camera_sim}, solver: {solver}')
            axes[i].fill_between(
                bins[:-1],
                RMSE_extrema_bins[:,0],
                RMSE_extrema_bins[:,1],
                color=color,
                alpha=.2,
                )
        
            axes[i].legend()   
            axes[i].set_title(f'hour {i+1}')
            axes[i].set_xlabel('fluorescence')
            axes[i].set_ylabel(f'RMSE ({np.shape(fluo)[1]} futures of {np.shape(fluo)[0]} cells)\nMiddle {q*100:.0f}%')
            axes[i].set_ylim([0,600])
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_{cell_class}_qplot_err_across_diff_noise.png',dpi=300)
    
#%% Q-plot of error  v. time (better for overlay?)

q = 0.5
default_models = default_training_size & default_past_steps & default_horizon & default_h1 & default_h2

for cell_class in ['CcaSR_gillespie_simple_noE', 'CcaSR_gillespie']:
    plt.figure()
                                           
    df_cc = df_meta.loc[default_models & (df_meta['cell_class']==cell_class)].reset_index()
    
    for c in range(len(df_cc)):
        simul_id = df_cc.loc[c,'simul_id']
        camera_sim = df_cc.loc[c,'camera_sim']
        solver = df_cc.loc[c,'solver']
        color = df_simul_config.loc[(df_simul_config['camera_sim']==camera_sim)&(df_simul_config['solver']==solver),'color'].values[0]

        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True)
        x = np.arange(np.shape(fluo)[-1])/12
            
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2), axis=1)
        
        plt.plot(x, np.median(RMSE, axis=0),
                      '.-', color=color, 
                      label=f'camera noise {camera_sim}, solver: {solver}')
        plt.fill_between(
            x,
            np.nanquantile(RMSE, axis=0, q=0.5-q/2),
            np.nanquantile(RMSE, axis=0, q=0.5+q/2),
            color=color,
            alpha=.2,
            )
        
        plt.legend()   
        plt.title(cell_class)
        plt.xlabel('time( h)')
        plt.ylabel(f'RMSE: Middle {q*100:.0f}%')
        plt.ylim([0,450])
        plt.grid(True, 'both','both')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_{cell_class}_qplot_err_across_diff_noise_over_time.png',dpi=300)

#%% Violin plots of error

a = 0.05
q_range = [a, 1-a]
default_models = default_training_size & default_past_steps & default_horizon & default_h1 & default_h2

for cell_class in ['CcaSR_gillespie_simple_noE', 'CcaSR_gillespie']:
    plt.figure(figsize=(13,3))
        
    df_cc = df_meta.loc[default_models & (df_meta['cell_class']==cell_class)].reset_index()
    
    for c in range(len(df_cc)):
        simul_id = df_cc.loc[c,'simul_id']
        camera_sim = df_cc.loc[c,'camera_sim']
        solver = df_cc.loc[c,'solver']
        color = df_simul_config.loc[(df_simul_config['camera_sim']==camera_sim)&(df_simul_config['solver']==solver),'color']
        
        # Find best and worst error, arbitrarily relative to the first cell future
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt((fluo - fluo_pred)**2)
        
        for i in range(np.shape(RMSE)[1]):
            
            RMSE_i = RMSE[:,i]
            quantiles = np.quantile(RMSE_i, q_range)
            y = RMSE_i[(RMSE_i > quantiles[0]) & ((RMSE_i < quantiles[1]))]
            
            violins = plt.violinplot(y, 
                            positions=[(i+c/4)/12,], 
                            widths=1/12/4, 
                            showmeans=False,
                            showmedians=True,
                            showextrema=False)
            
            violins['cmedians'].set_color(color)
            for v in violins['bodies']:
                v.set_facecolor(color)
                v.set_edgecolor(color)
                
    plt.xlabel('time (h)')
    plt.ylabel(f'RMSE\n{int(q_range[0]*100)} to {int(q_range[1]*100)}th percentile')
    plt.title(cell_class)
    plt.tight_layout()
    plt.savefig(f'{fig_path}/fig1_{cell_class}_err_dist_over_time.png', dpi=300)

#%% Plot err v. value

alpha=0.03
markersize=5

default_models = default_training_size & default_past_steps & default_horizon

fig, axes = plt.subplots(1, len(model_list), figsize=(12,3), sharey=True)
for m, model in enumerate(model_list):

    simul_id = df_meta.loc[default_models & (df_meta['cell_class']==model), 'simul_id'].values[-1]
    
    # Find best and worst error, arbitrarily relative to the first cell future
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.sqrt((fluo - fluo_pred)**2)
    
    axes[m].plot(fluo, RMSE, '.', 
                     alpha=alpha,
                     markersize=markersize,
                     color = model_color_dict[model])
    
    axes[m].set_ylim([0, 2000])
    axes[m].set_xlim([0, 4095])     
    axes[m].set_xlabel('simulated fluo')
    axes[m].set_title(model)
    
axes[0].set_ylabel('RMSE')
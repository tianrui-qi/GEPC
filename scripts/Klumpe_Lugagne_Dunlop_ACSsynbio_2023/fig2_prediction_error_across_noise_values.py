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

model_style_dict = {'CcaSR_gillespie_simple': 'o',
                    'CcaSR_gillespie': 's'}
models_list = [key for key in model_style_dict.keys()]

horizon_style_dict = {12: '.', 24: 'x', 48: '_'}
horizon_color_dict = {12: 'r', 24: 'g', 48: 'b'}

past_steps_color_dict = {3: "#ce5451",
                         6: "#bc8b3d",
                         12: "#73af3d",
                         24: "#5da071", 
                         36: "#588acf"}

training_set_color_dict = {1: '#c3ac3d',
                           9: "#8f7f3b",
                           90:"#5bab67",
                           900: "#6a7fcf",
                           9000: "#bc5ead",}

markersize=8

#%% Check training data

plot_list = ['CcaSR_gillespie_simple','CcaSR_gillespie']

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
mu_list = []
sigma_list = []

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
    if 'mu' in model_params.keys():
        mu_list += [model_params['mu']]
    else: mu_list += [np.nan]
    if 'sigma' in model_params.keys():
        sigma_list += [model_params['sigma']]
    else: sigma_list += [np.nan]
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
                        'sigma': sigma_list,
                        'mu': mu_list,
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
default_camera_sim = df_meta['camera_sim']==True
default_solver = df_meta['solver']=='original'


_h1 = 4e-2
_h2 = 1e-3
_h1h2 = _h1 / _h2
default_h1 = np.abs((df_meta['h1'] - _h1) / _h1) < 0.01
default_h2 = np.abs((df_meta['h2'] - _h2) / _h2) < 0.01
default_h1h2 = np.abs(((df_meta['h1'] / df_meta['h2']) - _h1h2) / _h1h2) < 0.01

#%% Plot responses to pure light, to show variability

n_hours = 2
n_cells = 20
alpha=0.4
lw = 1.5

#  # Constant E
cell_class = 'CcaSR_gillespie_simple'
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
# light_sequence = [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24
x = [t/12. for t in range(len(light_sequence)+1)]

# new parameters - resampling should happen at the end
new_params_list = [{'sigma': 2, 'resample_species': True},
                   {'sigma': 8, 'resample_species': True},
                    ]

fig = plt.figure(figsize=(3*len(new_params_list), 4.5))
spec = fig.add_gridspec(3, len(new_params_list))

ylim0, ylim1 = [[0,100],[40-25, 40+25]]

for p, new_params in enumerate(new_params_list):
    
    ax0 = fig.add_subplot(spec[:2,p])
    ax1 = fig.add_subplot(spec[-1,p])
    plt.sca(ax0)
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    
    E_samples = []
    for i in range(n_cells):
    
        # Simulate cell
        cell = dcc.simulations.CcaSR_gillespie_simple()
        cell.update_params(new_params)
        
        # Simulate
        cell.set_light_events(light_sequence)     
        series = cell.run(len(light_sequence)*5, 
                               solver="original", 
                               realizations=1)
        
        ax0.plot(x, [state['F'] for state in series],
                     color='b',
                     alpha=alpha, lw=lw)
        
        for ax in [ax0, ax1]:
            ax.plot(x, [state['E'] for state in series],
                     color='g',
                     alpha=alpha, lw=lw)
            
            ax.set_xticks(np.arange(5*n_hours+1))
            # ax.set_xlabel("time (hours)")
            # ax.set_ylabel("proteins (#)")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    ax0.set_ylim(ylim0)
    ax1.set_ylim(ylim1)
    ax1.set_xlim([0,3])
    # axes[p].set_title(f"mu={cell.params['mu']:.2e}, sigma={cell.params['sigma']:.2e}")
    
plt.tight_layout()
plt.savefig(f'{fig_path}/fig2_{cell_class}_sample_responses.png', dpi=300)

# # Variable E
cell_class = 'CcaSR_gillespie'
new_params_list = [{'h1': (4e-2)/5, 'h2': (1e-3)/5},
                    {'h1': (4e-2)*1, 'h2': (1e-3)*1},
                    {'h1': (4e-2)*5, 'h2': (1e-3)*5},
                    ]

fig = plt.figure(figsize=(3*len(new_params_list), 4.5))
spec = fig.add_gridspec(5, len(new_params_list))

ylim0, ylim1 = [[0,100],[40-25, 40+25]]

for p, new_params in enumerate(new_params_list):
    
    ax0 = fig.add_subplot(spec[:3,p])
    ax1 = fig.add_subplot(spec[-2:,p])
    plt.sca(ax0)
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    
    # Simulate cells
    for n in range(n_cells):
        cell = dcc.simulations.CcaSR_gillespie()
        cell.update_params(new_params)
        cell.set_light_events(light_sequence)     
        series = cell.run(len(light_sequence)*5, 
                                solver="original")
        
        ax0.plot(x, [state['F'] for state in series],
                     color='b',
                     alpha=alpha, lw=lw)
        
        for ax in [ax0, ax1]:
            ax0.plot(x, [state['E'] for state in series],
                    '-',
                     color='g',
                     alpha=alpha, lw=lw)
            ax1.plot(x, [state['E'] for state in series],
                    '.-',
                     color='g',
                     markersize=2,
                     alpha=alpha, lw=lw/2)
            
            ax.set_xticks(np.arange(5*n_hours+1))
            # ax.set_xlabel("time (hours)")
            # ax.set_ylabel("proteins (#)")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    ax0.set_ylim(ylim0)
    ax1.set_ylim(ylim1)
    ax1.set_xlim([0,2])
    # ax0.set_title(f"h1={new_params['h1']:.2e}, h2={new_params['h2']:.2e}")
    
plt.tight_layout()
plt.savefig(f'{fig_path}/fig2_{cell_class}_sample_responses.png', dpi=300)


#%% Default model x training set size

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'

# Keep simulations with default past steps, default horizon, and same h1/h2 ratio
simul_slice = default_past_steps & default_horizon & default_h1 & \
            default_h2 & default_camera_sim & default_solver
            
not_short = ['short' not in ts for ts in df_meta['training_sets']]

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class) & not_short].sort_values(by=['training_sets']).reset_index()


alpha = 0.5
n_bins = 100
x_max = 1200
markersize=8

fig, axes = plt.subplots(2,1, figsize=(8,8), sharex=True)

for i in df_past.index:
    simul = df_past.loc[i]
    simul_id = simul['simul_id']
    training_set_folder = simul['training_sets']
    
    if training_set_folder == 'training_set':
        training_set_size = 10_000
    else:
        training_set_size = int(training_set_folder.split('_')[-2])
        
    if training_set_size > 1:
        training_set_size = int(0.9*training_set_size)
        
        
    for j, validated in enumerate([False, True]):
        if validated & (training_set_size==1):
            continue
        
        fluo, fluo_pred = get_fluo_and_pred(simul_id, validated=validated) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
        # Distribution across the prediction
        axes[j].hist(np.mean(RMSE, axis=1), 
                      bins=np.linspace(0, x_max, n_bins+1),
                      color=training_set_color_dict[training_set_size],
                    alpha=alpha,                 
                    label=f'trained on {training_set_size} cells',
                    density=True)
        
        
        
        # x = np.sort(np.mean(RMSE, axis=1))
        # y = np.arange(1,len(x)+1) / len(x)
        # axes[j].plot(x,y, '.',
        #           markersize=markersize,
        #           alpha=alpha,
        #           color=training_set_color_dict[training_set_size],
        #           label=f'trained on {training_set_size} cells')
        
        axes[j].set_title(f'Validated {validated}')
        axes[j].set_xlim([0, x_max])
        axes[j].set_ylabel('Frequency')


axes[0].legend()
axes[1].set_xlabel('RMSE (mean across time)')

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_default_effect_of_training_set_size.png', dpi=300)

#%% Default model x horizon

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_h1 & \
            default_h2 & default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]


fig, axes = plt.subplots(2,1, figsize=(4,6))
alpha = 0.5
n_bins = 100
x_max = 1000

for ho, horizon in enumerate(horizon_vals):
    df_index = (df_past['horizon']==horizon)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]

    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    # At final time point
    axes[0].hist(RMSE[:,horizon-1],
                 bins=np.linspace(0, x_max, n_bins+1),
                 color=horizon_color_dict[horizon],
                 alpha=alpha,
                 label=f'horizon={horizon}',
                 density=True)
    
    # Distribution at a particular point in time
    axes[1].hist(RMSE[:,11], 
                 bins=np.linspace(0, x_max, n_bins+1),
               color=horizon_color_dict[horizon],
               alpha=alpha,                 
               label=f'horizon={horizon}',
               density=True)

axes[0].set_xlabel('RMSE at final prediction time point')
axes[1].set_xlabel('RMSE after 1h of prediction')
axes[1].legend()
for i in range(2):
    axes[i].set_ylabel('Frequency')
    
plt.tight_layout()

plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_default_effect_of_horizon.png', dpi=300)

#%% Default model x past steps

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1 & \
            default_h2 & default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]


alpha = 0.5
n_bins = 100
x_max = 2000

plt.figure(figsize=(8,4))
for p, past_steps in enumerate(past_steps_vals):
    df_index = (df_past['past_steps']==past_steps)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]

    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
    # Distribution across the prediction
    plt.hist(np.mean(RMSE, axis=1), 
                 bins=np.linspace(0, x_max, n_bins+1),
               color=past_steps_color_dict[past_steps],
               alpha=alpha,                 
               label=f'past steps={past_steps}',
               density=True)

plt.xlabel('RMSE (mean over time)')
plt.ylabel('Frequency')
plt.legend()

plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_default_effect_of_past_steps.png', dpi=300)

#%% Gillespie -> evaluation sets

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]

h1_highlight = 4e-2
h1_plot = [8e-3, 2e-1]

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_h1h2 & \
            default_camera_sim & default_solver & default_horizon

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

alpha = 0.2
n_bins = 50
x_max = 4095

timepoint_list = [0,-1]
xlabel_list = ['RMSE (time 0)', 'RMSE (endpoint)']

label = ['slow','medium','fast']

for h, h1 in enumerate(h1_vals):
    
    df_index = (df_past['h1']==h1)
    simul_id = df_past.loc[df_index, 'simul_id'].values[0]
    h2 = df_past.loc[df_index, 'h2'].values[0]

    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    
    plt.hist(fluo.ravel(), bins=np.linspace(0,x_max, n_bins+1), 
             alpha=alpha, label=label[h], density=True)
                
plt.legend()
plt.xlabel('fluorescence')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_evaluation_set_distribution.png', dpi=300)

#%% Gillespie x horizon

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

h1_highlight = 4e-2
h1_plot = [8e-3, 2e-1]

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(2, len(h1_plot), figsize=(8,6), sharey=True)
alpha = 0.2
n_bins = 40
x_max = 800

timepoint_list = [11,-1]
xlabel_list = ['RMSE (1h)', 'RMSE (endpoint)']

for h, h1 in enumerate(h1_vals):
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['h1']==h1)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
        # For horizon, plot endpoint prediction
        # Highlighted parameter set as outline
        if h1==h1_highlight:
            for i in range(len(h1_plot)):
                for k, timepoint in enumerate(timepoint_list):
                    axes[k,i].hist(RMSE[:,timepoint], 
                                  bins=np.linspace(0, x_max, n_bins+1),
                                  color=horizon_color_dict[horizon],
                                  alpha=alpha,
                                label=f'horizon={horizon}',
                                density=True)
        # Other as distribution
        else: 
            for k, timepoint in enumerate(timepoint_list):
                ax = axes[k,h1_plot.index(h1)]
                ax.hist(RMSE[:,timepoint], 
                              bins=np.linspace(0, x_max, n_bins+1),
                              color=horizon_color_dict[horizon],
                              histtype='step',
                            label=f'horizon={horizon}',
                            density=True)
                
                ax.set_title(f'h1={h1:.2e}, h2={h2:.2e}')
                ax.set_xlabel(xlabel_list[k])
                
        
# axes[0].set_ylabel('Frequency')
axes[0,-1].legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_horizon.png', dpi=300)


#%% Gillespie x horizon x time 

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(h1_vals), figsize=(10,3), sharey=True)
alpha=0.2
q = 0.5
# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['h1']==h1)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[h].plot(t, np.median(RMSE, axis=[0]), '.-', 
                     markersize=markersize,
                     color = horizon_color_dict[horizon],
                     label=f'horizon={horizon}')
        axes[h].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=horizon_color_dict[horizon],
            alpha=alpha,
            )
    
    axes[h].set_title(f'h1={h1:.2e}, h2={h2:.2e}')
    # axes[h].set_xlabel('time (h)')
    # axes[h].set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    axes[h].grid(True, "both", "both")
axes[0].legend()
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_horizon.png', dpi=300)

#%% Gillespie x past steps

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [3, 6, 12, 24, 36]

h1_highlight = 4e-2
h1_plot = [8e-3, 2e-1]
color = ['r','k','b']

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(2, len(past_steps_vals), 
                         figsize=(3*len(past_steps_vals),4),
                         sharey=False)
alpha = 0.25
n_bins = 100
x_max_list = [4000, 1500, 1500, 1500, 1500]

for h, h1 in enumerate(h1_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
        
        x_max = x_max_list[p]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
    
        # For horizon, plot endpoint prediction
        # Highlighted parameter set as outline
        if h1==h1_highlight:
            for i in range(len(h1_plot)):
                # Time 0 error
                axes[0,p].hist(RMSE[:,0], 
                              bins=np.linspace(0, x_max, n_bins+1),
                              color=color[h],
                              histtype='step',
                            label=f'past steps={past_steps}',
                            density=True)
                # T final error
                axes[1,p].hist(RMSE[:,-1], 
                              bins=np.linspace(0, x_max, n_bins+1),
                              color=color[h],
                              histtype='step',
                            label=f'past steps={past_steps}',
                            density=True)
        # Other as distribution
        else:   
            # Time 0 error
            axes[0,p].hist(RMSE[:,0], 
                          bins=np.linspace(0, x_max, n_bins+1),
                              color=color[h],
                          alpha=alpha,
                        label=f'past steps={past_steps}',
                        density=True)
            # T final error
            axes[1,p].hist(RMSE[:,-1], 
                          bins=np.linspace(0, x_max, n_bins+1),
                              color=color[h],
                          alpha=alpha,
                        label=f'past steps={past_steps}',
                        density=True)
       
        axes[0,p].set_title(f'h1={h1:.2e}, h2={h2:.2e}')
        axes[0,p].set_xlabel('RMSE (time 0)')
        axes[1,p].set_xlabel('RMSE (end of prediction)')
                
        
# axes[0].set_ylabel('Frequency')
# axes[-1].legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_past_steps.png', dpi=300)

#%% Gillespie x past steps (no hist)

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [3, 6, 12, 24, 36]

h1_highlight = 4e-2
h1_plot = [8e-3, 2e-1]
color = ['r','k','b']

# Keep simulations with default training size, default past steps, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

title_list = ['RMSE (mean over time)', 
              'RMSE (first timepoint)', 
              'RMSE (final timepoint)']
# fig, axes = plt.subplots(1, 2, 
#                          figsize=(8,3),
#                          sharey=False)
markersize=5
lw=0.75
fig, axes = plt.subplots(3,1, figsize=(2,6), sharex=False)
for h, h1 in enumerate(h1_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
            
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        
        x = p + (h-1)/4
        RMSE_t_avg = np.mean(RMSE,axis=1)
        axes[0].plot(x, np.median(RMSE_t_avg),
                    '.', markersize=markersize,
                    color=color[h])
        axes[0].plot([x,x], [np.nanquantile(RMSE_t_avg, q=0.25),
                         np.nanquantile(RMSE_t_avg, q=0.75)],
                 '-', lw=lw, color=color[h])
        
        axes[1].plot(x, np.median(RMSE[:,0]),
                    '.', markersize=markersize,
                    color=color[h])
        axes[1].plot([x,x], [np.nanquantile(RMSE[:,0], q=0.25),
                         np.nanquantile(RMSE[:,0], q=0.75)],
                 '-', lw=lw, color=color[h])
        
        axes[2].plot(x, np.median(RMSE[:,-1]),
                    '.', markersize=markersize,
                    color=color[h])
        axes[2].plot([x,x], [np.nanquantile(RMSE[:,-1], q=0.25),
                         np.nanquantile(RMSE[:,-1], q=0.75)],
                 '-', lw=lw, color=color[h])
for i in range(3):
    axes[i].set_yscale('log')
    axes[i].set_xticks(np.arange(len(past_steps_vals)))
    axes[i].set_xticklabels(past_steps_vals)
    axes[i].set_ylim([90,5000])
    axes[i].set_title(title_list[i])
    
axes[-1].set_xlabel('Past timepoints')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_past_steps_summary.png', dpi=300)



#%% Gillespie x past steps x time

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [6, 12, 24]

# Keep simulations with default training size, default horizon, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(h1_vals), figsize=(10,3), sharey=True)
alpha=0.2
q = 0.5
# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[h].plot(t, np.median(RMSE, axis=[0]), '.-', 
                     markersize=markersize,
                     color = past_steps_color_dict[past_steps],
                     label=f'past steps={past_steps}')
        axes[h].fill_between(
            t,
            np.nanquantile(RMSE, axis=[0], q=0.5-q/2),
            np.nanquantile(RMSE, axis=[0], q=0.5+q/2),
            color=past_steps_color_dict[past_steps],
            alpha=alpha,
            )
    
    # axes[h].set_title(f'h1={h1:.2e}, h2={h2:.2e}')
    # axes[h].set_xlabel('time (h)')
    # axes[h].set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    axes[h].grid(True, "both", "both")
axes[0].legend()
plt.tight_layout()
# plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_past_steps_all.png', dpi=600)
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_past_steps.png', dpi=600)

#%% [by cell] Gillespie x past steps x time

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [3, 6, 12, 24]

# Keep simulations with default training size, default horizon, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(len(h1_vals), len(past_steps_vals), 
                         figsize=(3*len(past_steps_vals),3*len(h1_vals)), 
                         sharey=False)
alpha=0.02
# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
        t = [x/12. for x in range(np.shape(RMSE)[-1])]
        axes[h,p].plot(t, RMSE.T, '.-',
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
    
        axes[h,p].set_title(f'h1={h1}, h2={h2}')
        # axes[k].legend()
        axes[h,p].set_xlabel('time (h)')
        axes[h,p].grid(True, 'both', 'both')
axes[0,0].set_ylabel(f'Median error\n{np.shape(fluo)[0]} cells')
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_past_steps_over_time_individual_cells.png', dpi=300)


#%% ((In progress) Plot sample predictions

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]
past_steps_vals = [6, 12, 24, 36]

# Keep simulations with default training size, default horizon, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

percentile = [250, 500, 750]
lw = 0.75

# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    fig, axes = plt.subplots(len(percentile), len(past_steps_vals),
                              figsize=(3, 2*len(percentile)),
                              sharex=True, sharey=True)
    
    for p, past_steps in enumerate(past_steps_vals):
        
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
        color = past_steps_color_dict[past_steps]
    
        # Find best and worst error
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        RMSE_sum_across_time = np.sum(np.sqrt((fluo - fluo_pred)**2), axis=1)
        sorted_error = np.argsort(RMSE_sum_across_time)

        plot_list = [sorted_error[i] for i in percentile]

        stims, past_fluo, futures_fluo = get_eval_data(simul_id)
        model_params, training_params = get_params(simul_id)

        plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
        cutoff = past_fluo.shape[1]

        
        for i, pl in enumerate(plot_list):
            ax = axes[i,p]
                
            # Stimulations
            plt.sca(ax)
            dcc.utilities.OptoPlotBackground(
                stims[pl,cutoff-plot_past:cutoff+training_params["horizon"]],
                x=np.arange(-plot_past, training_params["horizon"])/12,
                ymax = 4095
                )
            
            # Past
            axes[i].plot(np.arange(-plot_past, 0)/12, 
                          past_fluo[pl, -plot_past:],
                          color, lw=3*lw)
            
            # Future
            axes[i].plot(np.arange(0, training_params["horizon"])/12, 
                      futures_fluo[pl, :, :training_params['horizon']].T, 
                      color=color, alpha=0.01)
            
            # Prediction
            axes[i].plot(np.arange(0, training_params["horizon"])/12, 
                      fluo_pred[pl],
                      color='k', lw=3*lw)
            
            # Prediction starts line
            axes[i].plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
            
            # Limits and labels
            axes[i].set_ylim([0,4095])
            axes[i].set_yticklabels([])
            axes[i].set_xticklabels([])
            # plt.title(f'Cell {c}')
            # axes[i].set_xlabel("time (h)")
            # axes[i].set_ylabel("Fluorescence (a.u.)")
            # axes[i].set_title(f'{100 * percentile[i] / len(sorted_error):.0f}th percentile')
        axes[-1].set_xlim([-plot_past/12, training_params["horizon"]/12])
        # plt.suptitle(f'{cell_class}, camera_sim {camera_sim}, solver:{solver}')
        plt.tight_layout()
        plt.savefig(fig_path+f'/fig2_variable_E_h1_vals_{h}_percentiles.png', dpi=300)


#%% Plot select predictions

cell_class = 'CcaSR_gillespie_simple'
simul_slice = default_past_steps & default_horizon
df_past = df_meta.loc[simul_slice&(df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index(drop=True)

training_style_dict = {100: '_', 1000: 'x', 10000: '.'}
sigma_list = [x for x in np.unique(df_past['sigma'])]
t_list = [x for x in training_style_dict.keys()]

q=0.5
alpha=0.02
markersize=4
for i in range(len(df_past)):
    sigma = df_past.loc[i,'sigma']
    training_folder = df_past.loc[i,'training_sets']
    if (training_folder.split('_')[-1]==1) or (training_folder.split('_')[-1]==2):
        continue
    if training_folder == 'training_set':
        training_set_size = 10000
    else:
        training_set_size = int(training_folder.split('_')[-2])
    simul_id = df_past.loc[i, 'simul_id']
    
    # Find best and worst error, arbitrarily relative to the first cell future
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 
    
    # Median across futures, but then the sum across all time
    RMSE_sum_across_time = np.sum(np.median(np.sqrt((fluo - fluo_pred)**2), axis=1), axis=1)
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
                 futures_fluo[pl, :, :training_params['horizon']].T, 
                 color = 0.5*np.ones((3,)), alpha=alpha)
        
        # Prediction
        axes[i].plot(np.arange(0, training_params["horizon"])/12, 
                 fluo_pred[pl, 0],
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
    plt.suptitle(f'sigma={sigma}, trained on {training_set_size}cells')
    plt.tight_layout()
    plt.savefig(fig_path+f'/fig2_{cell_class}_sigma{sigma}_ts{training_set_size}_predictions_percentiles.png', dpi=300)

#%% Gillespie simple x training_set_size

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie_simple'
sigma_vals = np.unique(sigma_list)
sigma_vals = sigma_vals[~np.isnan(sigma_vals)]

# Keep simulations with default past steps, default horizon, and same h1/h2 ratio
simul_slice = default_past_steps & default_horizon & default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index()

fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True)
alpha = 0.3
n_bins = 100
x_max = 1500

for s, sigma in enumerate(sigma_vals):
    
    df_s = df_past.loc[df_past['sigma']==sigma]
    
    for i in df_s.index:
        for j, validated in enumerate([False, True]):
            
            if validated & (training_set_size==1):
                continue

            simul = df_past.loc[i]
            simul_id = simul['simul_id']
            training_set_folder = simul['training_sets']
            
            if training_set_folder == 'training_set':
                training_set_size = 10_000
            else:
                training_set_size = int(training_set_folder.split('_')[-2])
                
            if training_set_size > 1:
                training_set_size = int(0.9*training_set_size)
        
            fluo, fluo_pred = get_fluo_and_pred(simul_id, validated=validated) 
            RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
            # # Distribution across the prediction
            # axes[s,j].hist(np.sum(RMSE, axis=1), 
            #               bins=np.linspace(0, x_max, n_bins+1),
            #               color=training_set_color_dict[training_set_size],
            #             alpha=alpha,                 
            #             label=f'trained on {training_set_size} cells',
            #             density=True)
            x = np.sort(np.mean(RMSE, axis=1))
            y = np.arange(1,len(x)+1) / len(x)
            axes[s,j].plot(x,y, '.', markersize=3, 
                          color=training_set_color_dict[training_set_size],
                          label=f'trained on {training_set_size}cell')
        axes[s,j].set_title(f'sigma={sigma} (validated {validated})')
        axes[s,j].set_xlabel('Total RMSE')
        axes[s,j].set_xlim([0,x_max])

# axes[0].set_ylabel('Frequency')
# axes[1].legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png', dpi=300)

#%% [summary] Gillespie simple x training_set_size

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie_simple'
sigma_vals = np.unique(sigma_list)
sigma_vals = sigma_vals[~np.isnan(sigma_vals)]
ts_list = [key for key in training_set_color_dict.keys()]

# Keep simulations with default past steps, default horizon, and same h1/h2 ratio
simul_slice = default_past_steps & default_horizon & default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index()

fig, axes = plt.subplots(2,2, figsize=(6,6), sharex=True)
alpha = 0.3
n_bins = 100
y_max = 1000
markersize=10
lw=1

for s, sigma in enumerate(sigma_vals):
    
    df_s = df_past.loc[df_past['sigma']==sigma]
    
    for i in df_s.index:
        for j, validated in enumerate([False, True]):
            
            if validated & (training_set_size==1):
                continue

            simul = df_past.loc[i]
            simul_id = simul['simul_id']
            training_set_folder = simul['training_sets']
            
            if training_set_folder == 'training_set':
                training_set_size = 10_000
                nudge = 0
            else:
                training_set_size = int(training_set_folder.split('_')[-2])
                nudge = int(training_set_folder.split('_')[-1])-1
                
            if training_set_size > 1:
                training_set_size = int(0.9*training_set_size)
        
            fluo, fluo_pred = get_fluo_and_pred(simul_id, validated=validated) 
            RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
            # # Distribution across the prediction
            # axes[s,j].hist(np.sum(RMSE, axis=1), 
            #               bins=np.linspace(0, x_max, n_bins+1),
            #               color=training_set_color_dict[training_set_size],
            #             alpha=alpha,                 
            #             label=f'trained on {training_set_size} cells',
            #             density=True)
            data = np.sort(np.mean(RMSE, axis=1))
            x = ts_list.index(training_set_size) + (nudge / 5)
            axes[j,s].plot(x, np.median(data), '.', 
                           markersize=markersize,
                           color=training_set_color_dict[training_set_size],
                           label=f'trained on {training_set_size}cell')
            axes[j,s].plot([x,x], 
                           [np.quantile(data, 0.25),np.quantile(data, 0.75)], 
                           '-', lw=lw,
                           color=training_set_color_dict[training_set_size],
                           label=f'trained on {training_set_size}cell')
            
            axes[j,s].set_title(f'sigma={sigma} (validated {validated})')
            axes[j,s].set_xlabel('log10(Training set size)')
            axes[j,s].set_ylim([0,y_max])
            axes[j,s].set_ylabel('RMSE (mean across time)')

# axes[0].set_ylabel('Frequency')
# axes[1].legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png', dpi=300)


#%% Gillespie simple x training_set_size [cp sigma side by side]

# Which cell class, and horizon values to plot
cell_class = 'CcaSR_gillespie_simple'
sigma_vals = np.unique(sigma_list)
sigma_vals = sigma_vals[~np.isnan(sigma_vals)]
sigma_colors = ["#a26dba","#71a65a"]
ts_list = [900, 90, 9]

# Keep simulations with default past steps, default horizon, and same h1/h2 ratio
simul_slice = default_past_steps & default_horizon & default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index()

fig, axes = plt.subplots(1,len(ts_list), figsize=(3.5*len(ts_list),3), sharey=True)
alpha = 0.2
n_bins = 100
x_max = 30000

for s, sigma in enumerate(sigma_vals):
    
    df_s = df_past.loc[df_past['sigma']==sigma]
    
    for i in df_s.index:

        simul = df_past.loc[i]
        simul_id = simul['simul_id']
        training_set_folder = simul['training_sets']
                
        # Get training set size
        if training_set_folder == 'training_set':
            training_set_size = 10_000
        else:
            training_set_size = int(training_set_folder.split('_')[-2])
            
        # Correct for test/eval split
        if training_set_size > 1:
            training_set_size = int(0.9*training_set_size)
        else:
            continue
    
        # which axes
        if training_set_size in ts_list:
            ax = axes[ts_list.index(training_set_size)]
        
            fluo, fluo_pred = get_fluo_and_pred(simul_id) 
            RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
            
            # # Distribution across the prediction
            ax.hist(np.sum(RMSE, axis=1), 
                          bins=np.linspace(0, x_max, n_bins+1),
                          color=sigma_colors[s],
                        alpha=alpha,                 
                        label=f'sigma={sigma}',
                        density=True)
            # x = np.sort(np.sum(RMSE, axis=1))
            # y = np.arange(1,len(x)+1) / len(x)
            # axes[s].plot(x,y, '.', markersize=3, 
            #              color=training_set_color_dict[training_set_size],
            #              label=f'trained on {training_set_size}cell')
            ax.set_title(f'Trained on {training_set_size} cells')
            ax.set_xlabel('Total RMSE')
        elif training_set_size == 9_000:
            for t in range(len(ts_list)):
                fluo, fluo_pred = get_fluo_and_pred(simul_id) 
                RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=1))
                
                # # Distribution across the prediction
                axes[t].hist(np.sum(RMSE, axis=1), 
                              bins=np.linspace(0, x_max, n_bins+1),
                              color=sigma_colors[s],
                              histtype='step',
                            label=f'sigma={sigma}',
                            density=True)
            

# axes[0].set_ylabel('Frequency')
axes[-1].legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png', dpi=300)


#%% (Gillespie simple) x training_set_size x time

cell_class = 'CcaSR_gillespie_simple'
simul_slice = default_past_steps & default_horizon
df_past = df_meta.loc[simul_slice&(df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index(drop=True)

sigma_list = [x for x in np.unique(df_past['sigma'])]
t_list = [key for key in training_set_color_dict.keys()]

fig, axes = plt.subplots(1, len(sigma_list), figsize=(8,3), 
                         sharex=True, sharey=True)
q=0.5
alpha=0.2
for i in range(len(df_past)):
    sigma = df_past.loc[i,'sigma']
    training_folder = df_past.loc[i,'training_sets']
    if training_folder == 'training_set':
        training_set_size = 10000
    else:
        training_set_size = int(training_folder.split('_')[-2])
    if (training_folder.split('_')[-1] == '1') or (training_folder.split('_')[-1] == '2'):
        continue
    
    simul_id = df_past.loc[i, 'simul_id']

    
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2,axis=1))
    t = [x/12. for x in range(np.shape(RMSE)[-1])]
    axes[sigma_list.index(sigma)].plot(
                    t, np.median(RMSE, axis=[0,]), '.-',
                    markersize=markersize,
                    label=f'{training_set_size:.0e} cells',
                    color=training_set_color_dict[training_set_size])
    axes[sigma_list.index(sigma)].fill_between(
                    t, 
                    np.nanquantile(RMSE, axis=[0,], q=0.5-q/2), 
                    np.nanquantile(RMSE, axis=[0,], q=0.5+q/2), 
                    color=training_set_color_dict[training_set_size],
                    alpha=alpha)

for j in range(len(sigma_list)):
    ax = axes[j]
    # ax.legend()
    # ax.set_xlabel('time (h)')
    # ax.set_title(f'sigma={sigma_list[j]}')
    # ax.set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    ax.grid(True, "both", "both")
axes[0].legend()
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png', dpi=600)
print(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png')

#%% "Simple" model error binned by fluorescence

cell_class = 'CcaSR_gillespie_simple'
simul_slice = default_past_steps & default_horizon
df_past = df_meta.loc[simul_slice&(df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index(drop=True)

training_style_dict = {100: '_', 1000: 'x', 10000: '.'}
sigma_list = [x for x in np.unique(df_past['sigma'])]
t_list = [x for x in training_style_dict.keys()]
t_color_list = ['#cf327b','r','#cf7332']

fig, axes = plt.subplots(1, len(sigma_list), figsize=(8,3), 
                         sharex=True, sharey=True)
q=0.5
alpha=0.1
markersize=4
n_bins = 20
for i in range(len(df_past)):
    sigma = df_past.loc[i,'sigma']
    training_folder = df_past.loc[i,'training_sets']
    if training_folder == 'training_set':
        training_set_size = 10000
    else:
        training_set_size = int(training_folder.split('_')[-2])
    simul_id = df_past.loc[i, 'simul_id']
    
    ax = axes[sigma_list.index(sigma)]
    
    fluo, fluo_pred = get_fluo_and_pred(simul_id, return_all=True) 

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
            
    ax.plot(bins[:-1], 
                 np.nanmedian(RMSE_bins, axis=[1,]),
                  '.-',color=t_color_list[t_list.index(training_set_size)])
    ax.fill_between(
        bins[:-1],
        np.nanquantile(RMSE_bins, axis=1, q=0.5-q/2),
        np.nanquantile(RMSE_bins, axis=1, q=0.5+q/2),
        color=t_color_list[t_list.index(training_set_size)],
        alpha=.2,
        )

    ax.set_xlabel('fluorescence')
    ax.set_ylabel(f'RMSE ({np.shape(fluo)[1]} futures of {np.shape(fluo)[0]} cells)\nMiddle {q*100:.0f}%')
    ax.set_ylim([0, 1200])
    ax.grid(True, 'both', 'both')
plt.tight_layout()
plt.savefig(f'{fig_path}/fig2_{cell_class}_training_set_size_qplot_err_fluor_bin.png',dpi=300)
    
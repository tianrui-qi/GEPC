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
    
def get_fluo_and_pred(simul_id):
    """
    Given a simul_id (str), return the fluo (arr, (n_cells, len(future)))
    and fluo_pred (arr, (n_cells, len(future))) for each individual evaluation
    cell
    """
        
    # Load predictions
    fluo_pred = get_fluo_pred(simul_id)

    # Load data
    stims, past_fluo, futures_fluo = get_eval_data(simul_id)
    
    # Keep only first realization, and as many time point as in the prediction
    fluo = futures_fluo[:,:,:np.shape(fluo_pred)[1]]
    
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

model_style_dict = {'CcaSR_gillespie_simple': 'o',
                    'CcaSR_gillespie': 's'}
models_list = [key for key in model_style_dict.keys()]

horizon_style_dict = {12: '.', 24: 'x', 48: '_'}
horizon_color_dict = {12: 'r', 24: 'g', 48: 'b'}

past_steps_color_dict = {12: 'r', 24: 'g', 36: 'b'}

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

n_hours = 4
n_cells = 40
random_bit = dcc.utilities.random_stimulations(
                        timepoints=3*n_hours*12,
                        nostim_timepoints=0,
                        total_simulations=7)[0]

light_sequence = [1]*12*n_hours + [0]*12*n_hours + [int(bit) for bit in random_bit]
# light_sequence = [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24 + [1]*24 + [0]*24
x = [t/12. for t in range(len(light_sequence)+1)]

new_params_list = [{'h1': (4e-2)/5, 'h2': (1e-3)/5},
                   {'h1': (4e-2)*1, 'h2': (1e-3)*1},
                   {'h1': (4e-2)*5, 'h2': (1e-3)*5},
                   ]

fig, axes = plt.subplots(1, len(new_params_list), 
                         figsize=(4*len(new_params_list),4))
alpha=0.3

for p, new_params in enumerate(new_params_list):
    
    # Simulate cell
    cell = dcc.simulations.CcaSR_gillespie()
    cell.update_params(new_params)
    cell.set_light_events(light_sequence)     
    series_list = cell.run(len(light_sequence)*5, 
                           solver="original", 
                           realizations=n_cells)
    
    # Plot results
    plt.sca(axes[p])
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    # final_F = [series[-1]['F'] for series in series_list]
    
    for series in series_list:
    
        axes[p].plot(x, [state['E'] for state in series],
                      color='g',
                      alpha=alpha, lw=.5)
        axes[p].plot(x, [state['F'] for state in series],
                      color='b',
                      alpha=alpha, lw=.5)
    axes[p].set_xlabel("time (hours)")
    axes[p].set_ylabel("proteins (#)")
    axes[p].set_ylim(0,150)
    axes[p].set_title(f"h1={new_params['h1']:.2e}, h2={new_params['h2']:.2e}")
    
plt.tight_layout()
# plt.savefig(f'{fig_path}/fig2_varh1h2_sample_responses.png', dpi=300)

#%% FFT of pure ON dynamics

import scipy.fft

cell_class = 'CcaSR_gillespie'
colors = ['r','g','b']
datasets_list = np.unique(df_meta.loc[df_meta['cell_class']==cell_class, 'datasets_folder'])

light_sequence = [1]*15*12
n_cells=10

for d, dataset in enumerate(datasets_list):
    # # Drop first 3 hours of darkness
    # fluo = np.load(dataset+'/training_set/fluo1.npy')[:,36:]
    
    model_json = glob.glob(dataset+'/model_parameters.json')[0]
    with open(model_json, 'r') as f:
        model_params = json.load(f)
    
    refcell = dcc.simulations.CcaSR_gillespie()
    refcell.update_params(model_params)
    refcell.set_light_events(light_sequence)
    states = refcell.run(len(light_sequence)*5, 
                         realizations=n_cells)
    
    N = len(light_sequence) #np.shape(fluo)[1]
    T = 1/12
    xf = scipy.fft.fftfreq(N,T)[:N//2]
    
    for i in range(n_cells):
        # y = fluo[i]
        y = [state['F'] for state in states[i]]
        yf = scipy.fft.fft(y)[:N//2]
        
        plt.plot(xf, 2/N * np.abs(yf), alpha=0.2, color=colors[d])
        
# plt.xlim([0,1])
plt.yscale('log')


#%% Plot MEAN error as function of horizon 

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
horizon_vals = np.sort(np.unique(horizon_list))[::-1]

# Keep simulations with default training size, default past, and same h1/h2 ratio
simul_slice = default_training_size & default_past_steps & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(h1_vals), figsize=(10,3), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    for ho, horizon in enumerate(horizon_vals):
        df_index = (df_past['h1']==h1)&(df_past['horizon']==horizon)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id)
        fluo_pred = np.repeat(fluo_pred[:, np.newaxis], fluo.shape[0], axis=1)
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=(0,1)))
        t = [x/12. for x in range(RMSE.shape[0])]
        axes[h].plot(t, RMSE, '.', 
                     color = horizon_color_dict[horizon],
                     alpha=alpha,
                     label=f'horizon={horizon}')
    
    axes[h].set_title(f'h1={h1:.2e}, h2={h2:.2e}')
    axes[h].legend()
    axes[h].set_xlabel('time (h)')
    axes[h].set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    axes[h].grid(True, "both", "both")
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_horizon.png', dpi=600)

#%% Plot MEAN error as function of past steps 

# Which cell class, h1, and horizon values to plot
cell_class = 'CcaSR_gillespie'
h1_vals = np.sort(np.unique(h1_list))
h1_vals = h1_vals[~np.isnan(h1_vals)]
past_steps_vals = np.sort(np.unique(past_steps_list))[::-1]

# Keep simulations with default training size, default horizon, and same h1/h2 ratio
simul_slice = default_training_size & default_horizon & default_h1h2 & \
            default_camera_sim & default_solver

df_past = df_meta.loc[simul_slice & (df_meta['cell_class']==cell_class)]

fig, axes = plt.subplots(1, len(h1_vals), figsize=(10,3), sharey=True)
alpha=0.5
# Rather than plotting everything, just plot one replicate of each trained model
for h, h1 in enumerate(h1_vals):
    
    for p, past_steps in enumerate(past_steps_vals):
        df_index = (df_past['h1']==h1)&(df_past['past_steps']==past_steps)
        simul_id = df_past.loc[df_index, 'simul_id'].values[0]
        h2 = df_past.loc[df_index, 'h2'].values[0]
    
        fluo, fluo_pred = get_fluo_and_pred(simul_id) 
        fluo_pred = np.repeat(fluo_pred[:, np.newaxis], fluo.shape[0], axis=1)
        RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=(0,1)))
        # RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
        t = [x/12. for x in range(len(RMSE))]
        axes[h].plot(t, RMSE, '.', 
                     color = past_steps_color_dict[past_steps],
                     alpha=alpha,
                     label=f'past steps={past_steps}')
    
    axes[h].set_title(f'h1={h1:.2e}, h2={h2:.2e}')
    axes[h].legend()
    axes[h].set_xlabel('time (h)')
    axes[h].set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    axes[h].grid(True, "both", "both")
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_past_steps.png', dpi=600)



#%% Show variability of Gillespie_simple training data

n_hours = 4
n_cells = 200
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

fig, axes = plt.subplots(len(new_params_list), 2, 
                         figsize=(8, 3*len(new_params_list)))
alpha=0.1

for p, new_params in enumerate(new_params_list):
    
    plt.sca(axes[p,1])
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    
    E_samples = []
    for i in range(n_cells):
    
        # Simulate cell
        cell = dcc.simulations.CcaSR_gillespie_simple()
        
        # Update parameters (including new E sample)
        # new_E = round(np.random.normal(loc=cell.params['mu'],
        #                             scale=new_params['sigma'])), # steady state E
        # new_params.update({'E': new_E[0]})
        cell.update_params(new_params)
        
        # Simulate
        cell.set_light_events(light_sequence)     
        series = cell.run(len(light_sequence)*5, 
                               solver="original", 
                               realizations=1)
        E_samples += [series[0]['E']]
        # Plot results
    
        # axes[p,0].plot(x, [state['E'] for state in series],
        #              color='g',
        #              alpha=alpha*2, lw=.5)
        axes[p,1].plot(x, [state['F'] for state in series],
                     color='b',
                     alpha=alpha, lw=.5)
        axes[p,1].plot(x, [state['E'] for state in series],
                     color='g',
                     alpha=alpha, lw=.5)
    
    axes[p,1].set_xlabel("time (hours)")
    axes[p,1].set_ylabel("proteins (#)")

    axes[p,0].hist(E_samples, density=True, color='g')
    axes[p,0].set_xlabel("E")
    axes[p,0].set_ylabel("frequency")
    axes[p,0].set_xlim([20,60])
    
    # axes[p,0].set_ylims([30,50])
    axes[p,1].set_ylim([0,100])
    axes[p,1].set_title(f"mu={cell.params['mu']:.2e}, sigma={cell.params['sigma']:.2e}")
    
plt.tight_layout()
plt.savefig(f'{fig_path}/fig2_{cell_class}_example_responses.png', dpi=300)

#%% MEAN error (Gillespie simple) x training_set_size

cell_class = 'CcaSR_gillespie_simple'
simul_slice = default_past_steps & default_horizon
df_past = df_meta.loc[simul_slice&(df_meta['cell_class']==cell_class)].sort_values(by=['training_sets']).reset_index(drop=True)

training_style_dict = {100: '_', 1000: 'x', 10000: '.'}
sigma_list = [x for x in np.unique(df_past['sigma'])]

fig, axes = plt.subplots(len(sigma_list),1,figsize=(5,8), 
                         sharex=True)
markersize=4
for i in range(len(df_past)):
    sigma = df_past.loc[i,'sigma']
    training_folder = df_past.loc[i,'training_sets']
    if training_folder == 'training_set':
        training_set_size = 10000
    else:
        training_set_size = int(training_folder.split('_')[-2])
    simul_id = df_past.loc[i, 'simul_id']

    
    fluo, fluo_pred = get_fluo_and_pred(simul_id) 
    fluo_pred = np.repeat(fluo_pred[:, np.newaxis], fluo.shape[0], axis=1)
    RMSE = np.sqrt(np.mean((fluo - fluo_pred)**2, axis=(0,1)))
    # RMSE = np.median(np.sqrt((fluo - fluo_pred)**2), axis=0)
    t = [x/12. for x in range(len(RMSE))]
    axes[sigma_list.index(sigma)].plot(t, RMSE, 
                    training_style_dict[training_set_size], 
                    markersize=markersize,
                    label=f'trained on {training_set_size}cells',
                    color='k')

for i in range(len(sigma_list)):
    axes[i].set_xlabel('time (h)')
    axes[i].set_title(f'sigma={sigma_list[i]}')
    axes[i].set_ylim([150,450])
    axes[i].set_ylabel(f'RMSE\n{np.shape(fluo)[0]} cells')
    axes[i].grid(True, "both", "both")
plt.tight_layout()
plt.savefig(dcc_repo_path+f'/assets/figures/fig2_{cell_class}_effect_of_training_set_size.png', dpi=600)

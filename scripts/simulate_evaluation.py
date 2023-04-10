# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:50:39 2023

@author: jeanbaptiste
"""
import os
import json
import sys
import copy
import time
import uuid
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

username='hklumpe'
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

# Load default params:
params = copy.deepcopy(dcc.config.defaults)

# If path to parameters JSON file passed as argument, load additional
# parameters stored there:
if len(sys.argv) > 1:
    with open(sys.argv[1], "r") as f:
        params.update(json.load(f))
        
else:
    params['model'] = 'model_besteval.hdf5'
    # params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_simulated_{uuid.uuid4()}"
    params['eval_save_path'] = '/evaluation/'

save_path = params["models_folder"]+ "/" + params["save_folder"]
eval_save_path = save_path + params['eval_save_path']

#%% Evaluate predictor:
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Evaluating prediction network\n" + "="*100)
os.makedirs(eval_save_path, exist_ok = True)

# Load data from evaluation sets:
past_fluo, futures_fluo, stims = [], [], []
for f in params["eval_sets"]:
    eval_set_path = params["datasets_folder"] + "/" + f
    past_fluo += [np.load(eval_set_path + "/past_fluo.npy")]
    futures_fluo += [np.load(eval_set_path + "/futures_fluo.npy")]
    stims += [np.load(eval_set_path + "/stims.npy")]
past_fluo = np.concatenate(past_fluo, axis=0)
futures_fluo = np.concatenate(futures_fluo, axis=0)
stims = np.concatenate(stims, axis=0)

# Reformat to what the model expects:
cutoff = past_fluo.shape[1]
x = (
     np.stack([past_fluo/4095, stims[:,:cutoff]], axis = -1), 
     stims[:,cutoff:cutoff+params["horizon"]],
     )

# Load model
network = tf.keras.models.load_model(f"{save_path}/{params['model']}")

# Predict:
yhat = network.predict(x, verbose = True)

# Save predictions:
np.save(eval_save_path + "/predictions.npy", yhat)

#Plot results and write them to disk:
plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
for c in range(10): #range(stims.shape[0]):
    plt.figure(1)
    dcc.utilities.OptoPlotBackground(
        stims[c,cutoff-plot_past:cutoff+params["horizon"]],
        x=np.arange(-plot_past, params["horizon"])/12,
        ymax = 4095
        )
    plt.plot(np.arange(-plot_past, 0)/12, past_fluo[c, -plot_past:],"k")
    dcc.utilities.plotq(
        futures_fluo[c,:,:params["horizon"]],
        x = np.arange(0, params["horizon"])/12,
        color="k"
        )
    plt.plot(np.arange(0, params["horizon"])/12, yhat[c]*4095, "b")
    plt.plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
    plt.xlabel("time (h)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlim([-plot_past/12,params["horizon"]/12])
    plt.ylim([0,4095])
    plt.savefig(eval_save_path+f'/cell_{c:06d}.png',dpi=300)
    plt.savefig(eval_save_path+f'/cell_{c:06d}.svg',dpi=300)
    plt.cla()
    
#%% Without cluster: simulation information

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

#%% Without cluster: re-evaluate without best model

model_dir_list = df_meta.loc[df_meta['training_sets']!='training_set','simul_id'].values
default_past_steps = (df_meta['past_steps']==36)
default_horizon = (df_meta['horizon']==24)
model_dir_list = df_meta.loc[(default_past_steps)&(default_horizon),'simul_id'].values
    
for m, model_dir in enumerate(np.sort(model_dir_list)):
    
    # # Get information about this model
    model_dir = dcc_repo_path + '/assets/models/' + model_dir 
    
    with open(model_dir+'/training_parameters.json', 'r') as f:
        training_params = json.load(f)
    
    params['models_folder'] = model_dir.split('models')[0] + '/models/'
    params['save_folder'] = model_dir.split('models')[1]
    params['model'] = 'model.hdf5'
    params['eval_save_path'] = '/evaluation_no_validation/'
    
    feature_list = ['training_sets', 
                    'datasets_folder',
                    'eval_sets',
                    'features',
                    'cell_class']
    for feature in feature_list:
        params[feature] = training_params[feature]
        
    # # Run the evaluation
    save_path = params["models_folder"]+ "/" + params["save_folder"]
    eval_save_path = save_path + params['eval_save_path']
        
    print(f"{m} of {len(model_dir_list)}: {time.strftime('%Y-%m-%d_%H-%M-%S')}\n{model_dir}")
    # print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Evaluating prediction network\n" + "="*100)
    os.makedirs(eval_save_path, exist_ok = True)
    
    # Load data from evaluation sets:
    past_fluo, futures_fluo, stims = [], [], []
    for f in params["eval_sets"]:
        eval_set_path = params["datasets_folder"] + "/" + f
        past_fluo += [np.load(eval_set_path + "/past_fluo.npy")]
        futures_fluo += [np.load(eval_set_path + "/futures_fluo.npy")]
        stims += [np.load(eval_set_path + "/stims.npy")]
    past_fluo = np.concatenate(past_fluo, axis=0)
    futures_fluo = np.concatenate(futures_fluo, axis=0)
    stims = np.concatenate(stims, axis=0)
    
    # Reformat to what the model expects:
    cutoff = past_fluo.shape[1]
    x = (
         np.stack([past_fluo/4095, stims[:,:cutoff]], axis = -1), 
         stims[:,cutoff:cutoff+params["horizon"]],
         )
    
    # Load model
    network = tf.keras.models.load_model(f"{save_path}/{params['model']}")
    
    # Predict:
    yhat = network.predict(x, verbose = True)
    
    # Save predictions:
    np.save(eval_save_path + "/predictions.npy", yhat)
    
    # # # Plot results and write them to disk:
    # plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
    # for c in range(10): #range(stims.shape[0]):
    #     plt.figure(1)
    #     dcc.utilities.OptoPlotBackground(
    #         stims[c,cutoff-plot_past:cutoff+params["horizon"]],
    #         x=np.arange(-plot_past, params["horizon"])/12,
    #         ymax = 4095
    #         )
    #     plt.plot(np.arange(-plot_past, 0)/12, past_fluo[c, -plot_past:],"k")
    #     dcc.utilities.plotq(
    #         futures_fluo[c,:,:params["horizon"]],
    #         x = np.arange(0, params["horizon"])/12,
    #         color="k"
    #         )
    #     plt.plot(np.arange(0, params["horizon"])/12, yhat[c]*4095, "b")
    #     plt.plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
    #     plt.xlabel("time (h)")
    #     plt.ylabel("Fluorescence (a.u.)")
    #     plt.xlim([-plot_past/12,params["horizon"]/12])
    #     plt.ylim([0,4095])
    #     plt.savefig(eval_save_path+f'/cell_{c:06d}.png',dpi=300)
    #     plt.savefig(eval_save_path+f'/cell_{c:06d}.svg',dpi=300)
    #     plt.cla()

#%% Old: generate evaluation set AND evaluate
# #%% Parameters et al.
# # Load parameters from training:
# _folder = "D:/deepcellcontrol/assets/simulated_inverter/"
# with open(_folder + "/training_parameters.json", "r") as f:
#     params = json.load(f)

# # Load trained model:
# network = tf.keras.models.load_model(
#     params["save_folder"]+"/model_besteval.hdf5"
#     )

# # Create results subfolder:
# os.makedirs(params["save_folder"] + "/evaluation/", exist_ok = True)

# # Misc params:
# cutoff = 24*12 # Cut off b/w past and future

# #%% Generate evaluation dataset:
# # Get an array of random stimulations:
# stims = dcc.utilities.random_stimulations(
#     timepoints = cutoff + params["horizon"],
#     total_simulations=50
#     )

# # Generate evaluation set:
# eval_sims = dcc.simulations.evaluation_set(
#     stims,
#     cell_class = dcc.simulations.CcaSR_Inverter,
#     cut_off = cutoff,
#     future_realizations = 100,
#     num_workers = None
#     )

# # Reformat a little bit:
# past_fluo = np.stack([cell[0] for cell in eval_sims], axis=0)
# futures_fluo = np.stack([cell[1] for cell in eval_sims], axis=0)

# np.save(params["save_folder"]+"/evaluation/past_fluo.npy", past_fluo)
# np.save(params["save_folder"]+"/evaluation/futures_fluo.npy", futures_fluo)
# np.save(params["save_folder"]+"/evaluation/stims.npy", stims)

# #%% Predict fluorescence:

# # Reformat to what the model expects:
# x = (np.stack([past_fluo/4095, stims[:,:cutoff]], axis = -1), stims[:,cutoff:])

# # Predict:
# yhat = network.predict(x, verbose = True)

# #%% Plot results and write them to disk:

# plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
# for c in range(stims.shape[0]):
#     plt.figure(1)
#     dcc.utilities.OptoPlotBackground(
#         stims[c,cutoff-plot_past:],
#         x=np.arange(-plot_past, params["horizon"])/12,
#         ymax = 4095
#         )
#     plt.plot(np.arange(-plot_past, 0)/12, past_fluo[c, -plot_past:],"k")
#     dcc.utilities.plotq(
#         futures_fluo[c], x = np.arange(0, params["horizon"])/12, color="k"
#         )
#     plt.plot(np.arange(0, params["horizon"])/12, yhat[c]*4095, "b")
#     plt.plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
#     plt.xlabel("time (h)")
#     plt.ylabel("Fluorescence (a.u.)")
#     plt.xlim([-plot_past/12,params["horizon"]/12])
#     plt.ylim([0,4095])
#     plt.savefig(params["save_folder"]+f'/evaluation/cell_{c:06d}.png',dpi=300)
#     plt.savefig(params["save_folder"]+f'/evaluation/cell_{c:06d}.svg',dpi=300)
#     plt.show()
#     plt.cla()

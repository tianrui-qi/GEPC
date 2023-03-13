#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:02:58 2023

@author: jlugagne
"""

import os
import json
import sys
import ast

import time
import uuid

import numpy as np

# Make sure the proper package is used:
sys.path.insert(0,'./../')
import deepcellcontrol as dcc

# Class of simulation model
cell_class = 'CcaSR_gillespie'
if len(sys.argv) > 1:
    cell_class = sys.argv[1]
cell_class_type = getattr(dcc.simulations, cell_class)
cell_class_object = cell_class_type()

# Have note checked that this works
WORKERS = None # NOne, 4, 8, 12 etc depending on number of cores
if len(sys.argv) > 2:
    WORKERS = int(sys.argv[2])
    
    # Convert to 
    if WORKERS==0:
        WORKERS = None

# Update parameters, if passed as new argument
new_params = {'h1': 0.0710/25, 
              'h2': 0.0303/50}
if len(sys.argv) > 3:
    with open(sys.argv[3], 'r') as f:
        new_params = json.load(f)

# Datasets folder (index with simulation data):
username="hklumpe"
simulated_data_folder = f"/projectnb/dunlop/{username}/deepcellcontrol/assets/simulated/data/"
base_folder = simulated_data_folder + f"/{cell_class_type.__name__}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_simulated_{uuid.uuid4()}"

# Training parameters:
training_cells = 10 # 10_000
timepoints = 36*12 # 36*12 How much time to simulate
nostim_start = 3*12 # 3*12 Timepoints with light off

# Evaluation parameters:
eval_cells = 10 #1_000
eval_cutoff = 24*12 # 24*12 # Number of past timepoints
eval_future_realizations = 1000 #1_000 # Number of future realizations per cell
eval_horizon = 4*12 # 4*12 # Number of future timepoints

print(cell_class)
print(new_params)

#%% Generate training set

# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(
            total_simulations=training_cells,
            timepoints=timepoints,
            nostim_timepoints=nostim_start
            )

# Generate cell responses:
fluorescence = dcc.simulations.training_set(
    stims, 
    cell_class = cell_class_type,
    num_workers = WORKERS,
    new_params = new_params,
    )

# Save to disk:
save_folder = base_folder + "/training_set/"
os.makedirs(save_folder, exist_ok = True)
np.save(save_folder+"/fluo1.npy", fluorescence)
np.save(save_folder+"/stims.npy", stims)

# Stash model parameters
cell_class_object.update_params(new_params)
model_params = getattr(cell_class_object, 'params')
with open(base_folder+'/model_parameters.json','w') as params_file:
    json.dump(model_params, params_file, indent=4)

#%% Generate evalaution set

# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(
    timepoints = eval_cutoff + eval_horizon, total_simulations = eval_cells
    )

# Generate evaluation set:
eval_sims = dcc.simulations.evaluation_set(
    stims,
    cell_class = cell_class_type,
    cut_off = eval_cutoff,
    future_realizations = eval_future_realizations,
    num_workers = WORKERS,
    new_params = new_params,
    )

# Reformat a little bit:
past_fluo = np.stack([cell[0] for cell in eval_sims], axis=0)
futures_fluo = np.stack([cell[1] for cell in eval_sims], axis=0)

# Save to disk
save_folder = base_folder + "/evaluation_set/"
os.makedirs(save_folder, exist_ok = True)
np.save(save_folder + "/past_fluo.npy", past_fluo)
np.save(save_folder + "/futures_fluo.npy", futures_fluo)
np.save(save_folder + "/stims.npy", stims)

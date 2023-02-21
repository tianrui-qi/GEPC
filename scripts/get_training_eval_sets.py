#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:02:58 2023

@author: jlugagne
"""

import os
import json

import numpy as np

import deepcellcontrol as dcc

# Datasets folder:
username='hklumpe'
base_folder = f"/projectnb/dunlop/{username}/deepcellcontrol/assets/simulated/data/"

# Class of cells to train:
cell_class = dcc.simulations.CcaSR_Autoactivation

# Training parameters:
training_cells = 10_000

# Evaluation parameters:
eval_cells = 500# 1_000
eval_cutoff = 24*12 # Number of past timepoints
eval_future_realizations = 500 # 1_000 # Number of future realizations per cell
eval_horizon = 4*12 # Number of future timepoints


#%% Generate training set

# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(total_simulations=training_cells)

# Generate cell responses:
fluorescence = dcc.simulations.training_set(stims, cell_class = cell_class)

# Save to disk:
save_folder = base_folder + "/" + cell_class.__name__ + "/training_set/"
os.makedirs(save_folder, exist_ok = True)
np.save(save_folder+"/fluo1.npy", fluorescence)
np.save(save_folder+"/stims.npy", stims)

#%% Generate evalaution set

# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(
    timepoints = eval_cutoff + eval_horizon, total_simulations = eval_cells
    )

# Generate evaluation set:
eval_sims = dcc.simulations.evaluation_set(
    stims,
    cell_class = cell_class,
    cut_off = eval_cutoff,
    future_realizations = eval_future_realizations,
    num_workers = None
    )

# Reformat a little bit:
past_fluo = np.stack([cell[0] for cell in eval_sims], axis=0)
futures_fluo = np.stack([cell[1] for cell in eval_sims], axis=0)

# Save to disk
save_folder = base_folder + "/" + cell_class.__name__ + "/evaluation_set/"
os.makedirs(save_folder, exist_ok = True)
np.save(save_folder + "/past_fluo.npy", past_fluo)
np.save(save_folder + "/futures_fluo.npy", futures_fluo)
np.save(save_folder + "/stims.npy", stims)

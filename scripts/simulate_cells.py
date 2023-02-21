#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:45:14 2023

@author: hklumpe
"""

import os
import copy
import matplotlib.pyplot as plt

import numpy as np

import deepcellcontrol as dcc

#%% Parameters
# Load default params:
params = copy.deepcopy(dcc.config.defaults)
params["save_folder"] = "../assets/test_simulation/"

os.makedirs(params["save_folder"], exist_ok = True)

#%% Generate training set:
# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(total_simulations=20)#3_000)

# Generate cell responses:
fluorescence = dcc.simulations.training_set(
    stims, 
    cell_class = dcc.simulations.CcaSR_gillespie_simple
    )

# Save to disk:
os.makedirs(params["save_folder"]+"training_set", exist_ok = True)
np.save(params["save_folder"]+"training_set/fluo1.npy", fluorescence)
np.save(params["save_folder"]+"training_set/stims.npy", stims)

#%% Plot results to check

fluorescence = np.load(params['save_folder']+'training_set/fluo1.npy')
stims = np.load(params['save_folder']+'training_set/stims.npy')

max_fluor = np.max(fluorescence)

n_cells = 20

for i in range(n_cells):

    plt.figure()
    plt.plot(fluorescence[i] / max_fluor)
    dcc.utilities.OptoPlotBackground(stims[i])
    plt.title(f'Cell {i}')

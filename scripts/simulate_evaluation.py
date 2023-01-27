# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:50:39 2023

@author: jeanbaptiste
"""
import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import deepcellcontrol as dcc

#%% Parameters et al.
# Load parameters from training:
_folder = "D:/deepcellcontrol/assets/simulated2/"
with open(_folder + "/training_parameters.json", "r") as f:
    params = json.load(f)

# Create results subfolder:
os.makedirs(params["save_folder"] + "/evaluation/", exist_ok = True)

# Load trained model:
network = tf.keras.models.load_model(
    params["save_folder"]+"/model_besteval.hdf5"
    )

# Misc params:
cutoff = 24*12 # Cut off b/w past and future

#%% Generate evaluation dataset:
# Get an array of random stimulations:
stims = dcc.utilities.random_stimulations(
    timepoints = cutoff + params["horizon"],
    total_simulations=50
    )

# Generate evaluation set:
eval_sims = dcc.simulations.evaluation_set(
    stims, cut_off = cutoff, future_realizations = 100
    )
# Reformat a little bit:
past_fluo = np.stack([cell[0] for cell in eval_sims], axis=0)
futures_fluo = np.stack([cell[1] for cell in eval_sims], axis=0)

np.save(params["save_folder"]+"/evaluation/past_fluo.npy", past_fluo)
np.save(params["save_folder"]+"/evaluation/futures_fluo.npy", futures_fluo)
np.save(params["save_folder"]+"/evaluation/stims.npy", stims)

#%% Predict fluorescence:
# Reformat to what the model expects:
x = (np.stack([past_fluo/4095, stims[:,:cutoff]], axis = -1), stims[:,cutoff:])

# Predict:
yhat = network.predict(x, verbose = True)

#%% Plot results and write them to disk:

plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
for c in range(stims.shape[0]):
    plt.figure(1)
    dcc.utilities.OptoPlotBackground(
        stims[c,cutoff-plot_past:],
        x=np.arange(-plot_past, params["horizon"])/12,
        ymax = 4095
        )
    plt.plot(np.arange(-plot_past, 0)/12, past_fluo[c, -plot_past:],"k")
    dcc.utilities.plotq(
        futures_fluo[c], x = np.arange(0, params["horizon"])/12, color="k"
        )
    plt.plot(np.arange(0, params["horizon"])/12, yhat[c]*4095, "b")
    plt.plot([-0.5/12, -0.5/12], [0, 4095], color="gray")
    plt.xlabel("time (h)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlim([-plot_past/12,params["horizon"]/12])
    plt.ylim([0,4095])
    plt.savefig(params["save_folder"]+f'/evaluation/cell_{c:06d}.png',dpi=300)
    plt.savefig(params["save_folder"]+f'/evaluation/cell_{c:06d}.svg',dpi=300)
    plt.show()
    plt.cla()

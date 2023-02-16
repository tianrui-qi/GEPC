#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:41:48 2023

@author: jlugagne
"""

import os
import copy
import time
import sys
import json
import uuid

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Make sure the proper package is used:
sys.path.insert(0,"/project/dunlop/shared_python_packages/deepcellcontrol/")
import deepcellcontrol as dcc


# Load params:
params = copy.deepcopy(dcc.config.defaults)

# If path to parameters JSON file passed as argument:
if len(sys.argv) > 1:
    with open(sys.argv[-1], "r") as f:
        params.update(json.load(f))

# Save folder:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_simulated_{uuid.uuid4()}"
save_path = params["models_folder"]+ "/" + params["save_folder"]


#%% Timeseries predictor training:
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Training prediction network\n" + "="*100)

# Generate Dataset object from saved data:
training_folders = [
    params["datasets_folder"] + "/" + x for x in params["training_sets"]
    ]
training_set = dcc.data.Datasets(
    training_folders,
    formatter = dcc.data.LSTMFormatter(params["features"]),
    parameters = params
    )
training_set.test_ratio = 0.1 # Fraction of samples that are left out of training
# Actually load data and normalize:
training_set.load()
training_set.normalize()
training_set.data_type='normalized_dataset'

# Init model:
network = dcc.models.lstm_mlp(params)

# Train, evaluate, and save to disk:
network = dcc.timeseries.batch_train_eval(training_set, network, params)

#%% Evaluate predictor:
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Evaluating prediction network\n" + "="*100)
os.makedirs(save_path + "/evaluation/", exist_ok = True)

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

# Load best evaluated model from disk:
network = tf.keras.models.load_model(save_path + "/model_besteval.hdf5")

# Predict:
yhat = network.predict(x, verbose = True)

# Save predictions:
np.save(save_path + "/evaluation/predictions.npy", yhat)

#Plot results and write them to disk:
plot_past = 3*12 # Only plot the past 3 hours (even though whole past is used)
for c in range(stims.shape[0]):
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
    plt.savefig(save_path+f'/evaluation/cell_{c:06d}.png',dpi=300)
    plt.savefig(save_path+f'/evaluation/cell_{c:06d}.svg',dpi=300)
    plt.cla()

#%% Evaluate control:
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Evaluating control\n" + "="*100)
os.makedirs(save_path + "/control/", exist_ok = True)

cell_class = getattr(dcc.simulations, params["cell_class"])
control_cells = 1_000
objectives = dcc.utilities.sine_objective(
    period=8*60,
    offset=1250,
    amplitude=750,
    delay=0,
    duration=48*60,
    sampling=5
    )
objectives = np.repeat(objectives[np.newaxis], repeats=control_cells, axis=0)
no_control = 3*12
end_control = 24*12

# Init controller:
controller = dcc.control.SplitLSTMMPC(
    model_file = save_path + '/model_besteval.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=params["horizon"], iterations=25, particles=40
        )
    )

# Create cells:
chip = [cell_class() for _ in range(control_cells)]
fluorescence = np.empty([control_cells, end_control + no_control])
stims = np.empty([control_cells, end_control + no_control])

# Run control experiment:
for t in range(end_control + no_control):
    
    # Run control:
    if t < no_control:
        # "Open-loop":
        stims[:, t] = 0
    else:
        stims[:,t] = controller.feedback(
            np.stack((fluorescence[:,:t]/4095, stims[:,:t]), axis=-1),
            objectives[:,t-no_control:t-no_control+params["horizon"]]/4095
            )
    
    # Run cells:
    for c, cell in enumerate(chip):
        # Set optogenetic signal:
        cell.set_light_events([stims[c,t]])
        
        # Run model for next 5 minutes:
        timeseries = cell.run((t+1)*5)
        fluorescence[c,t] = dcc.simulations.camera_sim(
            np.array(timeseries[-1]['F'])
            )
    
    time_str = f"{int(t/12):02d}h {5*(t%12):02d}m"
    print(time_str)
    

plt.figure(1)
plt.cla()
plt.plot((np.arange(objectives.shape[1])+no_control)/12, objectives[0],"k--")

# Plot cell fluorescence:
dcc.utilities.plotq(fluorescence[:,:(t+1)], x = np.arange(t+1)/12)

# Clean up the plot, save to disk:
plt.xlim([0, (end_control + no_control)/12])
plt.ylim([0, 4095])
plt.xlabel("time (h)")
plt.ylabel("Fluorescence (a.u.)")
plt.savefig(save_path+'/control/Population_quantiles.png',dpi=300)
plt.savefig(save_path+'/control/Population_quantiles.svg',dpi=300)

# Save results to disc:
np.save(save_path + "/control/fluo.npy", fluorescence)
np.save(save_path + "/control/stims.npy", stims)

print("="*50 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Done\n" + "="*50)
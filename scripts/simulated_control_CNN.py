# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:06:05 2023

@author: jeanbaptiste
"""
import os
import json
import sys
import glob
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

# Make sure the proper path is used:
dcc_repo_path = './../'
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

#%% Parameters:
model_path = glob.glob("../assets/models/cnn_model_JB*")[0]
# model_path = glob.glob("../assets/models/*4-20_05*")[0]
# with open(model_path + "/training_parameters.json", "r") as f:
#     params = json.load(f)
# Assume defaults are correct
# Load default params:
params = copy.deepcopy(dcc.config.defaults)
params['model_file'] = glob.glob(model_path + '/*.hdf5')[0]
params["past_steps"] = [36, 144]
params["horizon"] = 8*12
params["features"] = ["fluo1", "stims"]
params["training_parameters"]["epochs"] = 100
params["batch_size"] = 1000
params["loss"] = "Huber"

# Create results subfolder:
os.makedirs(model_path + "/control/", exist_ok = True)

# Misc params:
num_cells = 10
SAMPLING = 5 # Minutes between samples
no_control = 3*12 # Timepoints without control
control_duration = 3*12 # Timepoints with control

#%% Set up "experiment":
controller = dcc.control.SplitLSTMCNN_MPC(
    model_file = params['model_file'],
    n_bins = params['cnn_bins'],
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=params["horizon"], 
        iterations=25, particles=40,
        )
    )

objectives = dcc.utilities.sine_objective(
    period=8*60, #8*60,
    offset=1250,
    amplitude=750,
    delay=0,
    duration=(control_duration + params['horizon']) * SAMPLING, #24*60, # units of minutes
    sampling=SAMPLING
    )
objectives = np.repeat(objectives[np.newaxis], repeats=num_cells, axis=0)

# Create cells:
chip = [dcc.simulations.CcaSR_Autoactivation() for _ in range(num_cells)]
# chip = [dcc.simulations.CcaSR_gillespie() for _ in range(num_cells)]
fluorescence = np.empty([num_cells, objectives.shape[1] + no_control])
stims = np.empty([num_cells, objectives.shape[1] + no_control])

#%% Run control experiment:
for t in range(control_duration + no_control - 1):
    
    # # Init plot for timepoint:
    plt.figure(1)
    plt.cla()
    plt.plot((np.arange(objectives.shape[1])+no_control)/12, objectives[0],"k--")
    
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
        timeseries = cell.run((t+1)*SAMPLING)
        fluorescence[c,t] = dcc.simulations.camera_sim(
            np.array(timeseries[-1]['F'])
            )
        
        # Plot cell fluorescence:
        plt.plot(np.arange(t)/12, fluorescence[c,:t])
    
    # # Clean up the plot, save to disk:
    time_str = f"{int(t/12):02d}h {5*(t%12):02d}m"
    print(time_str)
    plt.ylim([0, 4095])
    plt.xlabel("time (h)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.title(time_str)
    plt.savefig(model_path+f'/control/time_{t:06d}.png',dpi=300)
    plt.show()
  
plt.figure(1)
plt.cla()
plt.plot((np.arange(objectives.shape[1])+no_control)/12, objectives[0],"k--")

# Plot cell fluorescence:
dcc.utilities.plotq(fluorescence[:,:(t+1)], x = np.arange(t+1)/12)

# Clean up the plot, save to disk:
plt.xlim([0, (control_duration + no_control)/12])
plt.ylim([0, 4095])
plt.xlabel("time (h)")
plt.ylabel("Fluorescence (a.u.)")
plt.savefig(model_path+'/control/Population_quantiles.png',dpi=300)
plt.savefig(model_path+'/control/Population_quantiles.svg',dpi=300)

# Save results to disc:
np.save(model_path + "/control/fluo.npy", fluorescence)
np.save(model_path + "/control/stims.npy", stims)

print("="*50 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Done\n" + "="*50)

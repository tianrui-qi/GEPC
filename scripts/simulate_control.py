# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:06:05 2023

@author: jeanbaptiste
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import deepcellcontrol as dcc

#%% Parameters:
_folder = "D:/deepcellcontrol/assets/simulated_inverter/"
with open(_folder + "/training_parameters.json", "r") as f:
    params = json.load(f)

# Create results subfolder:
os.makedirs(params["save_folder"] + "/control/", exist_ok = True)

# Misc params:
num_cells = 10
SAMPLING = 5 # Minutes between samples
nocontrol = 3*12 # Timepoints without control

#%% Set up "experiment":
controller = dcc.control.SplitLSTMMPC(
    model_file = params["save_folder"] + '/model_besteval.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=params["horizon"], iterations=25, particles=40
        )
    )

objectives = dcc.utilities.sine_objective(
    period=8*60,
    offset=1250,
    amplitude=750,
    delay=0,
    duration=24*60,
    sampling=SAMPLING
    )
objectives = np.repeat(objectives[np.newaxis], repeats=num_cells, axis=0)

# Create cells:
chip = [dcc.simulations.CcaSR_Inverter() for _ in range(num_cells)]
fluorescence = np.empty([num_cells, objectives.shape[1] + nocontrol])
stims = np.empty([num_cells, objectives.shape[1] + nocontrol])

#%% Run control experiment:
for t in range(objectives.shape[1] + nocontrol):
    
    # Init plot for timepoint:
    plt.figure(1)
    plt.cla()
    plt.plot((np.arange(objectives.shape[1])+nocontrol)/12, objectives[0],"k--")
    
    # Run control:
    if t < nocontrol:
        # "Open-loop":
        stims[:, t] = 0
    else:
        stims[:,t] = controller.feedback(
            np.stack((fluorescence[:,:t]/4095, stims[:,:t]), axis=-1),
            objectives[:,t-nocontrol:t-nocontrol+params["horizon"]]/4095
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
    
    # Clean up the plot, save to disk:
    time_str = f"{int(t/12):02d}h {5*(t%12):02d}m"
    print(time_str)
    plt.ylim([0, 4095])
    plt.xlabel("time (h)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.title(time_str)
    plt.savefig(params["save_folder"]+f'/control/time_{t:06d}.png',dpi=300)
    plt.show()

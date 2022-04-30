#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is script illustrate how the controller works with dummy examples

Created on Fri Apr 29 15:48:38 2022

@author: jeanbaptiste
"""
import copy
import time

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

import deepcellcontrol as dcc
# import qsub

# import qsub
params = copy.deepcopy(dcc.config.MLP_params)
params["features"] = [
    'fluo1',
    'area',
    'sharpness',
    'cell_count',
    'chamber_mean_fluo1',
    'chamber_std_fluo1',
    'neighbor_stims',
    'stims'
    ]

sets_folder = "D:/shared_packages/deepcellcontrol/assets/data/experimental/"
training_files = (
    sets_folder + "2022-04-13_TrainingSet2_dataset.pkl",
    sets_folder + "2022-04-15_TrainingSet3_dataset.pkl",
    sets_folder + "2022-04-22_TrainingSet6_dataset.pkl",
    sets_folder + "2022-04-23_TrainingSet7_dataset.pkl",
    sets_folder + "2022-04-24_TrainingSet8_dataset.pkl",
    )

# Load dataset:
dataset = dcc.data.Datasets(
    training_files,
    features = params["features"],
    formatter = dcc.data.LSTMFormatter(params["features"])
    )
dataset.load()
dataset.normalize()
dataset.data_type='normalized_dataset'
dataset.horizon = 48

# Generate random control "situations"
inputs = []
objectives = []
for _b in range(100):
    
    # Get random sample:
    sample = dataset.sample()
    
    # Create random objective:
    num_control_points = 3
    control_points = np.linspace(0, dataset.horizon-1, num=num_control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    while True:
        random_points[0] = sample[0][-1,0] + np.random.normal(0,.1)
        if 0<random_points[0]<1:
            break
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    objective = mapping(np.linspace(0, dataset.horizon-1, dataset.horizon))
    
    # Compile inputs for controller:
    inputs += [sample[0]]
    objectives += [objective]

inputs = np.array(inputs)

#%% Test out controller:

controller = dcc.control.SplitLSTMMPC(
    model_file = 'D:/shared_packages/deepcellcontrol/assets/models/2022-04-29_13-25-54/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=dataset.horizon, iterations=10, particles=20
        )
    )
print("Run time:")
for _ in range(1):
    t_start = time.perf_counter()
    controller.feedback(inputs,objectives)
    print(time.perf_counter() - t_start)

# outputs:
strategies = controller.strategy
predictions = controller.show_predict(inputs, strategies)

# Display control & strategy:
for s in range(20):
    
    dcc.utilities.OptoPlotBackground(
        np.concatenate(
            (inputs[s,:,dataset.features.index("stims")],strategies[s]),
            axis=0
            ),
        x=list(range(-dataset.past_steps,dataset.horizon))
        )
    plt.plot(
        range(-dataset.past_steps,0),
        inputs[s,:,dataset.features.index("fluo1")],
        label="past fluo"
        )
    plt.plot(objectives[s],label="objective")
    plt.plot(predictions[s,:,0], label="prediction")
    plt.plot([-0.5,-0.5],[0,1],'k--', linewidth=1)
    plt.ylim(0,1)
    plt.xlim(-dataset.past_steps,dataset.horizon-1)
    plt.title("DeepMPC controller")
    plt.xlabel("timepoints")
    plt.ylabel("Fluorescence (a.u.)")
    plt.legend()
    plt.show()

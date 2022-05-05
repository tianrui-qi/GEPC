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


params = copy.deepcopy(dcc.config.defaults)
params["training_sets"] += params["eval_sets"]
params["eval_sets"] = ()


# Load dataset:
dataset, _ = dcc.data.load_datasets(params)

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
objectives = np.array(objectives)

#%% Test out controller:

controller = dcc.control.SplitLSTMMPC(
    model_file = params["models_folder"] + '2022-05-01_21-53-02_9c02d092-97be-4378-82e3-d36e9b0509af/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=dataset.horizon, iterations=20, particles=20
        )
    )
print("Run time:")
for _ in range(10):
    t_start = time.perf_counter()
    controller.feedback(inputs,objectives)
    print(time.perf_counter() - t_start)

# outputs:
strategies = controller.strategy
predictions = controller.show_predict(inputs, strategies)
predictions = np.squeeze(predictions)

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
    plt.plot(predictions[s,:], label="prediction")
    plt.plot([-0.5,-0.5],[0,1],'k--', linewidth=1)
    plt.ylim(0,1)
    plt.xlim(-dataset.past_steps,dataset.horizon-1)
    plt.title("DeepMPC controller")
    plt.xlabel("timepoints")
    plt.ylabel("Fluorescence (a.u.)")
    plt.legend()
    plt.show()

#%% Test as part of control server:

dummy_dispatcher = lambda output, meta: print(f"{meta['index']} dispatched")

server = dcc.server.Server(controller, device = "CPU")
server.start()

for index in range(25):
    sub_selection = np.random.choice(inputs.shape[0],size=27,replace=False)
    server.queue.put(
        (
            (inputs[sub_selection],objectives[sub_selection]),
            dict(index=index),
            dummy_dispatcher
            )
        )
    time.sleep(1)
print("Done sending")
            
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is script illustrate how the controller works with dummy examples

Created on Fri Apr 29 15:48:38 2022

@author: jeanbaptiste
"""
import os
import copy
import time
import json

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import deepcellcontrol as dcc

# %% Import parameters and model from training folder:
res_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2"
# res_folder = "D:/shared_packages/deepcellcontrol/assets/models/2022-05-07_18-16-47_66539337-00a1-4b86-8fa2-48448c35d386"

with open(res_folder + "/training_parameters.json","r") as f:
    params = json.load(f)

params["datasets_folder"] = dcc.config.defaults["datasets_folder"]

# Load dataset:
dataset, eval_set = dcc.data.load_datasets(params)

# Load best eval model:
model = tf.keras.models.load_model(res_folder + "/model_besteval.hdf5")

# Evaluate it:
metrics, eval_d = dcc.timeseries.evaluate(
    eval_set, model, batch_size=100_000, num_batches = 1, return_eval=True
    )


# Plot evaluation:
plt.plot(metrics['mae'],"b", label="MAE")
plt.plot(metrics['rmse'],"g", label="RMSE")
plt.grid(axis='y',which='both')
plt.xlabel('horizon')
plt.ylabel('error (a.u.)')
plt.title(f'evaluation MAE ({np.mean(metrics["mae"]):.2g}) & RMSE ({np.mean(metrics["rmse"]):.2g})')
plt.legend()
plt.savefig(res_folder+'/evaluation_error_besteval.png',dpi=300)
plt.savefig(res_folder+'evaluation_error_besteval.svg',dpi=300)
plt.show()


# Plot single cell evaluations:
os.makedirs(res_folder+'/single_cell_evals', exist_ok = True)
fluos, stims = dataset.formatter.reconstruct(eval_d['input'],eval_d['groundtruth'])
for eval_num in range(50):
    dcc.utilities.evaluationPlot(
        stims[eval_num],
        fluos[eval_num,:-params['horizon']],
        fluos[eval_num,-params['horizon']:],
        eval_d["prediction"][eval_num],
        dyn_range=1,
        savefig = res_folder + '/single_cell_evals/sample_besteval_%02d'%eval_num,
        show = True
        )

# Plot RMSE and MAE distributions:
yval = eval_d["groundtruth"]
yhat = eval_d["prediction"]
rmse = np.sqrt(np.mean(np.square(yhat-yval),axis=1))  # RMSE over prediction horizon
mae = np.mean(np.abs(yhat-yval),axis=1) # MAE over prediction horizon
bins = np.logspace(
    np.log10(np.min(rmse)),
    np.log10(np.max(rmse)), 
    100
    )
plt.hist(rmse,bins=bins, density=True)
plt.xscale("log")
plt.xlabel("RMSE")
plt.ylabel("density")
plt.savefig(res_folder+'/rmse_distribution.png',dpi=300)
plt.savefig(res_folder+'/rmse_distribution.svg',dpi=300)
plt.show()

#%% Generate random control "situations"

params["training_sets"] += params["eval_sets"]
params["eval_sets"] = ()
dataset, _ = dcc.data.load_datasets(params)
# dataset.past_steps = 36

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

os.makedirs(res_folder+'/dummy_control', exist_ok = True)

controller = dcc.control.SplitLSTMMPC(
    model_file = res_folder + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=dataset.horizon, iterations=20, particles=20
        )
    )

print("Run time:")
for _ in range(1):
    t_start = time.perf_counter()
    im_strat = controller.feedback(inputs,objectives)
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
        x=list(range(-dataset.past_steps[-1],dataset.horizon))
        )
    plt.plot(
        range(-dataset.past_steps[-1],0),
        inputs[s,:,dataset.features.index("fluo1")],
        label="past fluo"
        )
    plt.plot(objectives[s],label="objective")
    plt.plot(predictions[s,:], label="prediction")
    plt.plot([-0.5,-0.5],[0,1],'k--', linewidth=1)
    plt.ylim(0,1)
    plt.xlim(-dataset.past_steps[-1],dataset.horizon-1)
    plt.title("DeepMPC controller")
    plt.xlabel("timepoints")
    plt.ylabel("Fluorescence (a.u.)")
    plt.savefig(res_folder+f'/dummy_control/sample_{s:06d}.png',dpi=300)
    plt.savefig(res_folder+f'/dummy_control/sample_{s:06d}.svg',dpi=300)
    plt.legend()
    plt.show()

#%% Test as part of control server:

# dummy_dispatcher = lambda output, meta: print(f"{meta['index']} dispatched")
def dummy_dispatcher(output, meta):
    
    similarity = np.logical_xor(output, im_strat[meta["selection"]])
    
    similarity = 1-np.mean(similarity)
    
    print(f"{meta['index']} dispatched, similarity: {100*similarity:.3f} %")


# server = dcc.server.Server(controller, device = "GPU")
# server.start()

# server = dcc.server.DistantServer("DESKTOP-A5D6QR1")
server = dcc.server.DistantServer("127.0.0.1")
server.start()

for index in range(100_000):
    sub_selection = np.random.choice(inputs.shape[0],size=27,replace=False)
    server.queue.put(
        (
            (inputs[sub_selection],objectives[sub_selection]),
            dict(index=index, selection = sub_selection),
            dummy_dispatcher
            )
        )
    time.sleep(1)
print("Done sending")
            
    

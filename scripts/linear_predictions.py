# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:16:27 2022

@author: jeanbaptiste
"""

import copy
import os
import time
import sys
import json
import uuid
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras

sys.path.insert(0,"/project/dunlop/shared_python_packages/deepcellcontrol/")
import deepcellcontrol as dcc

# Load params:
params = copy.deepcopy(dcc.config.defaults)

# Save folder:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
params["past_steps"] = 24
params["features"] = (
    'fluo1',
    'area',
    'sharpness',
    'cell_count',
    'chamber_mean_fluo1',
    'chamber_std_fluo1',
    'neighbor_stims',
    'stims'
    )
params["batch_size"] = 1000
params["training_parameters"]["epochs"] = 100
params["datasets_folder"] = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/data/"
params["models_folder"] = "D:/deepcellcontrol/assets/models/"

save_folder = params["models_folder"]+ params["save_folder"]

# Load datasets:
training_set, evaluation_set = dcc.data.load_datasets(params)

#%% Init model:
linear_combination = dcc.models.linear_predictor(params)

# Train and evaluate:
linear_combination = dcc.timeseries.batch_train_eval(
    training_set, linear_combination, params, evaluation_dataset=evaluation_set
    )

#%% Load 2-hour horizon evaluation:

metrics, eval_d = dcc.timeseries.evaluate(
    evaluation_set, linear_combination, batch_size=100_000, num_batches = 1, return_eval=True
    )

# RMSE over prediction horizon:
rmse = np.sqrt(
    np.mean((4095*(eval_d["prediction"]-eval_d["groundtruth"]))**2,axis=1)
    )
rmse_order = np.argsort(rmse)

# Original LSTM:
lstm_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-35_b0d0b5c3-158d-4476-926b-75b2607d6154/"
with open(lstm_folder + "/training_parameters.json","r") as f:
    lstm_params = json.load(f)
lstm_params["datasets_folder"] = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/data/"
lstm_dataset, lstm_eval = dcc.data.load_datasets(lstm_params)
lstm = tf.keras.models.load_model(lstm_folder+ "/model_besteval.hdf5")
_, lstm_eval_d = dcc.timeseries.evaluate(
    lstm_eval, lstm, batch_size=100_000, num_batches = 1, return_eval=True
    )
lstm_rmse = np.sqrt(
    np.mean((4095*(lstm_eval_d["prediction"]-lstm_eval_d["groundtruth"]))**2,axis=1)
    )
lstm_rmse_order = np.argsort(lstm_rmse)

#%% Panel D - RMSE distro

plt.figure()

# Log-spaced bins:
nbins = 100
bins = np.logspace(0, 4, nbins + 1, base=10)
# Plot hist:
n, _, _ = plt.hist(rmse, bins=bins, color="b")
# Plot 25th, median, and 75th %ile:
qs = np.quantile(rmse, [.25, .5, .75])
for q in qs:
    plt.plot([q, q], [0, 5000], "b--", zorder=-1)

_ = plt.hist(lstm_rmse, bins=bins, histtype="step", color="orange")
qs = np.quantile(lstm_rmse, [.25, .5, .75])
for q in qs:
    plt.plot([q, q], [0, 5000], "--", color="orange", zorder=-1)


plt.xscale("log")
plt.xlim([5, 5000])
plt.ylim([0, 5000])
plt.xlabel("Root mean square error (a.u.)")
plt.ylabel("Count")
plt.savefig(save_folder+"RMSEdistro.png", dpi=300)
plt.savefig(save_folder+"RMSEdistro.svg", dpi=300)
plt.show()

print(f"Linear RMSE - mean: {np.mean(rmse)}, median: {np.median(rmse)}")
print(f"LSTM RMSE - mean: {np.mean(lstm_rmse)}, median: {np.median(lstm_rmse)}")
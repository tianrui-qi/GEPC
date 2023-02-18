# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:19:27 2023

@author: jeanbaptiste
"""
import os
import copy

import numpy as np

import deepcellcontrol as dcc

#%% Parameters
# Load default params:
params = copy.deepcopy(dcc.config.defaults)
params["save_folder"] = "D:/deepcellcontrol/assets/simulated_inverter/"
params["past_steps"] = [36, 144]
params["features"] = ["fluo1", "stims"]
params["training_parameters"]["epochs"] = 200
params["models_folder"] = "" # TODO: fix this in timeseries.batch_train_eval

os.makedirs(params["save_folder"], exist_ok = True)

#%% Generate Dataset object from saved data:

training_set = dcc.data.Datasets(
    [params["save_folder"]+"training_set/"],
    formatter = dcc.data.LSTMFormatter(params["features"]),
    parameters = params
    )

training_set.test_ratio = 0.1 # Fraction of samples that are left out of training
# Actually load data and normalize:
training_set.load()
training_set.normalize()
training_set.data_type='normalized_dataset'

#%% Train

# Init model:
network = dcc.models.lstm_mlp(params)

# Train, evaluate, and save to disk:
network = dcc.timeseries.batch_train_eval(
    training_set, network, params
    )

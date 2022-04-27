# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:16:27 2022

@author: jeanbaptiste
"""

import copy
import os

import deepcellcontrol as dcc

# import qsub
params = copy.deepcopy(dcc.config.MLP_params)
params["training_parameters"]["epochs"]=30
params["features"] = ("fluo1", "stims")

data_files = ("D:/shared_packages/deepcellcontrol/assets/data/experimental/2022-04-19_TrainingSet4_dataset.pkl",)

# Load dataset:
dataset = dcc.data.Datasets(
    data_files, features = params["features"]
    )

dataset.format_mode = "lstm"
dataset.load()
dataset.normalize()
dataset.data_type='normalized_dataset'

# Init LSTM network:
network = dcc.models.lstm(
    past_steps=36, 
    features=len(params["features"]),
    horizon=24,
    latent_dim=16,
    activation='linear',
    loss='mse',
    metrics=None,
    learning_rate=0.001
    )

# Create save folder:
if not os.path.exists(params['save_folder']):
    os.makedirs(params['save_folder'])

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(dataset, network, params)

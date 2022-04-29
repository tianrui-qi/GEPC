# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:16:27 2022

@author: jeanbaptiste
"""

import copy
import os
import datetime

import deepcellcontrol as dcc

# import qsub
params = copy.deepcopy(dcc.config.MLP_params)
params["training_parameters"]["epochs"]=3
params["training_parameters"]["steps_per_epoch"]=200
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
params["save_folder"] = "/home/jeanbaptiste/data/shared_packages/deepcellcontrol/assets/models/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sets_folder = "/home/jeanbaptiste/data/shared_packages/deepcellcontrol/assets/data/experimental/"
training_files = (
    sets_folder + "2022-04-13_TrainingSet2_dataset.pkl",
    sets_folder + "2022-04-15_TrainingSet3_dataset.pkl",
    sets_folder + "2022-04-22_TrainingSet6_dataset.pkl",
    sets_folder + "2022-04-23_TrainingSet7_dataset.pkl",
    sets_folder + "2022-04-24_TrainingSet8_dataset.pkl",
    )
# validation_files = ("D:/shared_packages/deepcellcontrol/assets/data/experimental/2022-04-19_TrainingSet4_dataset.pkl",)

# Load dataset:
training_set = dcc.data.Datasets(
    training_files,
    features = params["features"],
    formatter = dcc.data.LSTMAutoencoderFormatter(params["features"])
    )
training_set.load()
training_set.normalize()
training_set.data_type='normalized_dataset'

# Start with autoencoder:
network = dcc.models.lstm(
    past_steps=36,
    horizon=36,
    features_dim=len(params["features"]),
    latent_dim=64,
    output_dim=len(params["features"]),
    activation='linear',
    loss='mse',
    metrics=None,
    learning_rate=0.001
    )

# Create save folder:
if not os.path.exists(params['save_folder']):
    os.makedirs(params['save_folder'])

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(
    training_set,
    network,
    params,
    plot_autoencoding=True,
    plot_singlecell= False,
    )

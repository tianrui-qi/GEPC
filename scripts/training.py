# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:16:27 2022

@author: jeanbaptiste
"""

import copy
import os
import time
import sys

sys.path.insert(0, "/project/dunlop/shared_python_packages/deepcellcontrol")
import deepcellcontrol as dcc

# import qsub
params = copy.deepcopy(dcc.config.MLP_params)
params["training_parameters"]["epochs"]=10_000
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
params["save_folder"] = (
    "/projectnb/dunlop/JB/deepcellcontrol/assets/models/" + 
    time.strftime("%Y-%m-%d_%H-%M-%S")
    )

sets_folder = "/projectnb/dunlop/JB/deepcellcontrol/assets/data/"
training_files = (
    sets_folder + "2022-04-13_TrainingSet2_dataset.pkl",
    sets_folder + "2022-04-19_TrainingSet4_dataset.pkl",
    sets_folder + "2022-04-22_TrainingSet6_dataset.pkl",
    sets_folder + "2022-04-24_TrainingSet8_dataset.pkl",
    )
validation_files = (
    sets_folder + "2022-04-15_TrainingSet3_dataset.pkl",
    sets_folder + "2022-04-21_TrainingSet5_dataset.pkl",
    sets_folder + "2022-04-23_TrainingSet7_dataset.pkl",
    )

# Load datasets:
training_set = dcc.data.Datasets(
    training_files,
    features = params["features"],
    formatter = dcc.data.LSTMFormatter(params["features"])
    )
training_set.load()
training_set.normalize()
training_set.data_type='normalized_dataset'

validation_set = dcc.data.Datasets(
    training_files,
    features = params["features"],
    formatter = dcc.data.LSTMFormatter(params["features"])
    )
validation_set.load()
validation_set.normalize()
validation_set.data_type='normalized_dataset'

# Init model:
hyper_parameters = dcc.models.default_hyper_parameters
hyper_parameters["features"] = len(params["features"])
hyper_parameters["latent_dim"] = 16
hyper_parameters["output_mode"] = "timedistributed"
hyper_parameters["output_dim"] = 1
network = dcc.models.lstm(hyper_parameters)

# Create save folder:
if not os.path.exists(params['save_folder']):
    os.makedirs(params['save_folder'])

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(
    training_set, network, params, evaluation_dataset=validation_set
    )

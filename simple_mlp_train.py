#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:46:15 2022

@author: jeanbaptiste
"""

import copy
import os

import deepcellcontrol as dcc
# import qsub
params = copy.deepcopy(dcc.config.MLP_params)

# Load dataset:
dataset = dcc.data.Datasets(
    params["datasets"], features = params['features']
    )
dataset.format_mode = "mlp"
dataset.load()
dataset.normalize()
dataset.data_type='normalized_dataset'

# Init LSTM network:
network = dcc.models.mlp(
    past_steps=36,
    horizon=24,
    features=2,
    )

# Create save folder:

if not os.path.exists(params['save_folder']):
    os.makedirs(params['save_folder'])

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(dataset, network, params)

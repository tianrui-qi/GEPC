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

import deepcellcontrol as dcc

# Load params:
params = copy.deepcopy(dcc.config.defaults)

# If path to parameters JSON file passed as argument:
if len(sys.argv) >= 3:
    with open(sys.argv[2], "r") as f:
        params.update(json.load(f))

# Save folder:
params["save_folder"] = time.strftime("%Y-%m-%d_%H-%M-%S")

# Load datasets:
training_set, evaluation_set = dcc.data.load_datasets(params)

# Init model:
network = dcc.models.lstm(params)

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(
    training_set, network, params, evaluation_dataset=evaluation_set
    )

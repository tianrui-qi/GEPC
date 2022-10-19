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

import numpy as np

sys.path.insert(0,"/project/dunlop/shared_python_packages/deepcellcontrol/")
import deepcellcontrol as dcc

# Load params:
params = copy.deepcopy(dcc.config.defaults)

# If path to parameters JSON file passed as argument:
if len(sys.argv) > 1:
    with open(sys.argv[-1], "r") as f:
        params.update(json.load(f))

# Save folder:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
params["past_steps"] = [36, 144]

# Load datasets:
training_set, evaluation_set = dcc.data.load_datasets(params)

# Patch batch methods:
import types
def nopastbatch(self):
    X, Y = dcc.data.Datasets.batch(self)
    X = X[1]
    return X, Y
training_set.batch = types.MethodType(nopastbatch, training_set)
evaluation_set.batch = types.MethodType(nopastbatch, evaluation_set)

# Init model:
network = dcc.models.mlp_nopast(params)

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(
    training_set, network, params, evaluation_dataset=evaluation_set
    )

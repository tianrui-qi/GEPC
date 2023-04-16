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

sys.path.insert(0,"/project/dunlop/JB/deepcellcontrol/")
import deepcellcontrol as dcc

# Load params:
params = copy.deepcopy(dcc.config.defaults)

# If path to parameters JSON file passed as argument:
if len(sys.argv) > 1:
    with open(sys.argv[-1], "r") as f:
        params.update(json.load(f))

# Load datasets:
training_set, evaluation_set = dcc.data.load_datasets(params)

# Init model:
network = dcc.models.lstm_mlp(params)

# Train and evaluate:
network = dcc.timeseries.batch_train_eval(
    training_set, network, params, evaluation_dataset=evaluation_set
    )

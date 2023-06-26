# -*- coding: utf-8 -*-
"""
Train the linear regression model & get the results for table S1

Created on Fri Apr 22 16:16:27 2022

@author: jeanbaptiste
"""

import copy
import time
import json
import uuid

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras

# sys.path.insert(0,"/project/dunlop/shared_python_packages/deepcellcontrol/")
import deepcellcontrol as dcc

# Load params:
# params = copy.deepcopy(dcc.config.defaults)
params_file = "Y:/projectnb/dunlop/hklumpe/deepcellcontrol/assets/models/2023-03-17_02-06-00_simulated_8b84bdd8-556c-41be-a90a-1431ad793f23/training_parameters.json"
with open(params_file, "r") as f:
    params = json.load(f)

# Save folder:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
params["batch_size"] = 100
params["training_parameters"]["epochs"] = 200
params["datasets_folder"] = "Y:" + params["datasets_folder"] + "/"
params["models_folder"] = "D:/deepcellcontrol/assets/models/"
params["past_steps"] = 72

# Load datasets:
training_set, _ = dcc.data.load_datasets(params)

#%% train models:

params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
save_folder = params["models_folder"]+ params["save_folder"]

linear_combination = dcc.models.linear_predictor(params)

# Train and evaluate:
_ = dcc.timeseries.batch_train_eval(
    training_set, linear_combination, params, evaluation_dataset=None
    )


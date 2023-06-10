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
params = copy.deepcopy(dcc.config.defaults)

# Save folder:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
params["past_steps"] = 72
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
params["batch_size"] = 100
params["training_parameters"]["epochs"] = 200
params["datasets_folder"] = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/data/"
params["models_folder"] = "D:/deepcellcontrol/assets/models/"

# Load datasets:
training_set, evaluation_set = dcc.data.load_datasets(params)

evaluation_set.batch_size = 100_000
evaluation_set.past_steps = 144
eval_inputs, eval_gt = next(evaluation_set)


pasts = [6, 12, 24, 36, 48, 72, 96, 120, 144]

#%% train models:

for past in pasts:
    params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
    params["past_steps"] = past
    evaluation_set.past_steps = past
    save_folder = params["models_folder"]+ params["save_folder"]

    linear_combination = dcc.models.linear_predictor(params)
    
    # Train and evaluate:
    _ = dcc.timeseries.batch_train_eval(
        training_set, linear_combination, params, evaluation_dataset=evaluation_set
        )


#%% Eval and plot:
models = [
    "D:/deepcellcontrol/assets/models/2023-05-19_09-52-40_085a1e1d-d9ea-44b0-8a0e-968ceac892ea",
    "D:/deepcellcontrol/assets/models/2023-05-18_23-39-40_a3b9ef48-1c4d-4930-bb32-8a589170d2c5",
    "D:/deepcellcontrol/assets/models/2023-05-18_23-55-51_8a76c151-bf3c-43d5-8799-4ea13239228a",
    "D:/deepcellcontrol/assets/models/2023-05-19_00-12-10_43d1d3e6-57a5-43dd-b47b-fe28123d9292",
    "D:/deepcellcontrol/assets/models/2023-05-19_00-28-22_db66f324-9107-4d35-9958-d8b5e5438dae",
    "D:/deepcellcontrol/assets/models/2023-05-19_00-44-59_fd24db6c-f596-4d78-adbf-b14de7e6f866",
    "D:/deepcellcontrol/assets/models/2023-05-19_08-58-19_348be4ad-f486-42e1-9d19-6615061b9a25",
    "D:/deepcellcontrol/assets/models/2023-05-19_09-35-32_54361e35-c20a-4d0c-a3c7-f777bd5b3a32",
    "D:/deepcellcontrol/assets/models/2023-05-19_11-23-12_db274182-b76a-4664-9dd1-2ff69ec62345",
    "D:/deepcellcontrol/assets/models/2023-05-19_11-39-34_2507e929-7933-47c5-a64b-5f67216bbc68",
    ]
    
rmses = [None]*len(pasts)
for model in models:
    
    linear_combination = tf.keras.models.load_model(model + "/model.hdf5")
    with open(model + "/training_parameters.json", "r") as f:
        past = json.load(f)["past_steps"]
    print(past)
    
    prediction = linear_combination.predict(
        [eval_inputs[0][:,-past:], eval_inputs[1]], verbose = 1
        )

    # RMSE over prediction horizon:
    rmse = np.sqrt(
        np.mean((4095*(prediction-eval_gt[:,:,0]))**2, axis=1)
        )
    
    p = pasts.index(past)
    rmses[p] = rmse

    plt.bar(p, np.median(rmse))
plt.xticks(list(range(len(pasts))), pasts)

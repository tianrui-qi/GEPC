# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:33:27 2023

@author: jeanbaptiste
"""
import json
import time
import sys
import uuid
import os

import pandas
import qsub

dcc_data_path = "/projectnb/dunlop/JB/deepcellcontrol/"
dcc_repo_path = "/project/dunlop/JB/deepcellcontrol/"

sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

def params_change(params):
    
    print(f"\n\n{'-'*50}\nChanges:\n{json.dumps(params, indent=4)}")
    
    _params = dict(
        datasets_folder = dcc_data_path + "assets/data/",
        models_folder = dcc_data_path + "assets/models/",
        )
    if "save_folder" not in params:
        _params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}/"
    
    _params.update(params)
    
    savefolder = _params["models_folder"] + _params["save_folder"]
    print(f"Save folder:\n{savefolder}")
    
    with open(savefolder+"/sub_parameters.json","w") as f:
        json.dump(_params,f, indent=4)
    
    return savefolder+"/submission_parameters.json", _params

def submit(
        changes, 
        name="dcc_training", 
        time_limit=2, 
        record_csv = "/projectnb/dunlop/JB/deepcellcontrol/revisions.csv",
        ):
    
    files, configs, folders = [], [], ""
    for c, change in enumerate(changes):
        file, config = params_change(change)
        files.append([file])
        configs.append(config)
        folders += configs["save_folder"] + "\n"
    
    job_id = qsub.submit(
        dcc_repo_path + "scripts/training.py",
        job_array = True,
        name = name,
        args = files,
        kwargs = [{}] * len(files),
        conda_env="delta_env",
        hardware_requirements = dict(
            time_limit = time_limit,
            cores=4,
            gpus=1,
            mem_per_core=4,
            )
        )
    
    df = pandas.DataFrame(
        columns=("time", "name", "job_id", "changes", "folders", "comments")
        )
    df.loc[0,"time"] = pandas.Timestamp.now()
    df.loc[0,"name"] = name
    df.loc[0,"changes"] = changes
    df.loc[0, "job_id"] = job_id
    df.loc[0, "folders"] = folders
    df.to_csv(record_csv, mode='a', header=not os.path.exists(record_csv))
    

#%% Range of increasing network size:
    
changes = (
    {"lstm_units":8, "latent_dim":4, "mlp_layers":1, "mlp_dim":4},
    {"lstm_units":16, "latent_dim":8, "mlp_layers":2, "mlp_dim":8},
    {"lstm_units":32, "latent_dim":16, "mlp_layers":3, "mlp_dim":16},
    {"lstm_units":64, "latent_dim":16, "mlp_layers":5, "mlp_dim":32},
    {"lstm_units":128, "latent_dim":32, "mlp_layers":10, "mlp_dim":64},
    {"lstm_units":256, "latent_dim":64, "mlp_layers":20, "mlp_dim":128},
    )
for change in changes:
    change["batch_size"] = 1000

submit(changes, "dcc_increasing_ntwk", time_limit=4)

#%% Different features:


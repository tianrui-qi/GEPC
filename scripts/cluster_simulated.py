# -*- coding: utf-8 -*-

import json
import time
import os
import sys
import copy

import qsub

# TODO: use different deepcellcontrol folder:
username='hklumpe'
dcc_data_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"
dcc_repo_path = "/project/dunlop/shared_python_packages/deepcellcontrol/"

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

def params_change(params):
    
    # Create relevant directories:
    assets = os.path.expanduser("~") + "/qsub_scripts/assets/"
    subfolders = [ f.path for f in os.scandir(assets) if f.is_dir() ]
    foldername = assets + time.strftime("%Y-%m-%d_%H-%M-%S") + f"_submission_{len(subfolders)}/"
    os.makedirs(foldername)
    print(foldername)
    
    # Save parameters to json file:
    with open(foldername+"parameters.json","w") as f:
        json.dump(params,f, indent=4)
    
    # Return path to json file:
    return foldername+"parameters.json"

#%% Launch single training:
    
cell_class = 'CcaSR_gillespie_simple' # in training data path, and name of class in dcc.simulations

# Fields to change in config.py:
config = dict(
    training_parameters = dict(
        epochs = 200, # 200 is typically enough
        ),
    datasets_folder = dcc_data_path + f"assets/simulated/data/{cell_class}/", # Point to generated sets folder
    training_sets = ("training_set",), # Training subfolder(s)
    eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
    features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
    cell_class = cell_class, # Cell class to use in dcc.simulations
    )

# Updated config and save it to disk:
saved_config_file = params_change(config)

# Submit qsub request for single job:
job_id = qsub.submit(
    dcc_repo_path + "scripts/simulated_pipeline.py",
    args = [saved_config_file],
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 2,
        cores=4,
        gpus=1,
        mem_per_core=4,
        )
    )

#%% Launch job array:

# Base config to update:
base_config = dict(
    training_parameters = dict(
        epochs = 200,
        ),
    datasets_folder = dcc_data_path + "assets/simulated/data/CcaSR_gillespie/",
    training_sets = ("training_set",),
    features = ("fluo1", "stims"),
    eval_sets = ("evaluation_set",),
    cell_class = "CcaSR_gillespie"
    )

# List of different configs for different jobs
configs = []

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 36}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 48}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 12}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 6}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"mlp_layers": 20, "mlp_dim": 256}) # for example
configs.append(params_change(new_config))

# Submit job array:
job_id = qsub.submit(
    dcc_repo_path + "scripts/training.py",
    job_array = True,
    args = configs,
    kwargs = [{}] * len(configs),
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 2,
        cores=4,
        gpus=1,
        mem_per_core=4,
        )
    )

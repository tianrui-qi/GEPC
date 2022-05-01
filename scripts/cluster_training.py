# -*- coding: utf-8 -*-

import json
import time
import os

import qsub

dcc_data_path = "/projectnb/dunlop/JB/deepcellcontrol/"
dcc_repo_path = "/project/dunlop/shared_python_packages/deepcellcontrol/"

def params_change(params):
    
    _params = dict(
        datasets_folder = dcc_data_path + "assets/data/",
        models_folder = dcc_data_path + "assets/models/",
        )
    
    _params.update(params)
    
    assets = "/usr2/postdoc/jlugagne/qsub_scripts/assets/"
    subfolders = [ f.path for f in os.scandir(assets) if f.is_dir() ]
    foldername = assets + time.strftime("%Y-%m-%d_%H-%M-%S") + f"_submission_{len(subfolders)}/"
    os.makedirs(foldername)
    
    with open(foldername+"parameters.json","w") as f:
        json.dump(_params,f)
    
    return foldername+"parameters.json"

#%% Launch single training:

saved_config = params_change(
    dict(
        training_parameters = dict(epoch = 10_000)
        )
    )
job_id = qsub.submit(
    dcc_repo_path + "scripts/training.py",
    args = [saved_config],
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 16,
        cores=4,
        gpus=1,
        mem_per_core=4,
        )
    )

#%% Launch training array:

configs = []
configs.append([params_change(dict())])
configs.append([params_change(dict(features = ("fluo1", "stims")))])
configs.append([params_change(dict(latent_dim = 32))])
configs.append([params_change(dict(horizon = 48))])

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



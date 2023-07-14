# -*- coding: utf-8 -*-

import json
import time
import sys
import uuid

import qsub

dcc_data_path = "/projectnb/dunlop/JB/deepcellcontrol/"
dcc_repo_path = "/project/dunlop/shared_python_packages/deepcellcontrol/"

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
    
    return savefolder+"/submission_parameters.json"

#%% Launch single training:

saved_config, _ = params_change(
    dict(
        training_parameters = dict(
            epochs = 200,
            ),
        training_sets = dcc.config.defaults["training_sets"] + dcc.config.defaults["eval_sets"],
        eval_sets = ()
        )
    )
job_id = qsub.submit(
    dcc_repo_path + "scripts/training.py",
    args = [saved_config],
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 1,
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



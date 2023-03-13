#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:17:10 2023

@author: hklumpe
"""

import sys
import os
import time
import json

sys.path.insert(0,'/project/dunlop/shared_python_packages/')
import qsub

# TODO: use different deepcellcontrol folder:
username='hklumpe'
dcc_data_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

#%% Function for dumping new parameters to a json (easy passing to script)

def params_change(params):
    """
    Store params (dict) in `~/qsub_scripts/assets/` in specific folder for
    current simulation; return location of this dict
    """
    
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

#%% Submit single script
cell_class = 'CcaSR_gillespie_simple_noE'
WORKERS = 0 #8, script interprets 0 as None

job_id = qsub.submit(
    dcc_repo_path + "scripts/get_training_eval_sets.py",
    conda_env="delta_env",
    args = [cell_class, WORKERS, ],
    hardware_requirements = dict(
        time_limit = 1, #2
        cores=8, #4
        gpus=1,
        mem_per_core=4,
        )
    )

#%% Submit scripts

cell_class = 'CcaSR_gillespie'

new_params_list = [{'h1': 0.0710/25*10, 'h2': 0.0303/50*10},
                   {'h1': 0.0710/25/10, 'h2': 0.0303/50/10},
                   {'h1': 0.0710/25*10, 'h2': 0.0303/50*1},
                   {'h1': 0.0710/25/10, 'h2': 0.0303/50*1},
                   {'h1': 0.0710/25*1, 'h2': 0.0303/50*10},
                   {'h1': 0.0710/25*1, 'h2': 0.0303/50/10},
                   ]

for new_params in new_params_list:
    new_params_dir = params_change(new_params)
    
    # Submit qsub request for single job:
    # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
    job_id = qsub.submit(
        dcc_repo_path + "scripts/get_training_eval_sets.py",
        conda_env="delta_env",
        args = [cell_class, None, new_params_dir,],
        hardware_requirements = dict(
            time_limit = 16, #2
            cores=2, #4
            gpus=1,
            mem_per_core=4,
            )
        )
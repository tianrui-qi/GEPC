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

#%% Submit scripts for simple activation cases

WORKERS=8
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie_simple',
                   'CcaSR_gillespie']

simple_params_list = [{'h1': 4e-2, 'h2': 1e-3},
                    {'h1': (4e-2)*10, 'h2': (1e-3)*10},
                    {'h1': (4e-2)/10, 'h2': (1e-3)/10},
                    {'h1': (4e-2)*1.5, 'h2': 1e-3},
                    {'h1': (4e-2)/1.5, 'h2': 1e-3},
                    ]

for cell_class in cell_class_list:
    for new_params in simple_params_list:
        new_params_dir = params_change(new_params)
        
        # Submit qsub request for single job:
        # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
        job_id = qsub.submit(
            dcc_repo_path + "scripts/get_training_eval_sets.py",
            conda_env="dcc_env_shared",
            args = [cell_class, WORKERS, new_params_dir,],
            hardware_requirements = dict(
                time_limit = 4, #2 (have used 16 for hard to simulate systems without parallel processing)
                cores=8, #4
                gpus=1,
                mem_per_core=4,
                )
            )

#%% Submit scripts for cascade

WORKERS=8
cell_class = 'CcaSR_Cascade'

cascade_params_list = [{'K_I': 45},
                   {'K_I': 45/2},
                   {'K_I': 45*1.5},
                    ]

for new_params in cascade_params_list:
    new_params_dir = params_change(new_params)
    
    # Submit qsub request for single job:
    # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
    job_id = qsub.submit(
        dcc_repo_path + "scripts/get_training_eval_sets.py",
        conda_env="dcc_env_shared",
        args = [cell_class, WORKERS, new_params_dir,],
        hardware_requirements = dict(
            time_limit = 4, #2 (have used 16 for hard to simulate systems without parallel processing)
            cores=8, #4
            gpus=1,
            mem_per_core=4,
            )
        )

#%% Submit scripts for feedforward

WORKERS=8
cell_class = 'CcaSR_FeedforwardPositive'

fold=1.5
ff_params_list = [{'K_I': 45, 'K_J': 45},
                   {'K_I': 45*fold, 'K_J': 45},
                   {'K_I': 45, 'K_J': 45*fold},
                   {'K_I': 45*fold, 'K_J': 45*fold},
                    ]

for new_params in ff_params_list:
    new_params_dir = params_change(new_params)
    
    # Submit qsub request for single job:
    # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
    job_id = qsub.submit(
        dcc_repo_path + "scripts/get_training_eval_sets.py",
        conda_env="dcc_env_shared",
        args = [cell_class, WORKERS, new_params_dir,],
        hardware_requirements = dict(
            time_limit = 4, #2 (have used 16 for hard to simulate systems without parallel processing)
            cores=8, #4
            gpus=1,
            mem_per_core=4,
            )
        )
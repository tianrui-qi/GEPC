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

# Make sure the proper package is used:
sys.path.insert(0,'./../')
import deepcellcontrol as dcc

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

#%% Scripts for simple activation cases, standard parameter values, stochastic

WORKERS=8
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie']

for cell_class in cell_class_list:
    # Submit qsub request for single job:
    # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
    job_id = qsub.submit(
        dcc_repo_path + "scripts/get_training_eval_sets.py",
        conda_env="dcc_env_shared",
        args = [cell_class, WORKERS,],
        hardware_requirements = dict(
            time_limit = 4, #2 (have used 16 for hard to simulate systems without parallel processing)
            cores=8, #4
            gpus=1,
            mem_per_core=4,
            )
        )
    
#%% Scripts for simple activation cases, standard parameter values, deterministic

WORKERS=8
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie']
new_params = {'eta': 1} # No new parameters, just need 3rd kwarg
solver = 'ode'

for do_camera_sim in [1, 0]:
    for cell_class in cell_class_list:
        new_params_dir = params_change(new_params)
        
        # Submit qsub request for single job:
        # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
        job_id = qsub.submit(
            dcc_repo_path + "scripts/get_training_eval_sets.py",
            conda_env="dcc_env_shared",
            args = [cell_class, WORKERS, new_params_dir, solver, do_camera_sim],
            hardware_requirements = dict(
                time_limit = 4, #4 (have used 16 for hard to simulate systems without parallel processing)
                cores=8, #8
                gpus=1,
                mem_per_core=4,
                )
            )

#%% Activation with diff E dynamics, same steady-state

WORKERS=8
cell_class = 'CcaSR_gillespie'

simple_params_list = [{'h1': (4e-2)*5, 'h2': (1e-3)*5},
                      {'h1': (4e-2)/5, 'h2': (1e-3)/5},
                    ]

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

#%% Cells have fixed diff responsiveness
      
WORKERS=8  
cell_class = 'CcaSR_gillespie_simple'

# new parameters - resampling should happen at the end
simple_params_list = [{'sigma': 2, 'resample_species': True},
                   {'sigma': 4, 'resample_species': True},
                   {'sigma': 8, 'resample_species': True},
                    ]

for new_params in simple_params_list:
    new_params_dir = params_change(new_params)
    
    # Submit qsub request for single job:
    # ARGS: cell_class (str), workers (int, default None), new_params_dir (str, optional)
    job_id = qsub.submit(
        dcc_repo_path + "scripts/get_training_eval_sets.py",
        conda_env="dcc_env_shared",
        args = [cell_class, WORKERS, new_params_dir,],
        hardware_requirements = dict(
            time_limit = 4, #4 (have used 16 for hard to simulate systems without parallel processing)
            cores=8, #8
            gpus=1,
            mem_per_core=4,
            )
        )

#%% Submit scripts for cascade

WORKERS=8
cell_class = 'CcaSR_Cascade'

# Find new parameters
K_I_new_list = [60/1.5, 60, 60*1.5]
K_I_new_list = [20, 40, 50, 70]
refcell = dcc.simulations.CcaSR_Cascade()

cascade_params_list = []
for K_I_new in K_I_new_list:
    
    H_ss = refcell.params['eta'] / refcell.params['nu'] 
    E_ss = refcell.params['h1'] / refcell.params['h2'] 
    I_ss_num = refcell.params['a'] * E_ss * H_ss**refcell.params['nh']
    I_ss_denom = refcell.params['nu'] * (refcell.params['K_H']**refcell.params['nh'] + H_ss**refcell.params['nh'])
    I_ss = I_ss_num / I_ss_denom
    a_update_num = (K_I_new**refcell.params['ni'] + I_ss**refcell.params['ni'])
    a_update_denom = (refcell.params['K_I']**refcell.params['ni'] + I_ss**refcell.params['ni'])
    a_update = a_update_num / a_update_denom
    
    new_params = {'K_I': K_I_new, 'a_F': a_update}
    cascade_params_list += [new_params]

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

# Solve for parameters
refcell = dcc.simulations.CcaSR_FeedforwardPositive()
starting_list = [{'K_I': 120, 'K_J': 120},
                   {'K_I': 120, 'K_J': 30},
                   {'K_I': 60, 'K_J': 120},
                   {'K_I': 60, 'K_J': 30},
                    ]

new_params_list = []
for starting in starting_list:
    
    K_I_new = starting['K_I']
    K_J_new = starting['K_J']
    
    eta = refcell.params['eta']
    nu = refcell.params['nu']
    h1 = refcell.params['h1']
    h2 = refcell.params['h2']
    K_H = refcell.params['K_H']
    nh = refcell.params['nh']
    K_I = refcell.params['K_I']
    ni = refcell.params['ni']
    K_J = refcell.params['K_J']
    nj = refcell.params['nj']
    a = refcell.params['a']
    
    H_ss = eta / nu
    E_ss = h1 / h2
    I_ss = (a * E_ss * H_ss**nh) / nu / (K_H**nh + H_ss**nh)
    
    J_ss = (a * E_ss * I_ss**ni) / nu / (K_I**ni + I_ss**ni)
    fac1 = I_ss**ni / (K_I**ni + I_ss**ni)
    fac2 = J_ss**nj / (K_J**nj + J_ss**nj)
    
    J_ss_new = (a * E_ss * I_ss**ni) / nu / (K_I_new**ni + I_ss**ni)
    fac1_new = I_ss**ni / (K_I_new**ni + I_ss**ni)
    fac2_new = J_ss_new**nj / (K_J_new**nj + J_ss_new**nj)
    
    a_update = (fac1 + fac2) / (fac1_new + fac2_new)
    
    new_params = {'K_J': K_J_new, 'K_I': K_I_new, 'a_F': a_update}
    new_params_list += [new_params]

for new_params in new_params_list:
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
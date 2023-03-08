#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:17:10 2023

@author: hklumpe
"""

import sys

sys.path.insert(0,'/project/dunlop/shared_python_packages/')
import qsub

# TODO: use different deepcellcontrol folder:
username='hklumpe'
dcc_data_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

# Submit qsub request for single job:
job_id = qsub.submit(
    dcc_repo_path + "scripts/get_training_eval_sets.py",
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 1, #2
        cores=8, #4
        gpus=0,
        mem_per_core=4,
        )
    )
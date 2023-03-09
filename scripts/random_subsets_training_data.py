#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:05:33 2023

@author: hklumpe
"""

import numpy as np
import glob
import os

username='hklumpe'
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

rng = np.random.default_rng(103122)

training_set_id = '*03-06_*'
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie_simple',
                   'CcaSR_gillespie']
subset_size_list = [1_000, 300, 100]
n_samples = 3

for cell_class in cell_class_list:
    data_dir = glob.glob(f'{dcc_repo_path}/assets/simulated/data/{cell_class}/{training_set_id}')[0]
    print(data_dir.split('/')[-1])
    
    # Load training data
    fluo1 = np.load(data_dir+'/training_set/fluo1.npy')
    stims = np.load(data_dir+'/training_set/stims.npy')
    
    for subset_size in subset_size_list:
        for i in range(n_samples):
            # Take a random subset and save it
            keep_idx = rng.choice(np.arange(np.shape(fluo1)[0]),
                                  size=subset_size,
                                  replace=False)
            
            output_dir = f'{data_dir}/training_set_{subset_size}_{i}'
            os.makedirs(output_dir, exist_ok=True)
            np.save(output_dir+'/fluo1.npy', fluo1[keep_idx])
            np.save(output_dir+'/stims.npy', stims[keep_idx])

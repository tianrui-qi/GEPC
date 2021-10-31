#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:06:54 2021

@author: jeanbaptiste
"""
import os
import pickle
import csv
import copy
from datetime import datetime

import qsub

import config as cfg


#%% Main
if __name__=='__main__':
    

    # Batch folder:
    # Create save folder:
    main_folder = cfg.models_path+'/batch_train_'+datetime.now(
        ).strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize training batch:
    rates = [1e-3, 1e-4, 1e-5]
    
    # Resume batch:
    # features_list, dataset, params = resume_batch(main_folder)
    args = []
    kwargs = []
    
    
    
    for r, rate in enumerate(rates):
        
        # Training parameters:
        params = copy.deepcopy(cfg.LSTM_params)
        params['training_parameters']['learning_rate']=rate
        params['epochs'] = 10,000
        params['patience'] = 3,000
        
        # Training features:
        params['features'] = ('fluos','stims','area')
        
        # Save files:
        save_folder = os.path.join(
            main_folder, 'task_%d'%(r+1)
            )
        os.makedirs(save_folder)
        params['save_folder']=save_folder
        params['logfile']=main_folder+'/log.csv'
        
        pickle_file = save_folder+'/parameters.pkl'
        with open(pickle_file,'wb') as f:
            pickle.dump(params, f)
        
        # Arguments to pass the script:
        args+=[[]]
        kwargs += [dict(parameters=pickle_file)]
        
    # Train and evaluate:
    job=qsub.submit(
        cfg.package_path+'/core/timeseries.py',
        args=args,
        kwargs=kwargs,
        job_array=True,
        conda_env='delta_env',
        hardware_requirements=dict(cores=4, time_limit=80)
        )
    print(job)
        
    
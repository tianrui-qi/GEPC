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

import numpy as np

import core
import config as cfg

# Features groups to test:
CORE_FEATURES = ('fluos','stims')
MORPHO_FEATURES = ('area',)
STIMS_FEATURES = ('left_stims','right_stims')
CHAMBER_FEATURES = ('chamber_median_fluo','chamber_std_fluo','cell_count')
IMAGE_FEATURES = ('sharpness',)


def init_batch(main_folder, dataset, params):

    # Create folder:
    os.makedirs(main_folder)
    
    # Generate features lists:
    features_lists = []
    for morpho in ((), MORPHO_FEATURES):
        for stims in ((), STIMS_FEATURES):
            for chamber in ((), CHAMBER_FEATURES):
                for image in ((), IMAGE_FEATURES):
                    features_lists += [CORE_FEATURES + morpho + stims + chamber + image]
    
    # Save features lists to be evaluated:
    with open(main_folder+'/features_lists.pkl','wb') as lists_file:
        pickle.dump(dict(features_lists=features_lists), lists_file)

    return features_lists



def resume_batch(main_folder):
    
    import ast
    
    # Load features lists that were already run:
    run_features = []
    with open(main_folder+'/log.csv', newline='') as filehandle:
        logger = csv.reader(filehandle)
        row_num = 0
        for row in logger:
            if row_num > 0:
                run_features += [ast.literal_eval(row[1])['features']]
            row_num+=1
    
    # Load complete features list:
    with open(main_folder+'/features_lists.pkl','rb') as lists_file:
        features_lists = pickle.load(lists_file)['features_lists']
    
    # Remove features that were already run:
    new_features = copy.copy(features_lists)
    for feature in features_lists:
        for run in run_features:
            if feature==run:
                new_features.remove(feature)
    
    # Load dataset:
    with open(main_folder+'/dataset.pkl','rb') as set_file:
        dataset = pickle.load(set_file)['dataset']

    # Load training parameters:
    with open(main_folder+'/training_parameters.pkl','rb') as params_file:
        params = pickle.load(params_file)['params']
    
    return new_features, dataset, params

#%% Main
if __name__=='__main__':
    
    import qsub
    # Batch folder:
    # Create save folder:
    main_folder = cfg.models_path+'/batch_train_'+datetime.now(
        ).strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize training batch:
    features_list = init_batch(main_folder)
    
    # Resume batch:
    # features_list, dataset, params = resume_batch(main_folder)
    args = []
    kwargs = []
    
    for features in features_list:
        
        # Format for command line:
        features_str = '"'
        for f in features:
            features_str += f+', '
        features_str=features_str[:-2]+'"'

        # Save folder:
        save_folder = os.path.join(
            main_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        os.makedirs(save_folder)
        
        args+=[[]]
        kwargs += [dict(
            features=features_str,
            save_folder=save_folder,
            logfile=main_folder+'/log.csv'
            )]
        
    # Train and evaluate:
    job=qsub.submit(
        cfg.package_path+'/core/timeseries.py',
        args=args,
        kwargs=kwargs,
        job_array=True,
        conda_env='delta_env',
        hardware_requirements=dict(time_limit=16)
        )
    print(job)
        
    
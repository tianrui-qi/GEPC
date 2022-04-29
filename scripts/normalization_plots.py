#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:23:34 2022

@author: jeanbaptiste
"""


import pickle

import numpy as np
import matplotlib.pyplot as plt

import deepcellcontrol as dcc


sets_folder = "/home/jeanbaptiste/data/shared_packages/deepcellcontrol/assets/data/experimental/"
training_files = (
    sets_folder + "2022-04-13_TrainingSet2_dataset.pkl",
    sets_folder + "2022-04-15_TrainingSet3_dataset.pkl",
    sets_folder + "2022-04-22_TrainingSet6_dataset.pkl",
    sets_folder + "2022-04-23_TrainingSet7_dataset.pkl",
    sets_folder + "2022-04-24_TrainingSet8_dataset.pkl",
    )

# Load data:
total_data = {}
for file in training_files:
    with open(file,"rb") as f:
        data = pickle.load(f)["raw_dataset"]
    
    for feature in data:
        if feature not in total_data:
            total_data[feature] = data[feature].flatten()
        else:
            total_data[feature] = np.concatenate(
                (total_data[feature], data[feature].flatten()),
                axis = 0
                )

# Normalize data with default parameters:
normalized = dcc.data.Normalization().normalize(total_data)

# Plot distributions:
for feature, data in total_data.items():
    
    # Raw hist:
    plt.hist(
        data, 
        bins = np.linspace(np.quantile(data,.001),np.quantile(data,.999),100),
        )
    plt.title(f"{feature}, raw")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.show()
    
    # Normalized:
    plt.hist(normalized[feature], np.linspace(0,1,100))
    plt.title(f"{feature}, normalized")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.show()
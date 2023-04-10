#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:18:43 2021

@author: jeanbaptiste
"""
import os
import time

# Default LSTM-MLP parameters:
defaults = dict(
    datasets_folder = os.path.dirname(__file__) + '/../assets/data/experimental/',
    training_sets=(
        "2022-04-13_TrainingSet2_dataset.pkl",
        "2022-04-19_TrainingSet4_dataset.pkl",
        "2022-04-22_TrainingSet6_dataset.pkl",
        "2022-04-24_TrainingSet8_dataset.pkl",
        ), # Training datasets
    eval_sets=(
        "2022-04-15_TrainingSet3_dataset.pkl",
        "2022-04-21_TrainingSet5_dataset.pkl",
        "2022-04-23_TrainingSet7_dataset.pkl",
        ), # Evaluation datasets
    models_folder = os.path.dirname(__file__) + '/../assets/models/', # Folder to save models to
    features=(
        'fluo1',
        'area',
        'sharpness',
        'cell_count',
        'chamber_mean_fluo1',
        'chamber_std_fluo1',
        'neighbor_stims',
        'stims'
        ), # Features to compile for training/control
    past_steps=36, # obsolete.
    horizon = 24, # Number of future time points to use for prediction horizon
    lstm_units = 64, # Number of LSTM units in the first encoder layer
    latent_dim=16, # Number of LSTM units in 2nd encoder layer
    output_mode = "timedistributed", # or "dense" # For LSTM decoder, obsolete.
    output_dim=1, # Number of features of decoder output
    batch_size=100, # Training batch size
    loss = "mse", # Loss to use for training
    learning_rate=1e-3, # Training learning rate
    dropout=0, # Obsolete
    mlp_layers = 5, # Number of MLP decoder layers
    mlp_dim = 32, # Units per MLP decoder layer
    training_parameters = dict(
        steps_per_epoch = 200,
        epochs=1000
        ), # Keras training parameters
    cnn_bins = 96, # Number of "bins" along the fluorescence axis
    cnn_filters = [16, 16, 32, 32, 16, 16, 8, 8], # Number of filters per convolutional layer
    save_folder=time.strftime("%Y-%m-%d_%H-%M-%S"), # Folder to save model, evaluation etc...
    centralized_records = "/project/dunlop/shared_python_packages/deepcellcontrol/assets/records.csv",
    )

# Feature types names (for normalization purposes):
fluo_features = (
    "fluos",
    "fluo",
    "fluo1",
    "fluo2",
    "fluo3",
    "chamber_mean_fluo",
    "chamber_median_fluo",
    "chamber_std_fluo",
    "chamber_mean_fluo1",
    "chamber_median_fluo1",
    "chamber_std_fluo1"
    )
length_features = ("length", "width")
count_features = ("cell_count",)
area_features = ("area","perimeter")
sharpness_features = ("sharpness",)
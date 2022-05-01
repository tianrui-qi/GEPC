#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:18:43 2021

@author: jeanbaptiste
"""
import os
import time

# Default LSTM parameters:
defaults = dict(
    datasets_folder = os.path.dirname(__file__) + '/../assets/data/experimental/',
    training_sets=(
        "2022-04-13_TrainingSet2_dataset.pkl",
        "2022-04-19_TrainingSet4_dataset.pkl",
        "2022-04-22_TrainingSet6_dataset.pkl",
        "2022-04-24_TrainingSet8_dataset.pkl",
        ),
    eval_sets=(
        "2022-04-15_TrainingSet3_dataset.pkl",
        "2022-04-21_TrainingSet5_dataset.pkl",
        "2022-04-23_TrainingSet7_dataset.pkl",
        ),
    models_folder = os.path.dirname(__file__) + '/../assets/models/',
    features=(
        'fluo1',
        'area',
        'sharpness',
        'cell_count',
        'chamber_mean_fluo1',
        'chamber_std_fluo1',
        'neighbor_stims',
        'stims'
        ),
    past_steps=36,
    horizon = 24,
    latent_dim=16,
    output_mode = "timedistributed", # or "dense"
    output_dim=1,
    batch_size=100,
    loss = "mse",
    learning_rate=1e-3,
    training_parameters = dict(
        steps_per_epoch = 200,
        epochs=200
        ),
    save_folder=time.strftime("%Y-%m-%d_%H-%M-%S"),
    logfile=None
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
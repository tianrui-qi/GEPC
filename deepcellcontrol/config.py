#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:18:43 2021

@author: jeanbaptiste
"""
from datetime import datetime

# General stuff:
package_path='/project/dunlop/shared_python_packages/deepcellcontrol/'


# Datasets:
datasets_path = '/projectnb/dunlop/deepcellcontrol/assets/data/'
datasets = dict(
    experimental_1=(
        datasets_path+'experimental/2021-09-20_Dataset_2/dataset.pkl',
        datasets_path+'experimental/2021-09-23_Dataset_3/dataset.pkl',
        datasets_path+'experimental/2021-09-28_Dataset_4/dataset.pkl',
        datasets_path+'experimental/2021-09-30_Dataset_5/dataset.pkl',
        )
    )

# Models:
models_path = '/projectnb/dunlop/deepcellcontrol/assets/models/'

# Default LSTM parameters:
LSTM_params = dict(
    datasets=datasets['experimental_1'],
    features=('fluos','stims'),
    past_steps=36,
    horizon = 24,
    latent_dim=64,
    training_parameters = dict(
        learning_rate=1e-4,
        patience = 500,
        steps_per_epoch = 200,
        epochs=2000
        ),
    save_folder=models_path+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    logfile=None
    )

MLP_params = dict(
    datasets=datasets['experimental_1'],
    features=('fluos','stims'),
    past_steps=36,
    horizon=24,
    hidden_layers=10, 
    hidden_dim=64,
    dropout=0,
    training_parameters = dict(
        learning_rate=1e-3,
        patience = 500,
        steps_per_epoch = 200,
        epochs=2000
        ),
    save_folder=models_path+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    logfile=None
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to training and evaluating time-series forecasting networks

@author: jeanbaptiste
"""
import os
import json
import pickle
import csv
from datetime import datetime

import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from . import utilities as utils
from . import config as cfg

def train(
        dataset,
        network,
        batch_size=400,
        epochs=2000,
        steps_per_epoch=250,
        patience=50,
        learning_rate=1e-3,
        save_folder='./'
        ):

    # Adjust learning rate:
    network.optimizer.learning_rate = learning_rate
    
    # Make sure dataset is properly setup:
    dataset.mode = 'training'
    dataset.batch_size=batch_size

    # Initialize network
    model_checkpoint = ModelCheckpoint(
        os.path.join(save_folder,'model.hdf5'), 
        monitor='loss',
        verbose=1, 
        save_best_only=True
        )
    early_stopping = EarlyStopping(
        monitor='loss', mode='min', verbose=1, patience=patience
        )
    evaluation_clbk = EvaluationCallback(dataset)

    # train network
    history = network.fit_generator(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint, evaluation_clbk, early_stopping]
        )
    network.load_weights(os.path.join(save_folder,'model.hdf5'))
    
    # Add rmse and mae to history metrics a posteriori:
    history.history['metrics'] = evaluation_clbk.metrics
    
    return network, history


def evaluate(
        dataset,
        network,
        batch_size=10_000,
        num_batches=10,
        verbose = 1,
        return_eval = False
        ):
    
    # Set up dataset:
    dataset.mode='evaluation'
    dataset.batch_size=batch_size
    
    # Get batches and compile evaluation:
    rmse = mae = 0
    for _ in range(num_batches):
        
        # get X and Y data:
        xval, yval = next(dataset)
        yhat = network.predict(xval,verbose=verbose)
        
        # Compile error metrics:
        rmse = rmse + np.sqrt(np.mean(np.square(yhat-yval[:,:,0]),axis=0))  # RMSE over prediction horizon
        mae = mae + np.mean(np.abs(yhat-yval[:,:,0]),axis=0) # MAE over prediction horizon
    
    rmse/=num_batches
    mae/=num_batches
    
    if verbose:
        print("RMSE = %g, MAE = %g"%(np.mean(rmse),np.mean(mae)))
    
    if return_eval:
        return dict(rmse=rmse, mae=mae), dict(input=xval, groundtruth=yval, prediction=yhat)
    return dict(rmse=rmse, mae=mae)


def batch_train_eval(dataset, network, params):

    # Write training parameters to disk:
    with open(params['save_folder']+'/training_parameters.txt','w') as params_file:
        json.dump(params, params_file)

    # Update dataset's features:
    dataset.features = params['features']
    dataset.past_steps = params['past_steps']
    dataset.horizon = params['horizon']
    dataset.save_state(params['save_folder']+'/dataset_state.pkl')
    
    # Train:
    network, history = train(
        dataset, 
        network,
        save_folder = params['save_folder'],
        **params['training_parameters']
        )
    
    # Plot history:
    metrics_lin = lambda name : [np.mean(x[name]) for x in history.history['metrics']]
    plt.semilogy(history.history['loss'],label='$loss (MSE)$')
    plt.semilogy(np.sqrt(history.history['loss']),label='$\sqrt{loss}$')
    plt.semilogy(metrics_lin('mae'),label='$validation MAE$')
    plt.semilogy(metrics_lin('rmse'), label='$validation RMSE$')
    plt.grid(axis='y',which='both')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss and validation')
    plt.legend()
    plt.savefig(os.path.join(params['save_folder'],'training_loss.png'),dpi=300)
    plt.savefig(os.path.join(params['save_folder'],'training_loss.svg'),dpi=300)
    # plt.show()
    plt.clf()
    
    # Evaluate:
    metrics, eval_d = evaluate(dataset, network, return_eval=True)

    # Plot evaluation:
    plt.plot(metrics['mae'])
    plt.plot(metrics['rmse'])
    plt.grid(axis='y',which='both')
    plt.xlabel('horizon')
    plt.ylabel('error (a.u.)')
    plt.title('evaluation MAE & RMSE')
    plt.legend(('MAE','RMSE'))
    plt.savefig(os.path.join(params['save_folder'],'evaluation_error.png'),dpi=300)
    plt.savefig(os.path.join(params['save_folder'],'evaluation_error.svg'),dpi=300)
    # plt.show()
    plt.clf()
    
    # Plot single cell evaluations:
    os.makedirs(os.path.join(params['save_folder'],'single_cell_evals'))
    fluos, stims = dataset.batch_reconstruct(eval_d['input'],eval_d['groundtruth'])
    for eval_num in range(50):
        utils.evaluationPlot(
            stims[eval_num],
            fluos[eval_num,:params['past_steps']],
            fluos[eval_num,params['past_steps']:],
            eval_d["prediction"][eval_num],
            dyn_range=1,
            savefig = os.path.join(params['save_folder'],'single_cell_evals','sample_%02d'%eval_num),
            show = False
            )
    
    # Save relevant training data to pickle file:
    with open(params['save_folder']+'/training_output.pkl','wb') as res_file:
        pickle.dump(
            dict(
                params=history.params, 
                history=history.history,
                **metrics
                ),
            res_file
            )
    
    # Append information to CSV log file:
    csv_log_training(
        params['logfile'],
        dict(
            folder=params['save_folder'],
            parameters=str(params),
            epochs=history.epoch[-1],
            loss=min(history.history['loss']),
            rmse=np.mean(metrics['rmse']),
            mae=np.mean(metrics['mae'])
            )
        )
    
    return network


def csv_log_training(logfile, row):
    
    if logfile is not None:
        if not os.path.exists(logfile):
            # Init CSV file:
            with open(logfile, 'w', newline='') as filehandle:
                logger = csv.DictWriter(filehandle, fieldnames=list(row.keys()))
                logger.writeheader()
        with open(logfile, 'a', newline='') as filehandle:
            logger = csv.DictWriter(filehandle,fieldnames=list(row.keys()))
            logger.writerow(row)


class EvaluationCallback(Callback):
    
    def __init__(self, dataset):
        super(Callback, self).__init__()
        self._dataset = dataset
        self.metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        
        # Record dataset batch size:
        batch_size = self._dataset.batch_size
        
        # Evaluate RMSE and MAE and record:
        print('Evaluation: ',end='')
        self.metrics += [evaluate(self._dataset, self.model, num_batches=1)]
        
        # Re-set dataset to training parameters:
        self._dataset.batch_size = batch_size
        self._dataset.mode = 'training'

def _inputs_processor(kwargs):
    
    import copy
    params = copy.deepcopy(cfg.LSTM_params)
    
    for k, v in kwargs.items():
        
        if k=='parameters':
            with open(v,'rb') as f:
                params=pickle.load(f)
        
        if k in ('features', 'datasets'):
            params[k] = tuple(v.split(','))
            
        if k in ('save_folder','logfile'):
            params[k] = v
    
        if k in ('past_steps', 'horizon', 'latent_dim'):
            params[k] = int(v)
    
    return params
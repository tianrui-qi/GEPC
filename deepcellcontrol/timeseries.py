#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to training and evaluating time-series forecasting networks

@author: jeanbaptiste
"""
import os
import json
import pickle

import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from . import utilities as utils

def train(
        dataset,
        network,
        batch_size=400,
        epochs=2000,
        steps_per_epoch=250,
        patience=1000,
        learning_rate=1e-3,
        save_folder='./',
        evaluation_dataset = None
        ):
    """
    Train forecasting network

    Parameters
    ----------
    dataset : data.Dataset
        Dataset for training.
    network : tensorflow.keras.models.Model
        The network to train.
    batch_size : int, optional
        Training batch size. The default is 400.
    epochs : int, optional
        Number of training epochs. The default is 2000.
    steps_per_epoch : int, optional
        Number of steps / batches per epoch. The default is 250.
    patience : int, optional
        The number of epochs without loss decrease that will stop learning 
        before the total number of epochs is reached. See keras's earlystopping
        The default is 50.
    learning_rate : TYPE, optional
        Network learning rate. The default is 1e-3.
    save_folder : str, optional
        The path under which the model should be saved. 
        The default is './'.
    evaluation_dataset : data.Dataset, optional
        Dataset for evaluation. Test partition must be > 0. If None, the test 
        partition of the training dataset will be used instead. If training 
        test partition is 0, no on-line evaluation is performed. The default is
        None.

    Returns
    -------
    network : tensorflow.keras.models.Model
        Trained network/model.
    history : tensorflow.History object
        Record of training loss values and metrics values
        at successive epochs.

    """

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
    callbacks = [model_checkpoint, early_stopping]
    
    # Evaluation callback:
    if evaluation_dataset is not None:
        if evaluation_dataset.test_ratio == 0:
            raise ValueError(
                "Evaluation dataset test ratio must be greater than 0"
                )
        evaluation_clbk = EvaluationCallback(evaluation_dataset, save_folder+'/model_besteval.hdf5')
        callbacks.append(evaluation_clbk)
    elif dataset.test_ratio > 0:
        evaluation_clbk = EvaluationCallback(dataset, save_folder+'/model_besteval.hdf5')
        callbacks.append(evaluation_clbk)
    else:
        evaluation_clbk = None
    

    # train network
    history = network.fit_generator(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks
        )
    network.load_weights(os.path.join(save_folder,'model.hdf5'))
    
    # Add rmse and mae to history metrics a posteriori:
    if evaluation_clbk is not None:
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
    """
    Evaluate model after training

    Parameters
    ----------
    dataset : data.Dataset
        Dataset for evaluation.
    network : tensorflow.keras.models.Model
        The trained network.
    batch_size : int, optional
        Evaluation batch size. The default is 10_000.
    num_batches : int, optional
        Number of batches to evaluate. The default is 10.
    verbose : int, optional
        Verbosity level. The default is 1.
    return_eval : bool, optional
        Whether to return the last batch evaluation results. 
        The default is False.

    Returns
    -------
    dict or Tuple of (dict, dict)
        RMSE and MAE for each evaluation and, optionally, the last batch to
        be evaluated.

    """
    
    # Set up dataset:
    dataset.mode='evaluation'
    dataset.batch_size=batch_size
    
    # Get batches and compile evaluation:
    rmse = mae = 0
    for _ in range(num_batches):
        
        # get X and Y data:
        xval, yval = next(dataset)
        yval = np.squeeze(yval)
        yhat = network.predict(xval,verbose=verbose)
        yhat = np.squeeze(yhat)
        
        # Compile error metrics:
        rmse = rmse + np.sqrt(np.mean(np.square(yhat-yval),axis=0))  # RMSE over prediction horizon
        mae = mae + np.mean(np.abs(yhat-yval),axis=0) # MAE over prediction horizon
    
    rmse/=num_batches
    mae/=num_batches
    
    if verbose:
        print("RMSE = %g, MAE = %g"%(np.mean(rmse),np.mean(mae)))
    
    if return_eval:
        return dict(rmse=rmse, mae=mae), dict(input=xval, groundtruth=yval, prediction=yhat)
    return dict(rmse=rmse, mae=mae)


def batch_train_eval(
        dataset,
        network,
        params,
        plot_singlecell = True,
        plot_autoencoding = False,
        evaluation_dataset = None
        ):
    """
    Train and evaluate network, and create and save post-training plots

    Parameters
    ----------
    dataset : data.Dataset
        Training dataset.
    network : tensorflow.keras.models.Model
        The network to train.
    params : dict
        Dict of training parameters as kwargs. See train() and config.py for
        more information.
    plot_singlecell : bool, optional
        Plot single-cell fluorescence predictions. The default is True.
    plot_autoencoding : bool, optional
        Plot single-cell autoencoder results. The default is False.
    evaluation_dataset : data.Dataset
        Dataset to use for evaluation. If None, the test partition of the 
        training dataset will be used. The default is None.

    Returns
    -------
    network : tensorflow.keras.models.Model
        The trained network.

    """
    
    save_folder = params["models_folder"] + params["save_folder"]
    os.makedirs(save_folder, exist_ok = True)

    # Write training parameters to disk:
    with open(save_folder+'/training_parameters.json','w') as params_file:
        json.dump(params, params_file, indent=4)

    # Update dataset's features:
    dataset.features = params['features']
    dataset.past_steps = params['past_steps']
    dataset.horizon = params['horizon']
    dataset.save_state(save_folder+'/dataset_state.pkl')
    
    # Train:
    network, history = train(
        dataset, 
        network,
        save_folder = save_folder,
        evaluation_dataset=evaluation_dataset,
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
    plt.savefig(os.path.join(save_folder,'training_loss.png'),dpi=300)
    plt.savefig(os.path.join(save_folder,'training_loss.svg'),dpi=300)
    # plt.show()
    plt.clf()
    
    # Evaluate:
    if evaluation_dataset is None:
        metrics, eval_d = evaluate(dataset, network, return_eval=True)
    else:
        metrics, eval_d = evaluate(evaluation_dataset, network, return_eval=True)

    # Plot evaluation:
    plt.plot(metrics['mae'])
    plt.plot(metrics['rmse'])
    plt.grid(axis='y',which='both')
    plt.xlabel('horizon')
    plt.ylabel('error (a.u.)')
    plt.title('evaluation MAE & RMSE')
    plt.legend(('MAE','RMSE'))
    plt.savefig(os.path.join(save_folder,'evaluation_error.png'),dpi=300)
    plt.savefig(os.path.join(save_folder,'evaluation_error.svg'),dpi=300)
    # plt.show()
    plt.clf()
    
    # Plot single cell evaluations:
    if plot_singlecell:
        os.makedirs(os.path.join(save_folder,'single_cell_evals'), exist_ok=True)
        fluos, stims = dataset.formatter.reconstruct(eval_d['input'],eval_d['groundtruth'])
        if type(params['past_steps']) is int:
            _past_steps = params['past_steps']
        else:
            _past_steps = params['past_steps'][1]
        for eval_num in range(50):
            utils.evaluationPlot(
                stims[eval_num],
                fluos[eval_num,:_past_steps],
                fluos[eval_num,_past_steps:],
                eval_d["prediction"][eval_num],
                dyn_range=1,
                savefig = os.path.join(save_folder,'single_cell_evals','sample_%02d'%eval_num),
                show = False
                )
    
    if plot_autoencoding:
        os.makedirs(os.path.join(save_folder,'autoencoding_evals'))
        for eval_num in range(50):
            utils.plot_autoencoding(
                eval_d['groundtruth'][eval_num],
                eval_d["prediction"][eval_num],
                features_list = dataset.features,
                savefig = os.path.join(
                    save_folder,
                    f'autoencoding_evals/sample_{eval_num:06d}'
                    )
                )
    
    # Save relevant training data to pickle file:
    with open(save_folder+'/training_output.pkl','wb') as res_file:
        pickle.dump(
            dict(
                params=history.params, 
                history=history.history,
                **metrics
                ),
            res_file
            )
    
    return network


class EvaluationCallback(Callback):
    """
    Callback object for keras training.
    This custom Callback runs the evaluate() function in this module at the
    end of each epoch.
    
    See https://www.tensorflow.org/guide/keras/custom_callback
    """
    
    def __init__(self, dataset, savefile=None):
        """
        Instanciate

        Parameters
        ----------
        dataset : data.Dataset
            Dataset to use for evaluation.
        savefile : str, optional
            Path to save the model that had the best evaluation result so far.
            The default is None.

        Returns
        -------
        None.

        """
        super(Callback, self).__init__()
        self._dataset = dataset
        self.metrics = []
        self.savefile = savefile
        self._best = np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Evaluate predictions RMSE on epoch end

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : Dict, optional
            see tf.keras.callbacks.Callback

        Returns
        -------
        None.

        """
        
        # Record dataset batch size:
        batch_size = self._dataset.batch_size
        
        # Evaluate RMSE and MAE and record:
        self.metrics += [evaluate(self._dataset, self.model, num_batches=1)]
        
        # Re-set dataset to training parameters:
        self._dataset.batch_size = batch_size
        self._dataset.mode = 'training'
        
        self.save_best()

    def save_best(self):
        """
        Save current model if evaluation was better than previous epochs.

        Returns
        -------
        None.

        """
        
        if self.savefile is None:
            return
        
        rmse = np.mean(self.metrics[-1]["rmse"])
        if self._best > rmse:
            print(
                f"Evaluation rmse improved from {self._best:.4g} to {rmse:.4g}, saving to {self.savefile}"
                )
            self._best = rmse
            self.model.save(self.savefile)
        else:
            print(f"Evaluation rmse did not improve from {self._best:.4g}")
            
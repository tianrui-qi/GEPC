#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:46:40 2021

@author: jeanbaptiste
"""
import os
from collections import Generator
import pickle

import numpy as np

# Normalization parameters:
DYN_RANGE = 4095
LENGTH_NORM_FACTOR = 200
AREA_NORM_FACTOR = 3_000
COUNT_NORM_FACTOR = 20
SHARP_NORM_FACTOR = 2_000
SHARP_NORM_SHIFT = 9_000
FLUO_FEATURES = (
    "fluos",
    "fluo",
    "fluo1",
    "fluo2",
    "fluo3",
    "chamber_mean_fluo",
    "chamber_median_fluo",
    "chamber_std_fluo"
    )
LENGTH_FEATURES = ("length",)
COUNT_FEATURES = ("cell_count",)
AREA_FEATURES = ("area",)
SHARPNESS_FEATURES = ("sharpness",)

# Linear normalizatino functions:
fluo_norm = lambda x: x / DYN_RANGE
# invert exp normalization functions:
length_norm = lambda x: 1 - np.power(10.0, -x / LENGTH_NORM_FACTOR)
area_norm = lambda x: 1 - np.power(10.0, -x / AREA_NORM_FACTOR)
cell_count_norm = lambda x: 1 - np.power(10.0, -x / COUNT_NORM_FACTOR)
# Sigmoid normalization functions:
sharp_norm = lambda x: 1 / (1 + np.exp(-(x - SHARP_NORM_SHIFT) / SHARP_NORM_FACTOR))


class Datasets(Generator):
    """
    Handle datasets for training and evaluation of NNs. Inherits from the
    Generator class so it can be fed directly to TF's .fit() function
    """

    def __init__(self, datasets, features):
        """
        Instanciate

        Parameters
        ----------
        datasets : List of str
            List of path to dataset pickle files.
        features : List of str
            List of features to use for training.
        test_ratio : float, optional
            Ratio of data to save for testing. The default is 0.1.

        Returns
        -------
        None.

        """
        self.datasets = datasets
        self.features = features
        self.test_ratio = 0.1
        self.data_type = "raw_dataset"
        self.mode = "training"
        self.format_mode = "lstm"
        self.horizon = 24
        self.past_steps = 36
        self.batch_size = 100

    def load(self):
        """Load up data from the datasets"""

        self.reload()

        # Compute set partitioning:
        self._cumsum_cells = np.cumsum(self.cells)
        self._evaluation_set = np.random.choice(
            a=self._cumsum_cells[-1],
            size=int(self._cumsum_cells[-1] * self.test_ratio),
            replace=False,
        )
        self._training_set = np.setdiff1d(
            np.arange(self._cumsum_cells[-1]), self._evaluation_set
        )
    
    def reload(self):
        '''Re-load data without touching partitioning'''
        
        self.data = []
        self.cells = []

        for dataset in self.datasets:
            if os.path.isdir(dataset):
                self._load_npy(dataset)
            else:
                self._load_pkl(dataset)

    def _load_pkl(self, dataset):
        """Load from pickle dataset file"""

        with open(dataset, "rb") as file:
            self.data += [pickle.load(file)]
            self.cells += [self.data[-1][self.data_type]["stims"].shape[0]]

    def _load_npy(self, dataset):
        """Load from numpy dataset files in directory"""

        self.data += [{self.data_type: dict()}]
        for file in os.listdir(dataset):
            if file.endswith(".npy"):
                key = os.path.splitext(file)[0]
                self.data[-1][self.data_type][key] = np.load(
                    os.path.join(dataset, file)
                )
        self.cells += [self.data[-1][self.data_type]["stims"].shape[0]]
    
    def save_state(self, filename):
        '''Save dataset state (ie all except actual data)'''
        
        # Save all fields except data:
        with open(filename, "wb") as f:
            pickle.dump({k:v for k,v in self.__dict__.items() if k!='data'}, f)
    
    def load_state(self, filename):
        '''Load dataset state (need to run self.reload() afterwards'''
        
         # Load and set fields:
        with open(filename, "wb") as f:
            state = pickle.load(f)
        
        for k, v in state:
            setattr(self, k, v)

    def normalize(self):
        """Normalize data loaded from raw datasets using normalization fcn below"""

        for set_number, dataset in enumerate(self.data):
            if "raw_dataset" not in dataset:
                raise RuntimeError(
                    "Raw values not loaded for dataset #%d: %s"
                    % (set_number, self.datasets[set_number])
                )
            dataset["normalized_dataset"] = normalization(dataset["raw_dataset"])

    def get_cell(self, set_number, cell_number):
        """Get features for specific cell"""

        cell = []
        for feature in self.features:
            cell += [
                np.squeeze(
                    self.data[set_number][self.data_type][feature][cell_number, :]
                )
            ]

        return np.moveaxis(
            np.array(cell), (0, 1), (1, 0)
        )  # axis 0 = time, axis 1 = features

    def get_random_cell(self):
        """Get one random cell from the datasets"""

        # Get random cell from proper partition:
        if self.mode == "training":
            cell_nb = np.random.choice(self._training_set)
        else:
            cell_nb = np.random.choice(self._evaluation_set)

        # Break down set/cell numbers:
        for set_nb, cumsum_cells in enumerate(self._cumsum_cells):
            if cumsum_cells > cell_nb:
                if set_nb > 0:
                    cell_nb -= self._cumsum_cells[set_nb - 1]
                break

        # Return cell:
        return self.get_cell(set_nb, cell_nb)

    def get_random_windows(self, cell):
        """Get features over random time windows for cell (past & future)"""

        timepoint = np.random.randint(
            self.past_steps, cell.shape[0] - self.horizon
        )  # Get random "time points"
        # "Past":
        past = cell[timepoint - self.past_steps : timepoint, :]
        # "Future":
        future = cell[timepoint : timepoint + self.horizon, :]

        return past, future

    def batch_format(self, past, future):
        """Format data to neural network i/o specs"""

        # TODO: MLP mode
        if self.format_mode == "lstm":
            X = [past, future[:, :, [feature == "stims" for feature in self.features]]]
            Y = future[:, :, [feature == "fluos" for feature in self.features]]
        
        if self.format_mode == "mlp":
            X = np.concatenate(
                np.reshape(
                    past,
                    shape = (past.shape[0], past.shape[1]*past.shape[2]),
                    order='C'
                    ), 
                future[:, :, [feature == "stims" for feature in self.features]],
                axis=1
                )
            Y = future[:, :, [feature == "fluos" for feature in self.features]]

        return X, Y
    
    def batch_reconstruct(self, X, Y):
        
        # TODO: MLP mode
        if self.format_mode == "lstm":
            fluos = np.concatenate(
                (
                    np.squeeze(X[0][:,:,[feature == "fluos" for feature in self.features]]),
                    np.squeeze(Y)
                    ),
                axis=1
                )
            stims = np.concatenate(
                (
                    np.squeeze(X[0][:,:,[feature == "stims" for feature in self.features]]),
                    np.squeeze(X[1])
                    ),
                axis=1
                )
        
        if self.format_mode == "mlp":
            fluos_ind = [f for f, feature in enumerate(self.features) if feature == "fluos"][0]
            stims_ind = [f for f, feature in enumerate(self.features) if feature == "stims"][0]
            fluos = np.concatenate(
                (
                    X[:,fluos_ind:fluos_ind+self.past_steps],
                    Y
                    ),
                axis=1
                )
            stims = np.concatenate(
                (
                    X[:,stims_ind:stims_ind+self.past_steps],
                    X[:,self.horizon:],
                    ),
                axis=1
                )
        
        return fluos, stims

    def get_batch(self):
        """Get batch of X, Y data to feed into NN"""

        past = np.empty(
            (self.batch_size, self.past_steps, len(self.features)), dtype=np.float32
        )
        future = np.empty(
            (self.batch_size, self.horizon, len(self.features)), dtype=np.float32
        )

        for batch_ind in range(self.batch_size):
            cell = self.get_random_cell()
            past[batch_ind], future[batch_ind] = self.get_random_windows(cell)

        X, Y = self.batch_format(past, future)

        return X, Y

    def send(self, ignored_arg):
        """Generator function (called by next())"""

        batch = self.get_batch()

        return batch

    def throw(self, type=None, value=None, traceback=None):
        """Generator function for end of iterations"""
        raise StopIteration


def normalization(raw):
    """Normalize data for raw dataset"""

    normalized = dict()
    for key in raw:

        normalized[key] = np.copy(raw[key]).astype(np.float32)
        normalized[key][np.isnan(normalized[key])] = 0

        if key in FLUO_FEATURES:
            normalized[key] = fluo_norm(normalized[key])

        if key in LENGTH_FEATURES:
            normalized[key] = length_norm(normalized[key])

        if key in COUNT_FEATURES:
            normalized[key] = cell_count_norm(normalized[key])
        
        if key in AREA_FEATURES:
            normalized[key] = area_norm(normalized[key])
        
        if key in SHARPNESS_FEATURES:
            normalized[key] = sharp_norm(normalized[key])

    return normalized

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:46:40 2021

@author: jeanbaptiste
"""
import os
import time
from collections.abc import Generator
import pickle

import numpy as np
import matplotlib.pyplot as plt

from . import utilities as utils
from . import config

class Normalization():
    """
    This class is dedicated to normalization of timeseries values.
    The models perform best if the data is normalized to the [0,1] range.
    This class is used here by the Datasets class, but should also be used 
    when running on-the-fly control experiments. The normalization factors are
    values that make sense based on the distribution of our data, but should
    be adapted to different use cases.
    """
    
    def __init__(self):
        
        # Normalization parameters:
        self.fluo_max = 4095
        "Max possible fluorescence value"
        self.length_factor = 200
        "Cell length scaling. factor = 0.9 value cutoff in inv exp norm"
        self.area_factor = 3_000
        "Cell area scaling. factor = 0.9 value cutoff in inv exp norm"
        self.count_factor = 9
        "Cells count scaling. factor = 0.9 value cutoff in inv exp norm"
        self.sharpness_factor = 1
        "Image sharpness factor. See sigmoid_normalization method"
        self.sharpness_shift = 3.6
        "Image sharpness mid-range shift. See sigmoid_normalization method"
    
    def normalize(self, data):
        """
        Apply normalization to data / dataset

        Parameters
        ----------
        data : dict
            Dictionary of features and their values. The values can be scalars
            or 1D vectors.

        Returns
        -------
        normalized : dict
            Normalized data.

        """
        
        normalized = {}
        
        for feature, values in data.items():
            
            if feature in config.fluo_features:
                values = self.fluo(values)
            
            if feature in config.length_features:
                values = self.length(values)
            
            if feature in config.area_features:
                values = self.area(values)
            
            if feature in config.count_features:
                values = self.count(values)
            
            if feature in config.sharpness_features:
                values = self.sharpness(values)
            
            normalized[feature] = values
        
        return normalized
    
    def linear_normalization(self, values, vmin, vmax):
        """
        Simple linear normalization function. 

        Parameters
        ----------
        values: array_like
            The values to normalize
        vmin : int or float
            The minimum of the inputs. Values below vmin will be clipped.
        vmax : int or float
            The maximum of the inputs. Values above vmax will be clipped.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = np.clip(values, a_min=vmin, a_max=vmax).astype(np.float32)
        values = (values-vmin)/(vmax-vmin)
        
        return values
    
    def invert_exp_normalization(self, values, factor, vmin=None):
        """
        Inverted exponential normalization (saturating function)
        y = 1 - 10^{-x/factor}

        Parameters
        ----------
        values : array_like
            The values to normalize.
        factor : int or float
            Normalization exponent. Values below factor will end up in
            the range [0, 0.9[, whereas values above factor will end up in
            the ]0.9, 1] range
        vmin : int, float or None, Optional.
            The minimum of the inputs. Values below vmin will be clipped, and
            the values will be shifted by -vmin. factor will NOT be shifted. If
            None, the values are not clipped or shifted.
            The default is None.

        Returns
        -------
        values: numpy array
            Normalized values.
            
        """
        
        if vmin is not None:
            values = np.clip(values, a_min = vmin)-vmin
        values = 1 - np.power(10.0, -values / factor)
        
        return values
    
    def sigmoid_normalization(self, values, factor, shift):
        """
        Sigmoidal normalization.
        y = 1 / (1 + e^{-(x-shift)/factor})
        y[shift-3*factor]   -> ~0.05
        y[shift-factor]     -> ~0.27
        y[shift]            -> 0.5
        y[shift+factor]     -> ~0.73
        y[shift+3*factor]   -> ~0.95

        Parameters
        ----------
        values : array_like
            The values to normalize.
        factor : int or float
            Normalization exponent.
        shift : int or float
            Center of the sigmoid.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        values = 1 / (1 + np.exp(-(values - shift) / factor))
        
        return values
    
    def fluo(self, values):
        """
        Fluorescence normalization function (linear)

        Parameters
        ----------
        values : array_like
            The values to normalize.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = self.linear_normalization(values, vmin=0, vmax=self.fluo_max)
        
        return values
    
    def length(self, values):
        """
        Length normalization function (inverted exp)

        Parameters
        ----------
        values : array_like
            The values to normalize.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = self.invert_exp_normalization(values, self.length_factor)
        
        return values
    
    def area(self, values):
        """
        Area normalization function (inverted exp)

        Parameters
        ----------
        values : array_like
            The values to normalize.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = self.invert_exp_normalization(values, self.area_factor)
        
        return values
    
    def count(self, values):
        """
        Cell count per chamber normalization function (inverted exp)

        Parameters
        ----------
        values : array_like
            The values to normalize.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = self.invert_exp_normalization(values, self.count_factor)
        
        return values
    
    def sharpness(self, values):
        """
        Image sharpness normalization function (sigmoid)

        Parameters
        ----------
        values : array_like
            The values to normalize.

        Returns
        -------
        values: numpy array
            Normalized values.

        """
        
        values = self.sigmoid_normalization(
            values, self.sharpness_factor, self.sharpness_shift
            )
        
        return values

class AbstractFormatter():
    """
    Abstract class to format past (& potentially future) data for training,
    evaluation, or feedback control
    """
    
    def __init__(self, features):
        self.features = features
        "List of features to compile as input (in addition to 'stims')"
    
    def training(self, past, future):
        """
        Formatting method when in training or evaluation mode

        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)
        future : 3D numpy array
            Future datapoints. Dimensions are (cells, horizon, features)

        Returns
        -------
        None.

        """
        pass
    
    def control(self, past):
        """
        Formatting method when in control mode
        
        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)
        future : 3D numpy array
            Future datapoints. Dimensions are (cells, horizon, features)

        Returns
        -------
        None.

        """
        pass
    

class LSTMFormatter(AbstractFormatter):
    """
    Formatter for the LSTM neural network
    """
    
    def training(self, past, future):
        """
        Formatting method when in training or evaluation mode

        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)
        future : 3D numpy array
            Future datapoints. Dimensions are (cells, horizon, features)

        Returns
        -------
        X : List of 2 3D numpy arrays
            The inputs to the LSTM network. Dimensions are 
            [(cells, past_steps, features) and (cells, horizon, 1)]
        Y : 3D numpy array
            The groundtruth for the LSTM network. Dimensions are
            (cells, horizon, 1)

        """
        
        X = [self.control(past), future[:, :, self.features.index("stims")]]
        Y = future[:, :, [feature in ("fluos", "fluo1") for feature in self.features]]
        
        return X, Y
    
    def control(self, past):
        """
        Formatting method when in control mode
        
        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)

        Returns
        -------
        X : 3D numpy array
            The inputs to an LSTMMPC controller. Dimensions are
            (cells, past_steps, features)

        """
        
        return past
    
    def reconstruct(self, X, Y):
        """
        Reconstruct original fluorescence and stimulations time-series from
        X, Y pair

        Parameters
        ----------
        X : List of 2 3D numpy arrays
            The inputs to the LSTM network. Dimensions are 
            [(cells, past_steps, features) and (cells, horizon, 1)]
        Y : 3D numpy array
            The groundtruth for the LSTM network. Dimensions are
            (cells, horizon, 1)

        Returns
        -------
        fluos : 2D numpy array
            Reconstructed fluorescence trajectories. Dimensions are 
            (cells, past_steps + horizon)
        stims : 2D numpy array
            Reconstructed stimulations. Dimensions are 
            (cells, past_steps + horizon)

        """
        
        fluo_ind = [f for f, feature in enumerate(self.features) if feature in ("fluos", "fluo1")][0]
        
        fluos = np.concatenate((X[0][:,:,fluo_ind], np.squeeze(Y[:,:])), axis=1)
        stims = np.concatenate(
            (X[0][:,:,self.features.index("stims")],X[1]),axis=1
            )
        
        return fluos, stims

class LSTMAutoencoderFormatter(LSTMFormatter):
    
    def training(self, past, future):
        
        X, _ = super().training(past, future)
        
        X = [X[0],np.zeros(shape=X[0].shape[0:2]+(1,),dtype=np.float32)]
        
        return X, X[0]



class MLPFormatter(AbstractFormatter):
    """
    Formatter for the MLP prediction network
    """
    
    def training(self, past, future):
        """
        Formatting method when in training or evaluation mode

        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)
        future : 3D numpy array
            Future datapoints. Dimensions are (cells, horizon, features)

        Returns
        -------
        X : 2D numpy array
            The inputs to the MLP network. Dimensions are 
            (cells, past_steps * features + horizon * 1)
        Y : 3D numpy array
            The groundtruth for the MLP network. Dimensions are
            (cells, horizon, 1)

        """
        
        X = np.concatenate(
            (self.control(past), future[:, :, self.features.index("stims")]),
            axis=1
            )
        Y = future[:, :, [feature in ("fluos", "fluo1") for feature in self.features]]

        return X, Y
    
    def control(self, past):
        """
        Formatting method when in control mode
        
        Parameters
        ----------
        past : 3D numpy array
            Past datapoints. Dimensions are (cells, past_steps, features)

        Returns
        -------
        X : 3D numpy array
            The inputs to an MLPMPC controller. Dimensions are
            (cells, past_steps * features)

        """
        
        X = np.reshape(
            past,
            newshape = (past.shape[0], past.shape[1]*past.shape[2]),
            order='F'
            )
        
        return X
        
    def reconstruct(self, X, Y):
        """
        Reconstruct original fluorescence and stimulations time-series from
        X, Y pair

        Parameters
        ----------
        X : 2D numpy array
            The inputs to the MLP network. Dimensions are 
            (cells, past_steps * features + horizon * 1)
        Y : 3D numpy array
            The groundtruth for the MLP network. Dimensions are
            (cells, horizon, 1)

        Returns
        -------
        fluos : 2D numpy array
            Reconstructed fluorescence trajectories. Dimensions are 
            (cells, past_steps + horizon)
        stims : 2D numpy array
            Reconstructed stimulations. Dimensions are 
            (cells, past_steps + horizon)

        """
        
        past_steps = int((X.shape[1] - Y.shape[1]) / len(self.features))
        
        
        fluos_ind = [f for f, feature in enumerate(self.features) if feature in ("fluos", "fluo1")][0]
        fluos_ind *= past_steps
        stims_ind = [f for f, feature in enumerate(self.features) if feature == "stims"][0]
        stims_ind *= past_steps
        
        fluos = np.concatenate(
            (X[:,fluos_ind:fluos_ind+past_steps],Y[:,:,0]), axis=1
            )
        stims = np.concatenate(
            (X[:,stims_ind:stims_ind+past_steps], X[:,-Y.shape[1]:]),
            axis=1
            )
        
        return fluos, stims

class Datasets(Generator):
    """
    Handle datasets for training and evaluation of NNs. Inherits from the
    Generator class so it can be fed directly to TF's .fit() function
    """

    def __init__(self, datasets, features, formatter):
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
        self.formatter = formatter
        self.test_ratio = 0.1
        self.data_type = "raw_dataset"
        self.mode = "training"
        self.horizon = 24
        self.past_steps = 36
        self.batch_size = 100
        self.normalization = Normalization()

    def load(self):
        """
        Load data from the datasets to memory

        Returns
        -------
        None.

        """

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
        """
        Re-load data without changing train/eval partitioning

        Returns
        -------
        None.

        """
        
        self.data = []
        self.cells = []

        for dataset in self.datasets:
            if os.path.isdir(dataset):
                self._load_npy(dataset)
            else:
                self._load_pkl(dataset)

    def _load_pkl(self, dataset):
        """
        Load from pickle dataset file

        Parameters
        ----------
        dataset : str
            Path to file.

        Returns
        -------
        None.

        """

        with open(dataset, "rb") as file:
            self.data += [pickle.load(file)]
            self.cells += [self.data[-1][self.data_type]["stims"].shape[0]]

    def _load_npy(self, dataset):
        """
        Load from numpy dataset files in directory

        Parameters
        ----------
        dataset : str
            Path to directory.

        Returns
        -------
        None.

        """

        self.data += [{self.data_type: dict()}]
        for file in os.listdir(dataset):
            if file.endswith(".npy"):
                key = os.path.splitext(file)[0]
                self.data[-1][self.data_type][key] = np.load(
                    os.path.join(dataset, file)
                )
        self.cells += [self.data[-1][self.data_type]["stims"].shape[0]]
    
    def save_state(self, filename):
        """
        Save dataset state (ie all except actual data)

        Parameters
        ----------
        filename : str
            Path to file.

        Returns
        -------
        None.

        """
        
        
        # Save all fields except data:
        with open(filename, "wb") as f:
            pickle.dump({k:v for k,v in self.__dict__.items() if k!='data'}, f)
    
    def load_state(self, filename):
        """
        Load dataset state (need to run self.reload() afterwards

        Parameters
        ----------
        filename : str
            Path to file.

        Returns
        -------
        None.

        """
        
         # Load and set fields:
        with open(filename, "wb") as f:
            state = pickle.load(f)
        
        for k, v in state:
            setattr(self, k, v)

    def normalize(self):
        """
        Normalize data loaded from raw datasets using normalization fcn below

        Raises
        ------
        RuntimeError
            If raw data was not loaded into memory with load function.

        Returns
        -------
        None.

        """

        for set_number, dataset in enumerate(self.data):
            if "raw_dataset" not in dataset:
                raise RuntimeError(
                    "Raw values not loaded for dataset #%d: %s"
                    % (set_number, self.datasets[set_number])
                )
            
            dataset["normalized_dataset"] = self.normalization.normalize(
                dataset["raw_dataset"]
                )
    
    def sample(self):
        """
        Get single sample

        Returns
        -------
        past : TYPE
            DESCRIPTION.
        future : TYPE
            DESCRIPTION.

        """
        
        if isinstance(self.past_steps, (list, tuple)):
            min_steps, max_steps = self.past_steps
        else:
            min_steps = max_steps = self.past_steps
        
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
        
        # Get dataset ref:
        dataset = self.data[set_nb][self.data_type]
        
        # Random time point:
        timepoint = np.random.randint(
            min_steps, dataset["stims"].shape[1] - self.horizon
        )
        if min_steps == max_steps:
            past_point = timepoint-max_steps
        else:
            past_point = np.random.randint(timepoint-max_steps, timepoint-min_steps)
        past_point = max(past_point, 0)
        
        # Init sample:
        past = np.zeros((max_steps, len(self.features)), dtype=np.float32)
        future = np.empty((self.horizon, len(self.features)), dtype=np.float32)
        
        # Run through features, compile sample:
        
        for f, feature in enumerate(self.features):
            past[past_point-timepoint:,f] = dataset[feature][
                cell_nb, past_point:timepoint, 0
                ]
            future[:,f] = dataset[feature][
                cell_nb, timepoint : timepoint + self.horizon, 0
                ]
        
        return past, future

    def batch(self):
        """
        Get batch of X, Y data to feed into NN

        Returns
        -------
        X : ND array
            Input batch, formatted as required by the trained model.
        Y : ND array
            Input batch, formatted as required by the trained model.

        """
        
        if isinstance(self.past_steps, (list, tuple)):
            past_steps = self.past_steps[1]
        else:
            past_steps = self.past_steps

        past = np.empty(
            (self.batch_size, past_steps, len(self.features)), dtype=np.float32
        )
        future = np.empty(
            (self.batch_size, self.horizon, len(self.features)), dtype=np.float32
        )

        for batch_ind in range(self.batch_size):
            past[batch_ind], future[batch_ind] = self.sample()

        X, Y = self.formatter.training(past, future)

        return X, Y

    def send(self, ignored_arg):
        """Generator function (called by next())"""

        batch = self.batch()

        return batch

    def throw(self, type=None, value=None, traceback=None):
        """Generator function for end of iterations"""
        raise StopIteration


def load_datasets(parameters):
    """
    Load and return training set and evaluation set

    Parameters
    ----------
    parameters : dict
        Datasets parameters. Typically config.defaults

    Returns
    -------
    training_set : Datasets object
        Training datasets object.
    evaluation_set : Datasets object
        Evaluation datasets object.

    """
    
    training_set = evaluation_set = None
    
    # Load and set up training set:
    if len(parameters["training_sets"]):
        training_files = [
            parameters["datasets_folder"] + x for x in parameters["training_sets"]
            ]
        training_set = Datasets(
            training_files,
            features = parameters["features"],
            formatter = LSTMFormatter(parameters["features"])
            )
        training_set.test_ratio = 0
        training_set.horizon = parameters["horizon"]
        training_set.past_steps = parameters["past_steps"]
        training_set.batch_size = parameters["batch_size"]
        training_set.mode = "training"
        training_set.load()
        training_set.normalize()
        training_set.data_type='normalized_dataset'
    
    # Load and set up evaluation set:
    if len(parameters["eval_sets"]):
        evaluation_files = [
            parameters["datasets_folder"] + x for x in parameters["eval_sets"]
            ]
        evaluation_set = Datasets(
            evaluation_files,
            features = parameters["features"],
            formatter = LSTMFormatter(parameters["features"])
            )
        evaluation_set.test_ratio = 1
        evaluation_set.horizon = parameters["horizon"]
        evaluation_set.past_steps = parameters["past_steps"]
        training_set.batch_size = parameters["batch_size"]
        training_set.mode = "evaluation"
        evaluation_set.load()
        evaluation_set.normalize()
        evaluation_set.data_type='normalized_dataset'
    
    return training_set, evaluation_set

def compile_dataset(xpfolder, min_area =  200):
    """
    Compile a dataset from the pickle files of a full DeLTA pipeline run

    Parameters
    ----------
    xpfolder : str
        Path to the experiment folder.
    min_area : int or float, optional
        The minimum area of a cell for it to be considered as a potential
        mother cell.

    Returns
    -------
    dataset : dict
        A dictionary containing the extracted features for each cell in the
        experiment.

    """
        
    import delta
    import cv2
    
    # Probably not necessary?
    delta.config.load_config(xpfolder+"/delta_config.json")
    
    # Lists of features that are extracted:
    delta_features = ['length', 'width', 'area', 'perimeter', 'fluo1']
    other_features = [
        'sharpness', 
        'cell_count', 
        'chamber_mean_fluo1', 
        'chamber_median_fluo1', 
        'chamber_std_fluo1', 
        'stims',
        'neighbor_stims',
        ]
    
    
    # Init dataset with empy lists:
    dataset = dict()
    for feature in delta_features+other_features:
        dataset[feature] = []
    
    
    # Re-load xpreader:
    reader = delta.utilities.xpreader(
        xpfolder,
        prototype = "pos%04d/chan%02d_frame%06d.tif",
        fileorder = "pct",
        filenamesindexing = 1,
        )
    
    # Load mothers measurements:
    with  open(os.path.join(xpfolder, "mothers.pkl"),"rb") as f:
        mothers_pkl = pickle.load(f)
    
    
    # Go through positions, re-load delta results, compile set:
    for pos_nb in range(reader.positions):
        
        print(f"Position {pos_nb}",flush=True)
        
        # Compute which series this is:
        _pos_counter = 0
        for series_nb, mothers_series in enumerate(mothers_pkl):
            if pos_nb < _pos_counter + len(mothers_series):
                series_pos_nb = pos_nb - _pos_counter
                break
            _pos_counter += len(mothers_series)
        
        # DeLTA position object:
        d_pos = delta.pipeline.Position(None,None,None)
        d_pos.load(os.path.join(xpfolder, f"delta_results/Pos{pos_nb:06d}.pkl"))
        
        # Read images for position:
        print("Reading images",flush=True)
        trans_frames = reader.getframes(positions=[pos_nb],channels=[0])
        print("Drift correction",flush=True)
        trans_frames, _ = delta.utilities.driftcorr(trans_frames, drift=d_pos.drift_values)
        
        # Go through ROIs:
        for roi in d_pos.rois:
            print(f"ROI {roi.roi_nb}",flush=True)
            mother_data = dict()
            
            # Init mother/chamber dict:
            for feature in delta_features+other_features:
                mother_data[feature] = np.zeros(
                    shape = (len(roi.label_stack),), dtype = np.float32
                    )
            
            # Go through frames
            for f, frame in enumerate(roi.label_stack):
                    
                # Extract top cell DeLTA features:
                cells = delta.utilities.getcellsinframe(
                    frame, return_contours=False
                    )
                
                # Sharpness
                img = delta.utilities.cropbox(trans_frames[f],roi.box) # Crop chamber image out
                img = delta.utilities.rangescale(img, rescale = (0, 1)) # Rescale values
                img = (img*255).astype(np.uint8) # Rescale to uint8 range
                mother_data["sharpness"][f] = np.mean(cv2.Laplacian(img,2)) # "sharpness"
                
                
                # Stimulations:
                mother_data["stims"][f] = mothers_pkl[series_nb][series_pos_nb][roi.roi_nb]["stims"][f]
                
                # Neighbor stimulations:
                if roi.roi_nb > 0:
                    mother_data["neighbor_stims"][f] += mothers_pkl[series_nb][series_pos_nb][roi.roi_nb-1]["stims"][f]*0.5
                if roi.roi_nb < len(mothers_pkl[series_nb][series_pos_nb]) - 1:
                    mother_data["neighbor_stims"][f] += mothers_pkl[series_nb][series_pos_nb][roi.roi_nb+1]["stims"][f]*0.5

                # Remove cells that are too small:
                for cell in cells.copy():
                    if roi.lineage.getvalue(cell,f,"area") < min_area:
                        cells.remove(cell)

                # If no cell in frame, skip to next frame:
                if len(cells) == 0:
                    continue
                    
                # DeLTA features:
                for feature in delta_features:
                    mother_data[feature][f] = roi.lineage.getvalue(
                        cells[0],f,feature
                    )
                
                # total_cells
                mother_data["cell_count"][f] = len(cells) 
                
                # All cells fluo:
                all_fluo = []
                for cell in cells:
                    all_fluo += [roi.lineage.getvalue(
                        cell,f,"fluo1"
                    )]
                
                # Mean, median, std dev:
                mother_data["chamber_mean_fluo1"][f] = np.mean(all_fluo)
                mother_data["chamber_median_fluo1"][f] = np.median(all_fluo)
                mother_data["chamber_std_fluo1"][f] = np.std(all_fluo)
                

            # Append to dataset:
            for feature in delta_features+other_features:
                dataset[feature].append(mother_data[feature][:,np.newaxis])
    
    # Compile feature lists of mothers into arrays:
    for feature in delta_features+other_features:
        dataset[feature] = np.array(dataset[feature])

    return dataset


def single_cell_plot(raw_dataset, cell_nb, savefig = None):
    """
    Plot a single cell's full time-series (all features)

    Parameters
    ----------
    raw_dataset : dict
        Non-normalized dataset, as returned by eg compile_dataset().
    cell_nb : int
        The index of the cell in the dataset to plot.
    savefig : str or None, optional
        Path to save the figure to. If None, it is not saved to disk. 
        The default is None.

    Returns
    -------
    None.

    """
    
    x = np.arange(0, raw_dataset["stims"][cell_nb].shape[0], dtype = np.float32) / 12
    
    plt.figure(figsize=(6, 12), dpi=300)
    
    # Fluo & stims plot:
    plt.subplot(3,1,1)
    
    # Plot stimulations:
    utils.OptoPlotBackground(
        raw_dataset["stims"][cell_nb], 
        x = x,
        ymin = 0,
        ymax = 4095,
        )
    
    # Plot fluorescence:
    plt.plot(x,raw_dataset["chamber_mean_fluo1"][cell_nb],"k",label="Mean")
    plt.plot(x,raw_dataset["chamber_median_fluo1"][cell_nb],"w",label="Median")
    plt.plot(x,raw_dataset["fluo1"][cell_nb],"b",label="Mother")
    plt.ylabel("Fluorescence")
    plt.xlim([x[0], x[-1]])
    plt.ylim(0, 4095)
    plt.legend()
    plt.title(f"Cell {cell_nb}")
    
    # Morpho plot:
    ax1 = plt.subplot(3,1,2)
    
    ax2 = ax1.twinx()
    ax1.plot(x,raw_dataset["length"][cell_nb],"g",label="length", zorder = 10)
    ax1.plot(x,raw_dataset["width"][cell_nb],"g",alpha=.5,label="width", zorder = 10)
    
    ax2.plot(x,raw_dataset["area"][cell_nb],"b",label="area", zorder = 10)
    ax2.plot(x,raw_dataset["perimeter"][cell_nb],"b",alpha=.5,label="perimeter", zorder = 10)
    
    # # Divisions:
    # yl = ax1.get_ylim()
    # for _x, _d in zip(x, raw_dataset["divisions"][cell_nb]):
    #     if _d:
    #         ax1.plot([_x, _x],yl,color="k", alpha=.1, zorder= 5)
    # ax1.set_ylim(yl)
    
    ax1.set_ylabel('length & width, pixels', color='g')
    ax2.set_ylabel('area & perim., pixels', color='b')
    plt.xlim([x[0], x[-1]])
    plt.grid(which="both", axis="both")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    
    # Other features:    
    plt.subplot(12,1,9)
    plt.plot(x,raw_dataset["chamber_std_fluo1"][cell_nb],"k")
    plt.ylabel("Std. Fluo")
    plt.xlim([x[0], x[-1]])
    plt.grid(which="both", axis="both")
    
    plt.subplot(12,1,10)
    plt.plot(x,raw_dataset["cell_count"][cell_nb],"k")
    plt.ylabel("# cells")
    plt.xlim([x[0], x[-1]])
    plt.yticks(list(range(*[int(x) for x in plt.ylim()])))
    plt.grid(which="both", axis="both")
    
    plt.subplot(12,1,11)
    plt.plot(x,raw_dataset["sharpness"][cell_nb],"k")
    plt.ylabel("Sharp.")
    plt.xlim([x[0], x[-1]])
    plt.grid(which="both", axis="both")
    
    plt.subplot(12,1,12)
    plt.plot(x,raw_dataset["neighbor_stims"][cell_nb],"k")
    plt.ylabel("Neigh. stims")
    plt.xlim([x[0], x[-1]])
    plt.xlabel("time (hours)")
    plt.grid(which="both", axis="both")
    
    if savefig is not None:
        plt.savefig(savefig+".png", dpi=300)
        plt.savefig(savefig+".svg", dpi=300)
    plt.show()
    
    
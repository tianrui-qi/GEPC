#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is dedicated to loading, formatting, and feeding data for
training the encoder-decoder network. The Normalization object is also
used during experiments to normalize data on the fly.

Created on Sun Sep 26 19:46:40 2021

@author: jeanbaptiste
"""
import os
from collections.abc import Generator
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    def __init__(self, features, future_features=("fluo1", "stims")):
        self.features = features
        "List of features to compile as input (in addition to 'stims')"
        self.future_features = future_features
        "List of features in the future part of the timeseries"

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

        X = [self.control(past), future[:, :, self.future_features.index("stims")]]
        Y = future[:, :, [feature in ("fluos", "fluo1") for feature in self.future_features]]

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


class Datasets(Generator):
    """
    Handle datasets for training and evaluation of NNs. Inherits from the
    Generator class so it can be fed directly to TF's .fit() function
    """

    def __init__(
            self,
            datasets,
            formatter,
            parameters,
            future_features= ("fluo1", "stims"),
            ):
        """
        Instanciate

        Parameters
        ----------
        datasets : List of str
            List of path to dataset pickle files.
        features : List of str
            List of features to use from the past part of the timeseries.
        formatter : AbstractFormatter
            Class of the formatter to use on the batch samples to adapt data to the
            network.
        future_features: list of str, optional
            List of features to extract from the future part of the timeseries,
            default is ("fluo1", "stims").

        Returns
        -------
        None.

        """
        self.datasets = datasets
        self.future_features = future_features
        self.test_ratio = 0.1
        self.data_type = "raw_dataset"
        self.mode = "training"
        self.features = parameters["features"]
        self.horizon = parameters["horizon"]
        self.past_steps = parameters["past_steps"]
        self.batch_size = parameters["batch_size"]
        self.formatter = formatter(self.features, self.future_features)
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

    def _random_cell(self):
        """
        Pick random cell out of the datasets

        Returns
        -------
        dataset : dict
            Dataset that the cell number was picked from.
        cell_nb : int
            Cell number in the dataset.

        """

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

        return dataset, cell_nb

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

        # Get random cell from partition:
        dataset, cell_nb = self._random_cell()

        # Random time point:
        if isinstance(self.past_steps, (list, tuple)):
            min_steps, max_steps = self.past_steps
        else:
            min_steps = max_steps = self.past_steps
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
        future = np.empty(
            (self.horizon, len(self.future_features)), dtype=np.float32
            )

        # Run through features, compile sample:
        for f, feature in enumerate(self.features):
            past[past_point-timepoint:,f] = dataset[feature][
                cell_nb, past_point:timepoint, 0
                ]
        for f, feature in enumerate(self.future_features):
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
            (self.batch_size, self.horizon, len(self.future_features)), dtype=np.float32
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


class Datasets_cnn(Datasets):


    def __init__(self, datasets, formatter, parameters, *args, **kwargs):
        super().__init__(datasets, formatter, parameters, *args, **kwargs)
        self.n_bins = parameters["cnn_bins"]
        "Number of bins to quantize fluorescence values (must be multiple of 8)"

    def normalize(self):

        super().normalize()

        for d, data in enumerate(self.data):

            density = data["raw_dataset"].copy()

            if os.path.exists(self.datasets[d]+"/compiled/density.npy"):
                print("Loading gaussian image representations from disk...", end=" ")
                density["fluo1"] = np.load(self.datasets[d]+"/compiled/density.npy")
                print("done.")
            else:
                print("Compiling gaussian image representations...")
                density["fluo1"] = np.zeros(
                    density["fluo1"].shape + (self.n_bins,), dtype=np.float32
                    )
                for c in tqdm(range(density["fluo1"].shape[0])):
                    density["fluo1"][c] = gaussian_img(
                        data["raw_dataset"]["fluo1"][c], self.n_bins
                        )
                os.makedirs(self.datasets[d]+"/compiled/", exist_ok = True)
                np.save(self.datasets[d]+"/compiled/density.npy", density["fluo1"])

            data["density_dataset"] = density

    def sample(self):

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
        density = self.data[set_nb]["density_dataset"]["fluo1"]

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

        # Run through features, compile past:
        for f, feature in enumerate(self.features):
            past[past_point-timepoint:,f] = np.squeeze(
                dataset[feature][cell_nb, past_point:timepoint]
                )

        # Future light inputs:
        future_stims = dataset["stims"][cell_nb, timepoint : timepoint + self.horizon]

        # Retrieve gaussian image representations for the groundtruth:
        groundtruth = density[cell_nb, timepoint : timepoint + self.horizon]

        return past, future_stims, groundtruth

    def batch(self):

        if isinstance(self.past_steps, (list, tuple)):
            past_steps = self.past_steps[1]
        else:
            past_steps = self.past_steps

        past = np.empty(
            (self.batch_size, past_steps, len(self.features)), dtype=np.float32
        )
        future_stims = np.empty(
            (self.batch_size, self.horizon), dtype=np.float32
        )
        ground_truth = np.empty(
            (self.batch_size, self.horizon, self.n_bins), dtype=np.float32
        )

        for b in range(self.batch_size):
            past[b], future_stims[b], ground_truth[b] = self.sample()

        return (past, future_stims), ground_truth

# import cv2
# def gaussianify(cell_trajectory, quantization):

#     img = np.zeros((cell_trajectory.shape[0],4096), dtype = np.uint16)
#     for t, value in enumerate(cell_trajectory.astype(np.uint16)):
#         img[t,value] = 65_535

#     sigma = int(4096/quantization)
#     kernel = np.transpose(cv2.getGaussianKernel(12*sigma,sigma))

#     img = cv2.filter2D(img, ddepth = -1, kernel = kernel)

#     final_img = np.empty((cell_trajectory.shape[0],quantization), dtype = np.float32)
#     bars = np.linspace(0, 4096, quantization+1).astype(int)
#     for b in range(quantization):
#         final_img[:,b] = np.sum(img[:,bars[b]:bars[b+1]],axis=1)
#     final_img /= 65_535

#     return final_img

def gaussian_img(cell_trajectory, n_bins):
    """
    Compile a "Gaussian image" of a single-cell trajectory

    Parameters
    ----------
    cell_trajectory : 1D array of floats
        Single cell fluorescence trajectory (not normalized, ie values between
        0 and 4095).
    n_bins : int
        Number of bins to "quantize" the trajectory over.

    Returns
    -------
    img : 2D array of float32
        Image representation of the trajectory.

    """

    # Precompute gaussian:
    sigma = int(4096/n_bins)+1
    x = np.arange(-6*sigma, 6*sigma)
    gau = np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)).astype(np.float32)

    # Init image reprensentation with 0s
    img = np.zeros((cell_trajectory.shape[0],n_bins), dtype = np.float32)

    # Run through timepoints:
    for t, value in enumerate(cell_trajectory):

        # Which bin is the central one:
        main_bin_ind = int(value // sigma)
        # How much shift?
        shift = round(value) % sigma

        # Run through bins:
        for b in range(-6,6):

            # Find how to bin the gaussian curve:
            if main_bin_ind+b<0 or main_bin_ind+b>=n_bins:
                continue
            gau_start = b*sigma + 6*sigma - shift -1
            if gau_start < 0:
                gau_start = 0
            gau_end = (b+1)*sigma + 6*sigma - shift -1

            # Integrate gaussian over the interval:
            img[t,main_bin_ind+b] = np.sum(gau[gau_start:gau_end])

    return img


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
            formatter = LSTMFormatter,
            parameters = parameters
            )
        training_set.test_ratio = 0.1
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
            formatter = LSTMFormatter,
            parameters = parameters
            )
        evaluation_set.test_ratio = 1
        training_set.mode = "evaluation"
        evaluation_set.load()
        evaluation_set.normalize()
        evaluation_set.data_type='normalized_dataset'

    return training_set, evaluation_set

def compile_dataset(
        xpfolder, min_area = 200, delta_features = None, other_features = None
        ):
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
    if delta_features is None:
        delta_features = ['length', 'width', 'area', 'perimeter', 'fluo1']
    if other_features is None:
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
                for fluoN in ["fluo1", "fluo2"]:
                    if fluoN not in delta_features:
                        continue

                    all_fluo = []
                    for cell in cells:
                        all_fluo += [roi.lineage.getvalue(
                            cell,f,fluoN
                        )]

                    # Mean, median, std dev:
                    mother_data["chamber_mean_"+fluoN][f] = np.mean(all_fluo)
                    mother_data["chamber_median_"+fluoN][f] = np.median(all_fluo)
                    mother_data["chamber_std_"+fluoN][f] = np.std(all_fluo)


            # Append to dataset:
            for feature in delta_features+other_features:
                dataset[feature].append(mother_data[feature][:,np.newaxis])

    # Compile feature lists of mothers into arrays:
    for feature in delta_features+other_features:
        dataset[feature] = np.array(dataset[feature])

    return dataset


def format_dataset_from_motherspkl(folder):

    # Load mothers measurements:
    with  open(folder + "/mothers.pkl","rb") as f:
        mothers = pickle.load(f)

    # Get feature names list
    with  open(folder + "/experiment_settings.json","r") as f:
        features = json.load(f)["features"]

    # Find last timepoint:
    values = np.sum(
        mothers[0][0][10][:, [x for x, n in enumerate(features) if n!="stims"]],
        axis=1
        )
    cutoff = np.nonzero(values)[0][-1]+1

    # Init dataset with empy lists:
    dataset = dict()
    for feature in features:
        dataset[feature] = []

    # Add mothers data to dataset:
    for series in mothers:
        for position in series:
            for chamber in position:
                for f, feature in enumerate(features):
                    dataset[feature].append(chamber[:cutoff,f])

    # Array-ify:
    for feature in features:
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

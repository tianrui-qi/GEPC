
Data & datasets
================

``deepcellcontrol/data.py`` contains the classes and functions to compile 
datasets from experimental data, to normalize data, and to feed training and 
evaluation data to timeseries forecasting models.


Downloading the data
---------------------

We have not yet released our experimental data, please stay tuned.

Compiling datasets
-------------------

During training and evaluation dataset experiments not all the features we 
eventually use / might use during training and control are extracted. For this 
reason after experiments were completed, we ran the complete DeLTA pipeline
on the acquired images.

Then the DeLTA output files are further processed with the ``compile_dataset`` 
function to extract the following features for the mother cell in each chamber:

* Original DeLTA features:
    * ``length``: Mother cell length (rotated bounding box, in pixels)
    * ``width``: Mother cell width (rotated bounding box, in pixels)
    * ``area``: Mother cell area (counted pixels)
    * ``perimeter``: Mother cell perimeter (in pixels)
    *  ``fluo1``: Mother cell mean fluorescence (Camera intensity a.u.)
* Other extracted features:
    * ``sharpness``: Mean of Laplacian over the phase contrast image of the chamber (a.u.)
    * ``cell_count``: Number of cells in the chamber
    * ``chamber_mean_fluo1``: Mean of ``fluo1`` for all cells in the chamber
    * ``chamber_median_fluo1``: Median of ``fluo1`` for all cells in the chamber
    * ``chamber_std_fluo1``: Standard deviation of ``fluo1`` for all cells in the chamber
    * ``stims``: Optogenetic stimulations for the chamber. 0 = red, 1 = green.
    * | ``neighbor_stims``: Optogenetic stimulations for the two immediate 
        neighbor chambers (0 = both red, 1 = one red one green, 2 = both green)

The compiled dataset is a dictionary, with each feature name as a key, and each
item under these keys is a 2D array containing the data for each chamber over time.

See also the script :ref:`compile_datasets.py <compile_script>`

Normalization
--------------

The ``Normalization`` class is used to normalize the data from the dataset to 
the [0, 1] range to feed into the neural networks.

We used the following formulas to ensure 
single-cell timeseries data were consistently normalized to the same range of 
values:

* | Fluorescence: Single-cell fluorescence measurements were limited to [0, 4095],
    the dynamic range of the data. Fluorescence features values (mother cell 
    fluorescence, average chamber fluorescence, and chamber fluorescence standard 
    deviation) were thus simply normalized linearly:
    
    :math:`x_{norm}=x_{raw}/4095` 
    
  | Where :math:`x_{raw}` is the initial measured value for a given feature and 
    time point, and :math:`x_{norm}` is the resulting normalized value that is 
    used as input for the neural network, both for training and on-the-fly for 
    feedback control.


* | Cell area: Mother cell area, measured in pixels, theoretically has no clear 
    upper bound. After analyzing area distributions, we used a negative 
    exponential function:
    
    :math:`x_{norm}=1-{10}^{-x_{raw}/3000}`
    

* | Cell length and width: Although we did not end up using cell length and 
    width, we normalized length and width with the following formula:
    
    :math:`x_{norm}=1-{10}^{-x_{raw}/200}`

* | Chamber cell count: The number of cells in the chamber at any given timepoint 
    also theoretically has no clear upper bound. We also used a negative 
    exponential function with a different normalization factor:
    
    :math:`x_{norm}=1-{10}^{-x_{raw}/9}`


* | Chamber image sharpness: Image sharpness, measured as the mean of the 
    Laplacian of the image, has no clear upper or lower bound. We used the 
    following formula to normalize sharpness to [0, 1]:
    
    :math:`x_{norm}=\frac{1}{1+e^{-x_{raw}+3.6}}`

The factors of all these formulas were selected manually to make sure that the
distribution of the values were spread over the [0, 1] range.

The ``Normalization`` class is used for training as well as on the fly during
control experiments.

Training generators
---------------------

The ``Datasets`` class inherits from regular python 
`generators <https://wiki.python.org/moin/Generators>`_ and as such can be fed
directly into Keras's ``model.predict()``. The role of the class is to load 
compiled datasets pickle files, normalize the data via the ``Normalization`` 
class, and then randomly yield data batches either as training inputs or 
validation data after formatting it depending on the network architecture.

The ``Datasets`` object can randomly partition the cell timeseries in the datasets 
into training or validation data and be switched between to make sure data from
different cells is used for training or validation. Alternatively, two objects
can be instanciated with different subsets of experimental datasets, where one
is used exclusively for training, and the other exclusively for validation. 
This makes it possible to train and validate on completely different 
experiments. This is what we did to evaluate the performance of our models.

The "state" of the object can be saved to disk, in the sense that, if data has
been loaded and cells accross experiments have been partitioned into training 
and validation sets, this partition can be saved to disk and reloaded. However 
it does not save the state of the random number generator that is used to
select random cells and random timepoints on the fly when generating batches
for training and evaluation. This can probably be worked around by setting the
numpy.random seed, but I have not experimented with it.

The ``load_datasets()`` function instanciates two ``Datasets`` objects, one for
training, the other for validation, based on a dictionary describing how to 
split the data. See :doc:`the config module <config>` and 
:ref:`the training script <training_script>`

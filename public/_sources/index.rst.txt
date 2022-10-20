.. deepcellcontrol documentation master file, created by
   sphinx-quickstart on Wed Oct 19 17:05:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deepcellcontrol's documentation!
===========================================

`deepcellcontrol (dcc) <https://gitlab.com/dunloplab/deepcellcontrol>`_
is a python package that performs deep-learning based 
feedback control of gene expression in single cells. It is the core repository 
of our manuscript available on bioRxiv:

[Reference and citation]

It interfaces with the 
`DeLTA <https://gitlab.com/dunloplab/delta>`_ and 
`pycromanager_tessie <https://gitlab.com/dunloplab/pycromanager>`_ packages on 
Tessie's computer (Tessie is the name of a microscope in the Dunlop lab) 
to conduct high-throughput feedback experiments.

This repository contains the dcc package itself under the
``/deepcellcontrol`` sub-folder, and scripts under the ``/scripts`` sub-folder
that use the dcc package to compile and format microscopy data, train and 
evaluate timeseries forecasting networks, run feedback control algorithms 
within a TCP/IP server, and finally to plot all the figures in our manuscript.
The ``/public`` subfolder simply contains this documentation.

.. warning::
    This documentation is a work in progress

Installation
-------------

The `dcc package <https://gitlab.com/dunloplab/deepcellcontrol>`_
can run either in the `DeLTA <https://gitlab.com/dunloplab/delta>`_ environment 
or in the `pycromanager_tessie <https://gitlab.com/dunloplab/pycromanager>`_ 
environment. 
We do not provide installation instructions here, you can check the 
documentation of these packages.

Some functions and scripts, especially to recreate our figures, will need the
DeLTA package to be installed and in the python path to be able to run.

Our experimental data required to run the scripts is not available yet, but as
soon as it is released we will either provide instructions on how to download 
it and modify scripts, or automate it like in DeLTA.

Issues and support
-------------------

We can not at the moment provide the same level of support that we do for 
DeLTA, but we will handle bug reports and questions related to the data via
the Gitlab issues system. Make sure you provide as much context as possible in
your report (OS, library versions etc...).



Contents
----------

.. toctree::
    :maxdepth: 2
    
    pages/data
    pages/models
    pages/timeseries
    pages/control
    pages/server
    pages/scripts
    pages/config


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

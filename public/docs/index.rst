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

Quick description
-------------------

This repository contains the dcc package itself under the
``/deepcellcontrol`` sub-folder, and scripts under the ``/scripts`` sub-folder
that use the dcc package to compile and format microscopy data, train and 
evaluate timeseries forecasting networks, run feedback control algorithms 
within a TCP/IP server, and finally to plot all the figures in our manuscript.
The ``/public`` subfolder simply contains this documentation.

The pages linked below get into the details of the implementation, but here is
a quick description of the package and scripts:

* ``deepcellcontrol``: main package/library (also referred to as ``dcc`` in the docs)
    * ``deepcellcontrol/data.py``: Compile datasets from experimental data, 
      normalize data, generators to feed training / validation data
    * ``deepcellcontrol/models.py``: Define neural network models, and 
      manipulate them (e.g. split encoder and decoder)
    * ``deepcellcontrol/timeseries.py``: Train and evaluate timeseries 
      forecasting neural network models with the datasets.
    * ``deepcellcontrol/control.py``: Model predictive control algorithms
      that use the timeseries forecasting models, and binary particle
      swarm optimizer.
    * ``deepcellcontrol/server.py``: Server and client classes to be able to
      run the control algorithms on a different machine.
    * ``deepcellcontrol/config.py``: Parameters used accross the package.
    * ``deepcellcontrol/utilities.py``: Utility functions and classes used 
      accross the package and in the scripts
    * ``deepcellcontrol/simulations.py``: Gillespie simulations of the CcaSR
      system, used early on for testing purposes.


* ``scripts``: scripts that use the main package
    * ``scripts/compile_datasets.py``: Generate datasets from experimental
      data using ``dcc.data`` and save them to pickle files
    * ``scripts/training.py``: Train timeseries forecasting models, evaluate
      them, and save models and evaluation to disk.
    * ``scripts/cluster_training.py``: Run training script on BU's Shared
      Computing Cluster.
    * ``scripts/run_server.py``: Run control server 
      ``dcc.server.SocketServer`` on local machine, to answer control compute
      requests from remote client that uses ``dcc.server.DistantServer``
    * ``scripts/figure1.py``: Script performing data analysis and plots
      for Figure 1 of our study. 
    * ``scripts/figure2.py``: Script performing data analysis and plots
      for Figure 2, and Figures S1 & S2 of our study.
    * ``scripts/figure3.py``: Script performing data analysis and plots
      for Figure 3, Figures S3-6, and Movies S1 & S2 of our study.
    * ``scripts/figure4.py``: Script performing data analysis and plots
      for Figure 4, and Movie S3 of our study.
    * ``scripts/figure5.py``: Script performing data analysis and plots
      for Figure 5, Figures S7 & S8, and Movie S4 of our study.


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

About this documentation
-------------------------

This documentation is built using 
`sphinx <https://www.sphinx-doc.org/en/master/>`_, with the numpydoc, autodoc, 
and autosummary extensions. The source RST pages are inside ``public/docs`` and
``public/docs/pages``.

To build simply run the following shell command at the root of the repository, 
in the delta or pycromanager environement (+ sphinx packages)::

    sphinx-build -b html public/docs/ public/


When the changes are pushed to gitlab, the CI/CD pipeline will publish the new
generated html pages (see ``.gitlab-ci.yml``)

Note: The modifications might take a while to appear online,
for caching reasons.

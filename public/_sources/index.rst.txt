.. deepcellcontrol documentation master file, created by
   sphinx-quickstart on Wed Oct 19 17:05:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deepcellcontrol's documentation!
===========================================

``deepcellcontrol`` is a python package that performs deep-learning based 
feedback control of gene expression in single cells.

It interfaces with the 
`DeLTA <https://gitlab.com/dunloplab/delta>`_ and 
`pycromanager_tessie <https://gitlab.com/dunloplab/pycromanager>`_ packages on 
Tessie's computer to conduct high-throughput feedback experiments.

It is divided in 2 main sub-packages: 

* | The :doc:`microscope sub-package <pages/microscope>`, that handles the hardware
    operations, and for the most part simply repackages pycromanager commands
    to make them more user-friendly and use default settings that seem to work
    best.
* | The :doc:`acquisitions sub-package <pages/acquisitions>`, that manages 
    experiments logic like order of acquisitions, timing, interfacing with
    DeLTA and DCC etc...


Contents
----------

.. toctree::
    :maxdepth: 2
    
    pages/devices
    pages/micromanager
    pages/microscope
    pages/acquisitions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

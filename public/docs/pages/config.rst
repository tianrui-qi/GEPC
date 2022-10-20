Configuration file
====================

``deepcellcontrol/config.py`` contains the parameters that are used throughout
the dcc package. You shouldn't need to modify it to make the package work.

``dcc.config.defaults`` is the dictionary that contains default parameters that 
are used for datasets, models construction, and training. The values in the
dictionary are the ones used in our publication, unless specified otherwise. In
order to try different parameters, you can create a copy of the dictionary
and then alter it, for example::

    import copy
    params = copy.deepcopy(dcc.config.defaults)
    params["horizon"] = 60

See comments in the ``config.py`` file directly for a description of each 
parameter's role.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:56:30 2020

@author: jeanbaptiste
"""

import numpy as np

from ..models import mlp, lstm


class _Controller:
    """
    Parent controller class for MPCs, PIs etc...

    Attributes
    ----------
    strategy : 2D numpy array of bools
        Array containing strategies for each cell over the entire
        prediction horizon. Size is cells -by- horizon.

    Methods
    -------
    __init__:
        Instanciation.
    feedback:
        Feedback function for multiple cells at the current timepoint.

    """

    def __init__(self):
        """
        Instanciation.

        Returns
        -------
        None.

        """
        self.strategy = None

    def feedback(self, inputs, objectives):
        """
        Feedback function for multiple cells at the current timepoint.

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        objectives : list
            List containing control objectives. Each list element contains a 1D
            numpy array of future objective values for each cell to process in
            parallel.


        Returns
        -------
        1D numpy array of bools
            Control inputs to apply at the current timestep for each cell.

        """

        # Get best strategy:
        self.strategy = self.get_strategy(inputs, objectives)

        # Return only next timepoint control inputs:
        return self.strategy[:, 0]

    def get_strategy(self, inputs):
        pass  # To be defined in subclasses


class _MPC(_Controller):
    """
    Parent class for model predictive controllers.
    Child class of _controller.

    Attributes
    ----------
    horizon : int
        Horizon over which to predict system responses, in # timepoints.
    past_steps : int or None
        Number of past steps to feed into the prediction model. If None, the
        whole history of the cell is fed into the model. Note that some models,
        for example MLPs, require a fixed number of past steps.
    strategy_optimizer : _optimizer object
        Strategy optimizer object of sub-classes of _optimizer.

    Methods
    -------
    __init__:
        Instanciation.
    getStrategy:
        Identify optimal strategies.
    run_strategies:
        Predict response to selected strategies.
    compute_scores:
        Compute RMSE over model predictions for each cell's objective.

    """

    def __init__(
        self, horizon=24, past_steps=None, strategy_optimizer=None, *args, **kwargs
    ):
        """
        Instanciation.

        Parameters
        ----------
        horizon : int, optional
            Horizon over which to predict system responses, in # timepoints.
            The default is 12.
        past_steps : int, optional
            Number of past steps to feed into the prediction model. If None,
            the whole history of the cell is fed into the model.
            The default is None.
        strategy_optimizer : _optimizer object or None, optional
            Strategy optimizer object of sub-classes of _optimizer. If None,
            a null_optimizer object is instanciated.
            The default is None.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.past_steps = past_steps
        self.model = None
        if strategy_optimizer is None:  # Default optimizer is null (brute force)
            self.strategy_optimizer = NullOptimizer(self.horizon)
        else:
            self.strategy_optimizer = strategy_optimizer

    def get_strategy(self, inputs, objectives):
        """
        Identify optimal strategies.

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        objectives : list
            List containing control objectives. Each list element contains a 1D
            numpy array of future objective values for each cell to process in
            parallel.

        Returns
        -------
        strategies : 2D numpy array of bools
            Array containing strategies for each cell over the entire
            prediction horizon. Size is cells -by- horizon.

        """

        # Initialize strategy optimization:
        strategies = self.strategy_optimizer.iterate(
            np.full(shape=len(inputs), fill_value=np.nan)
        )

        # Run optimization:
        while (
            self.strategy_optimizer.iterations > 0
        ):  # Optimizer sets iterations to -1 when stopping condition is met

            predictions = self.run_strategies(inputs, strategies)

            scores = self.compute_scores(predictions, objectives)

            strategies = self.strategy_optimizer.iterate(scores)

        return strategies

    def run_strategies(self, inputs, strategies):
        """
        Predict response to selected strategies

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        strategies : 3D numpy array of bools
            Array containing multiple strategies for each cell to predict the
            response for. Size is cells -by- strategies_per_cell -by- horizon.

        Returns
        -------
        yhat : 3D numpy array of floats.
            Output from the prediction model, for each cell and each strategy.
            Size is cells -by- strategies_per_cell -by- horizon.

        """

        # Compile array to feed into model:
        x = self.compile_x(inputs, strategies)

        # Predict:
        yhat = self.model.predict(x)

        # Format yhat into list similar to strategies:
        # yhat = np.split(yhat,np.cumsum([i.shape[0] for i in strategies]),axis=0)[:-1]
        yhat = np.reshape(yhat, newshape=strategies.shape[0:2] + (yhat.shape[1],))

        return yhat

    def compute_scores(self, predictions, objectives):
        """
        Compute RMSE over model predictions for each cell's objective.

        Parameters
        ----------
        predictions : 3D numpy array of floats
            Output from the prediction model, for each cell and each strategy.
            Size is cells -by- strategies_per_cell -by- horizon.
        objectives : list
            List containing control objectives. Each list element contains a 1D
            numpy array of future objective values for each cell to process in
            parallel.

        Returns
        -------
        scores : 2D numpy array of floats
            RMSE between each strategy per cell and corresponding objective.
            Size is cells -by- strategies_per_cell.

        """

        # Init array:
        scores = np.empty(shape=predictions.shape[0:2], dtype=float)

        for obj_ind in range(len(objectives)):

            # Reduce objective to dynamic range, cut down to horizon:
            truncated_horizon = int(min((self.horizon, objectives[obj_ind].shape[0])))
            reduced_obj = np.array(objectives[obj_ind][:truncated_horizon])

            # Repeat objective in numpy array of same size as predictions:
            obj_arr = np.repeat(
                np.expand_dims(reduced_obj, 0), predictions.shape[1], axis=0
            )

            # Compute RMSE:
            scores[obj_ind] = np.sqrt(
                np.mean(
                    (predictions[obj_ind, :, :truncated_horizon] - obj_arr) ** 2, axis=1
                )
            )

        return scores

    def compile_x():
        pass  # To be defined in subclasses


class MLPMPC(_MPC):
    """
    Model predictive controller based on the Multi-Layer Perceptron model.
    Child class of _mpc.

    Attributes
    ----------
    model : Keras model object.
        MLP as defined in models.py.

    Methods
    -------
    __init__:
        Instanciation.
    compile_x:
        Compile X array to do predictions over, formatted for the MLP model.

    """

    def __init__(self, model_file=None, hidden_layers=10, features=2, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str or None, optional
            Filepath to model weights file. If None, the following model file
            will be used: 'models/mlp/mlp%d_optimized.hdf5' %(self.horizon,)
            The default is None.
        hidden_layers : int, optional
            Number of hidden layers in the MLP model. Must match weights file.
            The default is 1.
        features : int, optional
            Number of observed variables (Fluorescence, cell length...)
            + control variables (only DMD input sequence).
            The default is 2.

        Returns
        -------
        None.

        """

        # Initialize parent properties:
        super().__init__(*args, **kwargs)

        # If no model file provided, load default optimized model
        if model_file is None:
            model_file = "models/mlp/mlp%d_optimized.hdf5" % (self.horizon,)

        # Initialize model:
        self.model = mlp(
            hidden_layers=hidden_layers,
            features=features,
            horizon=self.horizon,
            past_steps=self.past_steps,
        )
        self.model.load_weights(model_file)

    def compile_x(self, inputs, strategies):
        """
        Compile X array to do predictions over, formatted for the MLP model.

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        strategies : 3D numpy array of bools
            Array containing multiple strategies for each cell to predict the
            response to. Size is cells -by- strategies_per_cell -by- horizon.

        Returns
        -------
        2D numpy array of floats.
            Compiled X array for MLP model. Size is
            (cell * strategies_per_cell) -by-
            (past_steps * features + horizon)

        """

        # Flatten inputs together:
        x = np.empty(
            (
                strategies.shape[0] * strategies.shape[1],
                inputs[0][0].shape[0] * self.past_steps
                + self.past_steps
                + strategies.shape[2],
            ),
            dtype=float,
        )
        for strat_ind in range(strategies.shape[0]):
            single_input = np.concatenate(
                (
                    inputs[strat_ind][0][:, -self.past_steps :].flatten(),
                    inputs[strat_ind][1][-self.past_steps :].flatten(),
                ),
                axis=0,
            )

            # Compile full array for all strategies:
            x[
                strat_ind * strategies.shape[1] : (strat_ind + 1) * strategies.shape[1],
                : -strategies.shape[2],
            ] = np.repeat(
                np.expand_dims(single_input, axis=0), strategies.shape[1], axis=0
            )
            x[
                strat_ind * strategies.shape[1] : (strat_ind + 1) * strategies.shape[1],
                -strategies.shape[2] :,
            ] = strategies[strat_ind]

        return x


class LSTMMPC(_MPC):
    """
    Model predictive controller based on the LSTM encoder-decoder model.
    Child class of _mpc.

    Attributes
    ----------
    model : Keras model object.
        LSTM encoder-decoder as defined in models.py.

    Methods
    -------
    __init__:
        Instanciation.
    compile_x:
        Compile X array to do predictions over, formatted for the LSTM model.

    """

    def __init__(self, model_file=None, latent_dim=200, features=2, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str or None, optional
            Filepath to model weights file. If None, the following model file
            will be used: 'models/lstm/lstm%d_optimized.hdf5' %(self.horizon,)
            The default is None.
        latent_dim : int, optional
            Units in the encoder and decoder LSTM layers. Must match weights
            file.
            The default is 200.
        features : int, optional
            Number of observed variables (Fluorescence, cell length...)
            + control variables (only DMD input sequence).
            The default is 2.

        Returns
        -------
        None.

        """

        # Initialize parent properties:
        super().__init__(*args, **kwargs)

        # If no model file provided, load default optimized model
        if model_file is None:
            model_file = "models/lstm/lstm%d_optimized.hdf5" % (self.horizon,)

        # Initialize model:
        self.model = lstm(
            latent_dim=latent_dim,
            horizon=self.horizon,
            past_steps=self.past_steps,
            features=features,
        )
        self.model.load_weights(model_file)

    def compile_x(self, inputs, strategies):
        """
        Compile X array to do predictions over, formatted for the LSTM model.

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        strategies : 3D numpy array of bools
            Array containing multiple strategies for each cell to predict the
            response to. Size is cells -by- strategies_per_cell -by- horizon.

        Returns
        -------
        list
            List of two 2D arrays for LSTM model. First array is past values of
            observed variables + past control inputs, size is
            (cell * strategies_per_cell) -by- past_steps -by- features. Second
            array is future control inputs, size is
            (cell * strategies_per_cell) -by- horizon -by- 1.

        """

        x = []
        u = []
        if self.past_steps is None:
            past = None
        else:
            past = -self.past_steps

        for strat_ind in range(strategies.shape[0]):
            # Flatten inputs together:
            single_input = np.concatenate(
                (
                    np.transpose(inputs[strat_ind][0][:, past:]),
                    np.expand_dims(np.transpose(inputs[strat_ind][1][past:]), axis=1),
                ),
                axis=1,
            )

            # Compile full array for all strategies:
            x += [
                np.repeat(
                    np.expand_dims(single_input, axis=0),
                    strategies[strat_ind].shape[0],
                    axis=0,
                )
            ]
            u += [np.expand_dims(strategies[strat_ind], 2)]

        return (np.concatenate(x, axis=0), np.concatenate(u, axis=0))


class _RL(_Controller):
    def __init__(self, horizon=24, past_steps=None, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        horizon : int, optional
            Horizon over which to predict system responses, in # timepoints.
            The default is 12.
        past_steps : int, optional
            Number of past steps to feed into the prediction model. If None,
            the whole history of the cell is fed into the model.
            The default is None.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.past_steps = past_steps

    def get_strategy(self, inputs, objectives):

        # Compile array to feed into model:
        x = self.compile_x(inputs, objectives)

        # Predict:
        yhat = self.model.predict(x)

        return yhat > 0.5


class MPLRL(_RL):
    """
    Reinforcement-Learning controller that issues a single control strategy
    without the need for optimization.

    Attributes
    ----------
    model : Keras model object.
        LSTM encoder-decoder as defined in models.py.

    Methods
    -------
    __init__ :
        Instanciation.
    compile_x :
        Compile X array to do predictions over, formatted for the LSTM model.
    """

    def __init__(self, model_file=None, hidden_layers=10, features=2, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str or None, optional
            Filepath to model weights file. If None, the following model file
            will be used: 'models/mlp/mlp%d_optimized.hdf5' %(self.horizon,)
            The default is None.
        hidden_layers : int, optional
            Number of hidden layers in the MLP model. Must match weights file.
            The default is 1.
        features : int, optional
            Number of observed variables (Fluorescence, cell length...)
            + control variables (only DMD input sequence).
            The default is 2.

        Returns
        -------
        None.

        """

        # Initialize parent properties:
        super().__init__(*args, **kwargs)

        # If no model file provided, load default optimized model
        if model_file is None:
            model_file = "models/mlp_strategy_learn/mlp%d_optimized.hdf5" % (
                self.horizon,
            )

        # Initialize model:
        self.model = mlp(
            hidden_layers=hidden_layers,
            features=features,
            horizon=self.horizon,
            past_steps=self.past_steps,
            activation="sigmoid",
        )
        self.model.load_weights(model_file)

    def compile_x(self, inputs, objectives):
        """
        Compile X array to do predictions over, formatted for the MLP model.

        Parameters
        ----------
        inputs : list
            List containing past observed variables (Fluorescence, cell length
            etc...) and past control inputs (DMD inputs...).
            Each list element contains the data for each cell to process in
            parallel. Each of those elements is a list containing one 2D numpy
            arrays of size variables -by- past_timepoints for past observed
            variables and one 1D numpy array of size past_timepoints for past
            control inputs.
        objectives : list
            List containing objectives for each cell to determine the strategy
            for. Size is cells -by- horizon.

        Returns
        -------
        x : 2D numpy array
            Compiled X array for MLP model. Size is
            cells -by- (past_steps * features + horizon)

        """

        x = np.empty(
            (
                len(inputs),
                inputs[0][0].shape[0] * self.past_steps
                + self.past_steps
                + self.horizon,
            ),
            dtype=float,
        )

        for c in range(len(inputs)):
            # Process objective:
            if objectives[c].shape[0] >= self.horizon:
                obj = objectives[c][: self.horizon]
            else:  # If we're reaching the end of the control objective, pad with zeros
                obj = np.concatenate(
                    (
                        objectives[c],
                        np.zeros((self.horizon - objectives[c].shape[0]), dtype=float),
                    ),
                    axis=0,
                )

            x[c] = np.concatenate(
                (
                    inputs[c][0][:, -self.past_steps :].flatten(),
                    inputs[c][1][-self.past_steps :].flatten(),
                    obj,
                ),
                axis=0,
            )

        return x


class _Optimizer:
    """
    Parent optimizer class for null optimizer, bpso...

    Attributes
    ----------
    horizon : int
        Horizon over which to predict system responses, in # timepoints.
    iterations : int
        Number of optimization iterations run since re-init.
    max_iterations : int
        Maximum number of optimization iterations.
    strategies : None
        placeholder for strategies proposed by the optimizer.

    Methods
    -------
    __init__:
        Instanciation.
    iterate:
        Run one iteration step.

    """

    def __init__(self, horizon=12, iterations=1):
        """
        Instanciation.

        Parameters
        ----------
        horizon : int, optional
            Horizon over which to predict system responses, in # timepoints.
            The default is 12.
        iterations : int, optional
            Maximum number of optimization iterations. Set to 1 for null
            optimizer.
            The default is 1.

        Returns
        -------
        None.

        """
        self.horizon = horizon
        self.iterations = 0
        self.max_iterations = iterations
        self.strategies = None

    def iterate(self, scores):
        """
        Run one iteration step.

        Parameters
        ----------
        scores : 1D or 2D numpy array of floats
            Scores of proposed strategies from previous iteration.
            If a 1D array of NaN, the optimizer will be re-initialized
            (iterations set 0).

        Returns
        -------
        3D numpy array of bools
            New proposed strategies. If max number of iterations has been
            reached, only the best strategy per cell is returned. Size is
            cells -by- strategies_per_cell -by- horizon

        """

        # Check if stopping condition is met:
        if self.iterations >= self.max_iterations:
            self.iterations = 0  # Stopping condition is met -> set to -1
            return self.best_strategy(scores)
        # Otherwise run optimization step:
        self.iterations += 1
        return self.run(scores)

    def run(self):
        pass  # To be defined in sub-classes

    def best_strategy(self):
        pass  # To be defined in sub-classes


class NullOptimizer(_Optimizer):
    """
    "Brute-force" optimizer - All possible strategies are returned.

    Attributes
    ----------
    strategies : 2D numpy array of bools
        All possible control strategies over horizon. Size is 2^horizon -by-
        horizon.

    Methods
    -------
    __init__:
        Instanciation.
    run:
        Return all strategies for each cell.
    best_strategy:
        Identify best strategy for each cell.

    """

    def __init__(self, *args, **kwargs):
        """
        Instanciation.

        Returns
        -------
        None.

        """

        # Initialize parent class (only 1 iteration to do)
        super().__init__(iterations=1, *args, **kwargs)

        # Pre-compile all strategies:
        self.strategies = np.empty(shape=(2 ** self.horizon, self.horizon), dtype=bool)
        for i in range(2 ** self.horizon):
            self.strategies[i] = [bool(i & (1 << n)) for n in range(self.horizon)]

    def run(self, scores):
        """
        Return all strategies for each cell.

        Parameters
        ----------
        scores : 1D numpy array of NaNs
            The dimension of axis 0 is used to know the number of cells and
            duplicate the strategies array.

        Returns
        -------
        3D numpy array of bools
            All possible control strategies over horizon, repeated for each
            cell. Size is cells -by- 2^horizon -by- horizon.

        """

        # Null optimizer simply returns all strategies:
        return np.repeat(self.strategies[np.newaxis], scores.shape[0], axis=0)

    def best_strategy(self, scores):
        """
        Identify best strategy for each cell.

        Parameters
        ----------
        scores : 2D numpy array of floats
            RMSE between each strategy per cell and corresponding objective.
            Size is cells -by- 2^horizon.

        Returns
        -------
        2D numpy array of bools
            Array containing best strategies for each cell. Size is
            cells -by- horizon.

        """

        # Initialize array:
        best_strategies = np.empty(
            [scores.shape[0], self.strategies.shape[1]], dtype=bool
        )

        # For each cell, return best-scoring strategy from the pre-computed strategies array:
        for score_ind in range(scores.shape[0]):
            best_strategies[score_ind, :] = self.strategies[
                np.argmin(scores[score_ind], axis=0), :
            ]

        return best_strategies


class BinaryParticleSwarmOptimizer(_Optimizer):
    """
    Binary Particle Swarm Optimizer
    Implemented as described in Sudholdt & Witt 2010
    Original paper Kennedy & Eberhart 1997

    Attributes
    ----------
    strategies : 3D numpy array of np.int8
        Selected control strategies to evaluate. Equivalent to particle
        positions in Sudholdt & Witt 2010. Size is cells -by- particles -by-
        horizon.
    num_particles : int
        Number of particles/strategies to evaluate per cell.
    velocities : 3D numpy array of floats
        Particles velocity vectors, Size is cells -by- particles -by- horizon.
    best_scores : 2D numpy array of floats
        Best scores per particle per cell so far. Size is cells -by- particles
    bests : 3D numpy array of np.int8
        Best-scoring strategies per particle per cell so far. Size is
        cells -by- particles -by- horizon.
    """

    def __init__(self, particles=10, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        particles : int, optional
            Number of particles to run / strategies to evaluate per cell.
            The default is 10.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        self.num_particles = particles

    def run(self, scores):
        """
        Run 1 iteration of the BPSO algorithm

        Parameters
        ----------
        scores : 1D or 2D numpy array of floats.
            RMSE between each strategy per cell and corresponding objective.
            Size is cells -by- particles. On first iteration, only a 1D numpy
            array of NaNs needs to be passed, the dimension of axis 0 is used
            to know the number of cells.

        Returns
        -------
        3D numpy array of bools
            Control strategies to evaluate for each cell. Size is cells -by-
            particles -by- horizon.

        """

        if self.iterations == 1:
            # Initialize particles:
            self.velocities = np.zeros(
                shape=(scores.shape[0], self.num_particles, self.horizon)
            )
            self.best_scores = np.full(
                shape=(scores.shape[0], self.num_particles), fill_value=np.inf
            )
            self.bests = np.empty(self.velocities.shape, dtype=np.int8)
        else:
            self.update_best(scores)
            self.update_velocities()

        self.update_strategies()

        return self.strategies

    def update_best(self, scores):
        """
        Update best scores and strategies for each particle.

        Parameters
        ----------
        scores : 2D numpy array of floats.
            RMSE between each strategy per cell and corresponding objective.
            Size is cells -by- particles.

        Returns
        -------
        None.

        """

        new_bests = scores < self.best_scores
        self.best_scores[new_bests] = scores[new_bests]
        self.bests[new_bests] = self.strategies[new_bests, :]

    def update_velocities(self):
        """
        Update velocities vector per particle per cell.

        Returns
        -------
        None.

        """

        # Initialize random vectors:
        r_cog = np.random.uniform(low=0.0, high=2, size=self.velocities.shape)
        r_soc = np.random.uniform(low=0.0, high=2, size=self.velocities.shape)

        # Create array of repeated global best for each independant search:
        global_best_arr = []
        for score_ind in range(self.best_scores.shape[0]):
            global_best_arr += [
                np.repeat(
                    self.bests[
                        np.newaxis, score_ind, np.argmin(self.best_scores[score_ind]), :
                    ],
                    self.bests.shape[1],
                    axis=0,
                )
            ]
        global_best_arr = np.array(global_best_arr)

        # Update velocity:
        self.velocities += np.multiply(
            r_cog, self.bests - self.strategies
        ) + np.multiply(r_soc, global_best_arr - self.strategies)

        # Clip values to vmin/vmax (limited to +/-4)
        np.clip(self.velocities, -4, 4, out=self.velocities)

    def update_strategies(self):
        """
        Update strategies/positions per particle per cell.

        Returns
        -------
        None.

        """

        # Get sigmoid of velocity:
        sig = (1 + np.exp(-self.velocities)) ** -1

        # Threshold random uniform samples by sigmoid:
        self.strategies = (
            sig > np.random.uniform(0, 1, size=self.velocities.shape)
        ).astype(np.int8)

    def best_strategy(self, scores):
        """
        Identify best strategy per cell

        Parameters
        ----------
        scores : 2D numpy array of floats
            RMSE between each strategy per cell and corresponding objective.
            Size is cells -by- particles.

        Returns
        -------
        2D numpy array of bools
            Array containing best strategies for each cell. Size is
            cells -by- horizon.

        """

        # Initialize array
        best_strategies = np.empty(
            [self.strategies.shape[0], self.strategies.shape[2]], dtype=bool
        )

        # Run through cells, get best-scoring strategy:
        for strat_ind in range(self.strategies.shape[0]):
            best_strategies[strat_ind] = self.strategies[
                strat_ind, np.argmin(scores[strat_ind], axis=0), :
            ]

        return best_strategies

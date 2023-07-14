#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of control algorithms and strategy optimizers.

Created on Mon Oct 26 12:56:30 2020

@author: jeanbaptiste
"""
import gc

import tensorflow as tf
import numpy as np

from .models import split


class _Controller:
    """
    Parent controller class for MPCs, PIs etc...

    Attributes
    ----------
    strategy : 2D numpy array of bools
        Array containing strategies for each cell over the entire
        prediction horizon. Size is cells -by- horizon.
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
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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
        self.get_strategy(inputs, objectives)

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
    strategy_optimizer : _optimizer object
        Strategy optimizer object of sub-classes of _optimizer.
    """

    def __init__(
        self, strategy_optimizer=None, *args, **kwargs
    ):
        """
        Instanciation.

        Parameters
        ----------
        strategy_optimizer : _optimizer object or None, optional
            Strategy optimizer object of sub-classes of _optimizer. If None,
            a null_optimizer object is instanciated.
            The default is None.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        self.model = None
        self.strategy_optimizer = strategy_optimizer

    def get_strategy(self, inputs, objectives):
        """
        Identify optimal strategies.

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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
        while (self.strategy_optimizer.iterations > 0):  # Optimizer sets iterations to 0 when stopping condition is met

            predictions = self.run_strategies(inputs, strategies)

            scores = self.compute_scores(predictions, objectives)

            strategies = self.strategy_optimizer.iterate(scores)
        
        self.strategy = strategies
        
        del predictions
        del scores
        del inputs
        del strategies
        gc.collect()

    def run_strategies(self, inputs, strategies):
        """
        Predict response to selected strategies

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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
        with tf.device("GPU"):
            yhat = self.model.predict(x, batch_size=1000)

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
            
            # Repeat objective in numpy array of same size as predictions:
            obj_arr = np.repeat(
                np.expand_dims(objectives[obj_ind], 0),
                predictions.shape[1],
                axis=0
                )

            # Compute RMSE:
            scores[obj_ind] = np.sqrt(
                np.mean(
                    (predictions[obj_ind] - obj_arr) ** 2, axis=1
                )
            )

        return scores
    
    def show_predict(self, inputs, strategies):
        """
        Predict cells response based on inputs and strategies.
        This is only for verification purposes and is not used
        in the rest of the class.

        Parameters
        ----------
        inputs : 3D numpy array of float32
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
        strategies : 2D numpy array of bools
            Array containing strategies for each cell over the entire
            prediction horizon. Size is cells -by- horizon.

        Returns
        -------
        predictions : 3D numpy array of float32
            FLuorescence predictions for each control inputs under the 
            corresponding strategies

        """
        
        x = self.compile_x(inputs, strategies[:,np.newaxis,:])
        predictions = self.model.predict(x)
        
        return predictions

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
    """

    def __init__(self, model_file=None, hidden_layers=10, features=2, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str
            Filepath to model weights file.

        Returns
        -------
        None.

        """

        # Initialize parent properties:
        super().__init__(*args, **kwargs)

        # Initialize model:
        self.model = tf.keras.models.load_model(model_file)

    def compile_x(self, inputs, strategies):
        """
        Compile X array to do predictions over, formatted for the MLP model.

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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
        inputs_reshape = (inputs.shape[0],inputs.shape[1]*inputs.shape[2])
        strategies_reshape = (strategies.shape[0]*strategies.shape[1], strategies.shape[2])
        x = np.concatenate(
            (
                np.repeat(
                    np.reshape(inputs,inputs_reshape),
                    strategies.shape[1],
                    axis=0
                    ),
                np.reshape(strategies, strategies_reshape)
                ),
            axis = 1
            )

        return x


class LSTMMPC(_MPC):
    """
    Model predictive controller based on the LSTM encoder-decoder model.
    Child class of _MPC.

    Attributes
    ----------
    model : Keras model object.
        LSTM encoder-decoder as defined in models.py.
    """

    def __init__(self, model_file=None, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str
            Filepath to model weights file.
        *args and **kwargs : See _MPC class

        Returns
        -------
        None.

        """

        # Initialize parent properties:
        super().__init__(*args, **kwargs)

        # Initialize model:
        self.model = tf.keras.models.load_model(model_file)

    def compile_x(self, inputs, strategies):
        """
        Compile X array to do predictions over, formatted for the LSTM model.

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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

        # Compile full array for all strategies:
        inputs = np.repeat(
            inputs,
            strategies.shape[1],
            axis=0,
            )
        strategies = np.reshape(
            strategies,
            (strategies.shape[0]*strategies.shape[1], strategies.shape[2])
            )

        return inputs, strategies


class SplitLSTMMPC(_MPC):
    """
    Model predictive controller based on a plit version of the LSTM 
    encoder-decoder model. Because the past only needs to be encoded once and 
    then be evaluated against hundreds of optogenetic strategies, this version
    is much faster.
    Child class of _MPC.

    Attributes
    ----------
    model : Keras model object.
        LSTM encoder-decoder as defined in models.py.
    """
    
    def __init__(self, model_file, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        model_file : str
            Filepath to model weights file.
        *args and **kwargs : See _MPC class

        Returns
        -------
        None.

        """
        
        # Initialize parent properties:
        super().__init__(*args, **kwargs)
    
        # Initialize model:
        whole_model = tf.keras.models.load_model(model_file)
        encoder, decoder = split(whole_model)
        self.encoder = encoder
        self.model = decoder

    def get_strategy(self, inputs, objectives):
        """
        Identify optimal strategies.

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
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
        
        # Here, before moving on with the strategy search, we run the encoder
        # part of the network:
        with tf.device("CPU"):
            state_h, state_c = self.encoder.predict(inputs)
        self.state_h = np.repeat(
            state_h, self.strategy_optimizer.num_particles, axis=0
            )
        self.state_c = np.repeat(
            state_c, self.strategy_optimizer.num_particles, axis=0
            )
        
        # Moving on:
        return super().get_strategy(inputs, objectives)
        
    def compile_x(self, inputs, strategies):
        """
        Compile X array to do predictions over, formatted for the LSTM model.

        Parameters
        ----------
        inputs : 3D numpy array
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
            Note: In the case of the Split LSTM, the inputs are not actually
            used in this function, because they are already processed by 
            get_strategy into the latent states.
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
        
        strategies = np.reshape(
            strategies, 
            (strategies.shape[0]*strategies.shape[1], strategies.shape[2])
            )
        
        return self.state_h, self.state_c, strategies
    
    def show_predict(self, inputs, strategies):
        """
        Predict cells response based on inputs and strategies.
        This is only for verification purposes and is not used
        in the rest of the class.

        Parameters
        ----------
        inputs : 3D numpy array of float32
            Past observed variables (Fluorescence, cell length etc...) and past 
            control inputs (DMD inputs...). Dimensions are 
            (cells, past_steps, features)
        strategies : 2D numpy array of bools
            Array containing strategies for each cell over the entire
            prediction horizon. Size is cells -by- horizon.

        Returns
        -------
        predictions : 3D numpy array of float32
            FLuorescence predictions for each control inputs under the 
            corresponding strategies

        """
        
        self.state_h, self.state_c = self.encoder.predict(inputs)
        x = self.compile_x(None, strategies[:,np.newaxis,:])
        predictions = self.model.predict(x)
        
        return predictions


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
            self.iterations = 0  # Stopping condition is met -> set to 0
            return self.best_strategy(scores)
        # Otherwise run optimization step:
        self.iterations += 1
        return self.run(scores)

    def run(self, scores):
        pass  # To be defined in sub-classes

    def best_strategy(self, scores):
        pass  # To be defined in sub-classes


class NullOptimizer(_Optimizer):
    """
    "Brute-force" optimizer - All possible strategies are returned.

    Attributes
    ----------
    strategies : 2D numpy array of bools
        All possible control strategies over horizon. Size is 2^horizon -by-
        horizon.
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
                        np.newaxis,
                        score_ind,
                        np.argmin(self.best_scores[score_ind]),
                        :,
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


class OneShotOptimizer(_Optimizer):
    """
    One-shot optimizer that runs a number of random input sequences
    once and directly returns the best result, there are no
    iterations.
    """

    def __init__(self, particles=1000, *args, **kwargs):
        """
        Instanciation.

        Parameters
        ----------
        particles : int, optional
            Number of particles to run / strategies to evaluate per cell.
            The default is 1000.

        Returns
        -------
        None.

        """
        # Initialize class (only 1 iteration to do)
        super().__init__(iterations=1, *args, **kwargs)
        self.num_particles = particles

        self.strategies = None

    def run(self, scores):
        """
        Return random strategies for each cell.

        Parameters
        ----------
        scores : 1D numpy array of NaNs
            The dimension of axis 0 is used to know the number of cells and
            duplicate the strategies array.

        Returns
        -------
        3D numpy array of bools
            Random control strategies over horizon, repeated for each
            cell. Size is cells -by- particles -by- horizon.

        """

        rng = np.random.default_rng()
        self.strategies = np.empty(
            (scores.shape[0], self.num_particles, self.horizon),
            dtype=np.float32,
        )
        for cell in range(scores.shape[0]):
            unique_strategies = rng.choice(
                2**self.horizon, size=self.num_particles, replace=False
            )
            for particle, strategy in enumerate(unique_strategies):
                self.strategies[cell, particle, :] = \
                    strategy & (1 << np.arange(self.horizon)) > 0

        # Null optimizer simply returns all strategies:
        return self.strategies

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

        best_strategies = np.empty(
            (scores.shape[0], self.horizon), dtype=np.uint8
        )
        for score_ind in range(scores.shape[0]):
            best_strategies[score_ind] = self.strategies[
                score_ind, np.argmin(scores[score_ind], axis=0), :
                ]

        return best_strategies

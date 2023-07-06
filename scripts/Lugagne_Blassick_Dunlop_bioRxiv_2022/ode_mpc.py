# -*- coding: utf-8 -*-
"""
Script for evaluation of ODE-based forecasting. Specifically, Fig 1H and S7, S8
are plotted here.

Created on Fri Apr 28 16:50:03 2023

@author: jeanbaptiste
"""
import time
import json
import pickle

from scipy.integrate import solve_ivp
from scipy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cma
import tensorflow as tf
import tensorflow.keras

import deepcellcontrol as dcc

# Datasets (zenodo archive)
datasets_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/datasets/"

# Trained models (zenodo archive):
models_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/models/"

# Save images to:
save_folder = "D:/papers/deepmpc/revisions/"

SAMPLING = 5

config = dcc.config.defaults
config["datasets_folder"] = datasets_folder
training_set, evaluation_set = dcc.data.load_datasets(config)
training_set.data_type = "raw_dataset"
evaluation_set.data_type = "raw_dataset"

#%% Class definition for ODE based model + State estimation:

class ODEModel():
    
    def __init__(self):
        
        self._init_params = {
            'c2':0.0631, # Hill normalization parameter
            'a':0.2827*80, # PcpcG2 promoter rate
            'b':0.0104, # Proteins dilution rate
            'nh':3.6655, # Hill coefficient
            'K':0.4851, # Hill threshold (sort of)
            'tau':12.001, # Response delay
            }
        "Initial ODE model parameters from Chait et al to fit to the data"
        self.fixed_params = {
            "h1":0.0710,
            "h2":0.0303
            }
        "Fixed ODE params that will not be fitted."
        self._bounds = {
            'c2':[0, np.inf],
            'a':[0, np.inf], 
            'b':[.0058/self._init_params["b"], .0462/self._init_params["b"]], # Cell division period between 15min and 2 hours
            'nh':[1/self._init_params["nh"], 6/self._init_params["nh"]], # Between 1 and 6
            'K':[0, np.inf],
            'tau':[5/self._init_params["tau"], 60/self._init_params["tau"]], # Time delay between 5 minutes and 1 hour
            }
        "Fitting boundaries"
        
        self._cma_factors = [1]*len(self._init_params)
        "Multiplication factors from the CMA fit"
        self._cma = None
        "CMA optimizer object"
        self.training_set = None
        "Dataset for model fitting"
        
        self.technical_noise = 1_000_000
        "Variance of technical (measurement) noise used for state estimation"
        
        self._save_attrs = (
            "_init_params",
            "fixed_params",
            "_bounds",
            "_cma_factors",
            "technical_noise"
            )
        "Attributes to save to / load from disk"
    
    def save(self, filename):
        data = {attr: getattr(self, attr) for attr in self._save_attrs}
        data["_cma_factors"] = list(data["_cma_factors"])
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        for attr in self._save_attrs:
            setattr(self, attr, data[attr])
        
    def params(self):
        """
        Multiply CMA factors with initial parameters to obtain ODE model
        parameters.

        Returns
        -------
        new_params : dict
            Parameters dictionary to use with the `ccasr_ode` model.

        """
        
        # Multiply initial parameters:
        new_params = {}
        for (key, value), norm in zip(self._init_params.items(), self._cma_factors):
            new_params[key] = norm*value
        
        # These parameters are fixed:
        new_params.update(self.fixed_params)
        
        return new_params

    @staticmethod
    def ccasr_ode(t, y, u, params):
        """
        ODE system of the CcaSR system from Chait et al.

        Parameters
        ----------
        t : float
            time.
        y : 1D array of floats
            System state vector. Length can be 3 or 6. If 3, only first order
            moments will be evaluated, if 6 second order moments will also be
            evaluated.
        u : float
            Light input (after tau delay has been applied).
        params : dict
            System parameters.

        Returns
        -------
        derivatives: 1D array of floats
            Time derivatives of the system. Length can be 3 or 6 depending on
            the length of `y`.

        """
        
        H, E, F = y[0], y[1], y[2]
        
        # First order moments
        L = ((params["c2"]*H)**params["nh"])/(params["K"]+(params["c2"]*H)**params["nh"])
        dH_dt = u - params["c2"]*H
        dE_dt = params["h1"] - params["h2"]*E
        dF_dt = params["a"]*E*L - params["b"]*F
        
        if len(y)==6: # Also compute second order moments
            
            E2, EF, F2 = y[3], y[4], y[5]
            
            dE2_dt = params["h1"] + 2*params["h1"]*E + params["h2"]*E - 2*params["h2"]*E2
            dEF_dt = params["h1"]*F + params["a"]*L*E2 - (params["h2"] + params["b"])*EF
            dF2_dt = params["b"]*F + params["a"]*L*E + 2*params["a"]*L*EF - 2*params["b"]*F2
            
            return np.array([dH_dt, dE_dt, dF_dt, dE2_dt, dEF_dt, dF2_dt])
            
        return np.array([dH_dt, dE_dt, dF_dt])
    
    
    def run_sequence(self, y0, sequence, t=0, u=0, stop_time = None):
        """
        Integrate system model over a sequence of light inputs.

        Parameters
        ----------
        y0 : TYPE
            DESCRIPTION.
        sequence : TYPE
            DESCRIPTION.
        t : float, optional
            Start time. The default is 0.
        u : int, optional
            Initial value for the light input. The default is 0.
        stop_time : float, optional
            Time to stop integration. If None, the integration will stop at the
            end of the light sequence. The default is None.

        Returns
        -------
        timeseries : 2D array of floats
            Integrated system state over time. Dimensions are (time, species)

        """
        
        params = self.params()
        y = y0
        sequence_times = np.array(
            [x*SAMPLING + params["tau"] for x in range(len(sequence))]
            )
        record = [y]
        stop_time = stop_time or (len(sequence)-1)*SAMPLING
        
        while t < stop_time:
            
            # Set U:
            curr_seq_ind = np.argmin(sequence_times <= t)
            if curr_seq_ind > 0:
                u = sequence[curr_seq_ind-1]
            
            # Run until next sampling or light event:
            next_sample = (t//SAMPLING + 1) * SAMPLING
            if sequence_times[curr_seq_ind] > next_sample:
                run_to = next_sample
                _record = True
            else:
                run_to = sequence_times[curr_seq_ind]
                _record = False
            
            y = solve_ivp(
                    lambda t, y: self.ccasr_ode(t, y, u, params),
                    t_span = (0,run_to-t),
                    y0 = y,
                    method = "RK45",
                    )
            y = y.y[:,-1]
            
            t = run_to
            
            if _record:
                record.append(y)
        
        return np.array(record)
    
    def batch(self, x, batch_size = 100, return_eval = False):
        """
        Training batch evaluation.

        Parameters
        ----------
        x : 1D array of floats
            Parameter multiplication factors. These values will be multiplied
            by the `_init_params` elements to obtain the ODE parameters
            dictionary.
        batch_size : int, optional
            Number of single cell samples to evaluate the parameters against. 
            The default is 100.
        return_eval : bool, optional
            Whether to return full evaluation results. The default is False.

        Returns
        -------
        rmse : float
            Root mean squared error over all batch samples.
        eval_stims : 2D array of floats
            Light input sequences for all batch samples. Dimensions are 
            (samples, time). Only returned if `return_eval` is True.
        eval_gt : 2D array of floats
            Measured fluorescence timeseries that were used as grountruth. 
            Dimensions are (samples, time). Only returned if `return_eval` is True.
        eval_res : 2D array of floats
            Simulated fluorescence timeseries from the ODE model. 
            Dimensions are (samples, time). Only returned if `return_eval` is True.

        """
        
        self._cma_factors = x
        params = self.params()
        
        rmse = []
        eval_stims, eval_gt, eval_res = [], [], []
        for b in range(batch_size):
            
            # Get random cell data:
            dataset, cell_nb = self.training_set._random_cell()
            stims = dataset["stims"][cell_nb, 37:, 0]
            groundtruth = dataset["fluo1"][cell_nb, 37:, 0].copy()
            groundtruth[groundtruth<=0] = np.nan
            if groundtruth[0] == np.nan:
                f0 = 0
            else:
                f0 = groundtruth[0]
            
            # Run:
            result = self.run_sequence(
                [0, params["h1"] / params["h2"], f0], stims
                )
            fluo_predict = np.clip(result[:,2], 0, 4095)
            
            # Error:
            rmse += [np.sqrt(np.nanmean((fluo_predict-groundtruth)**2))]
            
            if return_eval:
                eval_gt.append(groundtruth)
                eval_res.append(result[:,2])
                eval_stims.append(stims)
        
        rmse = np.nanmean(rmse)
        if return_eval:
            return rmse, eval_stims, eval_gt, eval_res
        return rmse
    
    def fit(self, training_set, batch_size = 100):
        """
        Fit the ODE model parameters to single-cell data.

        Parameters
        ----------
        training_set : dcc.data.Datasets
            Dataset object to draw samples from.
        batch_size : int, optional
            Number of single cell samples to evaluate the parameters against. 
            The default is 100.

        Returns
        -------
        None.

        """
        
        self.training_set = training_set
        
        lower, upper = [], []
        for key, value in self._bounds.items():
            lower.append(value[0])
            upper.append(value[1])
        
        self._cma = cma.CMAEvolutionStrategy(
            self._cma_factors, 1, {'seed':123, 'bounds': [lower, upper]}
            )
        self._cma.optimize(self.batch, verb_disp=1)

        self._cma_factors = self._cma.result.best
        print(f"Done! System parameters: {self.params()}")
    
    def state_estimation(self, past):
        """
        Hybrid Kalman filter as described in Chait et al.

        Parameters
        ----------
        past : 2D array of floats
            Cell past to filter. Dimensions are (time, 2). Along axis 1, first 
            index is measured fluorescence, second index is applied optogenetic
            stimulations.

        Returns
        -------
        state : 2D array of floats
            Estimated states over time. Dimensions are (time, species).

        """
        
        params = self.params()
        
        measurements = past[:,0]
        commands = past[:,-1]
        
        # Initialize:
        X = np.array([[params["h1"]/params["h2"], measurements[0]]]).T
        P = np.array([[params["h1"]/params["h2"], 0], [0, 1000]])
        h = 0
        C = np.array([[0, 0.9958]])
        R = np.array([[self.technical_noise]])

        # Recorded states lists:
        h_rec = [h]
        e_rec = [X[0,0]]
        f_rec = [X[1,0]]

        # Iterate:
        for k in range(1, len(measurements)):
            t = k*SAMPLING
            
            # Prediction:
            prediction = self.run_sequence(
                [h, X[0,0], X[1,0], P[0,0], P[0,1], P[1,1]],
                commands,
                t=t-SAMPLING,
                stop_time=t
                )
            h = prediction[1,0]
            X[0,0] = prediction[1,1]
            X[1,0] = prediction[1,2]
            P[0,0] = prediction[1,3]
            P[0,1] = prediction[1,4]
            P[1,0] = prediction[1,4]
            P[1,1] = prediction[1,5]
            
            # Update:
            K = P @ C.T @ inv(C @ P @ C.T + R)
            X = X + K @ (measurements[k] - C @ X)
            P = P - K @ C @ P
            
            h_rec.append(h)
            e_rec.append(X[0,0])
            f_rec.append(X[1,0])
        
        state = np.array((h_rec, e_rec, f_rec)).T
        
        return state
    
    def predict(self, inputs, return_state = False, return_timing = False):
        """
        Predict cells response to optogenetic stimulations.

        Parameters
        ----------
        inputs : List of 2 arrays
            Prediction inputs. The first array is the cell(s) past fluorescence 
            and past optogenetic stimulations, with dimensions 
            (cells, past_steps, 2). The second array is future optogenetic 
            stimulations, with dimensions (cells, horizon).
        return_state : bool, optional
            Whether to return estimated states. The default is False.
        return_timing : bool, optional
            Whether to return computation timing. The default is False.

        Returns
        -------
        predictions : 2D array of floats
            Predicted fluorescence levels. Dimensions are (cell, horizon).
        states : 3D array of floats
            Estimated cell past states. Dimensions are 
            (cell, past_steps, species). Returned if `return_state` is True.
        timing : dict
            Wall time spent on state estimation and prediction. Returned if
            `return_timing` is True.

        """
        
        # Format inputs:
        past, stims = inputs
        
        # Init loop variables and run through cells:
        predictions, states = [], []
        timing = {"state_estimation": 0, "prediction": 0}
        for cell in range(len(stims)):
            
            # State estimation:
            _t = time.time()
            state = self.state_estimation(past[cell])
            timing["state_estimation"] += time.time() - _t
            if return_state:
                states+=[state]
            
            # Prediction:
            sequence = np.concatenate((past[cell,:,-1], stims[cell]))
            _t = time.time()
            result = self.run_sequence(
                state[-1,:], sequence, t=past.shape[1]*SAMPLING
                )
            timing["prediction"] += time.time() - _t
            predictions += [np.clip(result[:,2], 0, 4095)]
        
        # Format outputs:
        predictions = np.array(predictions)
        output = (predictions,)
        if return_state:
            output += (np.array(states),)
        if return_timing:
            output += (timing,)
            
        return output

#%% "Train" the model: (~12 hours)

model = ODEModel()
model.fit(training_set)
model.save(save_folder + "/fitted_ode_model.json")

#%% Evaluate performance for different filtering parameters

technical_noise_values = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
h1h2_factors = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

evaluation_set.past_steps = 72
evaluation_set.batch_size = 100
inputs, groundtruth = next(evaluation_set)
groundtruth = groundtruth[:,:,0]

records_fitted = []
for technical_noise in technical_noise_values:
    print("_"*80)
    for h1h2 in h1h2_factors:
        
        model = ODEModel()
        model.load(save_folder + "/fitted_ode_model.json")
        
        model.technical_noise = technical_noise
        model.fixed_params["h1"] *= h1h2
        model.fixed_params["h2"] *= h1h2
        
        predictions, states, timing = model.predict(
            inputs, return_state=True, return_timing=True
            )
        
        rmse = np.sqrt(np.nanmean((predictions - groundtruth)**2, axis=1))
        
        records_fitted.append(
            {
                "technical_noise": technical_noise,
                "h1h2": h1h2,
                "rmse": rmse,
                "timing": timing
                }
            )
        print(f"R: {technical_noise}, h1h2: {h1h2}, RMSE: {np.mean(rmse)}")

records_chait = []
for technical_noise in technical_noise_values:
    print("_"*80)
    for h1h2 in h1h2_factors:
        
        model = ODEModel()
        # model.load(save_folder + "/fitted_ode_model.json")
        
        model.technical_noise = technical_noise
        model.fixed_params["h1"] *= h1h2
        model.fixed_params["h2"] *= h1h2
        
        predictions, states, timing = model.predict(
            inputs, return_state=True, return_timing=True
            )
        
        rmse = np.sqrt(np.nanmean((predictions - groundtruth)**2, axis=1))
        
        records_chait.append(
            {
                "technical_noise": technical_noise,
                "h1h2": h1h2,
                "rmse": rmse,
                "timing": timing
                }
            )
        print(f"R: {technical_noise}, h1h2: {h1h2}, RMSE: {np.mean(rmse)}")

#Save results to disk:
with open(save_folder + "/ODE_noiseh1h2_records_fitted.pkl", "wb") as f:
    pickle.dump(records_fitted, f)

with open(save_folder + "/ODE_noiseh1h2_records_chait.pkl", "wb") as f:
    pickle.dump(records_chait, f)

with open(save_folder + "/ODE_noiseh1h2_inputs.pkl", "wb") as f:
    pickle.dump({"inputs": inputs, "groundtruth": groundtruth}, f)
    
#%% Plot tables S3 & S4:

records = records_chait
text = np.empty((len(technical_noise_values), len(h1h2_factors)), dtype="U3")
colors = np.empty((len(technical_noise_values), len(h1h2_factors), 3), dtype=float)
rmses = [np.median(r["rmse"]) for r in records]
norm = lambda x: (x - min(rmses)) / np.ptp(rmses)
plt.figure(dpi=300)
for t, technical_noise in enumerate(technical_noise_values):
    for h, h1h2 in enumerate(h1h2_factors):
        record = next(
            (r for r in records if r["technical_noise"]==technical_noise and r["h1h2"]==h1h2)
            )
        rmse = np.median(record["rmse"])
        text[t, h] = f"{rmse:3.00f}"
        colors[t, h] = cm.Purples(.9*(1-norm(rmse)))[:3]
plt.table(
    text, 
    colors, 
    cellLoc="center", 
    rowLabels=[f"{x:1.00e}" for x in technical_noise_values],
    colLabels=[f"{x:1.00e}" for x in h1h2_factors],
    colLoc="center",
    rowLoc="right"
    )
plt.axis("off")
plt.axis("tight")
plt.savefig(save_folder + "/ODE_h1h2_records_chait.png", dpi=300)
plt.savefig(save_folder + "/ODE_h1h2_records_chait.svg", dpi=300)
plt.savefig(save_folder + "/ODE_h1h2_records_chait.pdf", dpi=300)
plt.show()
# plt.axis("image")

#%% Sample predictions:

technical_noise = 1e1
h1h2 = 1e-4

model = ODEModel()
model.load(save_folder + "/fitted_ode_model.json")

model.technical_noise = technical_noise
model.fixed_params["h1"] *= h1h2
model.fixed_params["h2"] *= h1h2

predictions, states, timing = model.predict(
    [x[:100] for x in inputs], return_state=True, return_timing=True
    )

for cell in range(len(predictions)):
    fig, ax1 = plt.subplots()
    stims = np.concatenate((inputs[0][cell,:,-1],inputs[1][cell,:]))
    dcc.utilities.OptoPlotBackground(stims, ymax = 4095)
    ax1.plot(np.concatenate((inputs[0][cell,:,0], groundtruth[cell,:])), "k")
    ax1.plot(states[cell,:,2],"b--")
    x = np.arange(inputs[0].shape[1], inputs[0].shape[1]+predictions.shape[1])
    ax1.plot(x, predictions[cell],"b")
    ax2 = ax1.twinx()
    ax2.plot(states[cell,:,1],"--", color=[.5,.5,.5])
    plt.title(cell)
    plt.show()

#%% Evaluate for 3 best filtering parameters values:

hyper_params = (
    {"technical_noise": 1e1, "h1h2": 1e-4},
    {"technical_noise": 1e2, "h1h2": 1e-4},
    {"technical_noise": 1e3, "h1h2": 1e-3},
    )

evaluation_set.past_steps = 72
evaluation_set.batch_size = 100_000
inputs, groundtruth = next(evaluation_set)
groundtruth = groundtruth[:,:,0]

records_final = []
for hparams in hyper_params:
    technical_noise = hparams["technical_noise"]
    h1h2 = hparams["h1h2"]
    
    model = ODEModel()
    model.load(save_folder + "/fitted_ode_model.json")
    
    model.technical_noise = technical_noise
    model.fixed_params["h1"] *= h1h2
    model.fixed_params["h2"] *= h1h2
    
    predictions, states, timing = model.predict(
        inputs, return_state=True, return_timing=True
        )
    
    rmse = np.sqrt(np.nanmean((predictions - groundtruth)**2, axis=1))
    
    records_final.append(
        {
            "technical_noise": technical_noise,
            "h1h2": h1h2,
            "rmse": rmse,
            "timing": timing
            }
        )
    print(f"R: {technical_noise}, h1h2: {h1h2}, RMSE: {np.mean(rmse)}")

with open(save_folder + "/ODE_final_eval_inputs.pkl", "wb") as f:
    pickle.dump({"inputs": inputs, "groundtruth": groundtruth}, f)

with open(save_folder + "/ODE_final_eval_records.pkl", "wb") as f:
    pickle.dump(records_final, f)

for record in records_final:
    print(np.median(record["rmse"]))

#%% Evaluate for LSTM-MLP & Linear regression model:

with open(save_folder + "/ODE_final_eval_inputs.pkl", "rb") as f:
    data = pickle.load(f)
    inputs = data["inputs"]
    groundtruth = data["groundtruth"]

# Normalize inputs:
fake_dataset = {}
for f, feature in enumerate(evaluation_set.features):
    fake_dataset[feature] = inputs[0][:,:,f].copy()
fake_dataset = dcc.data.Normalization().normalize(fake_dataset)
norm_inputs = np.empty_like(inputs[0])
for f, feature in enumerate(evaluation_set.features):
    norm_inputs[:,:,f] = fake_dataset[feature]

# Original LSTM:
lstmmlp_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-35_b0d0b5c3-158d-4476-926b-75b2607d6154/"
lstmmlp = tf.keras.models.load_model(lstmmlp_folder+ "/model.hdf5")
encoder, decoder = dcc.models.split(lstmmlp)
_t = time.time()
latent = encoder.predict(norm_inputs, verbose=1, batch_size=1000)
encoder_timing = time.time() - _t
_t = time.time()
predictions = decoder.predict([latent, inputs[1]], verbose=1, batch_size=1000)
decoder_timing = time.time() - _t
lstmmlp_rmse = np.sqrt(
    np.mean(((4095*predictions-groundtruth))**2,axis=1)
    )

# Trained linear regression model:
linear_folder = "D:/deepcellcontrol/assets/models/2023-05-07_16-54-00_38dab3ab-152a-4b39-b162-4743d0f5514a/"
linear = tf.keras.models.load_model(linear_folder+ "/model.hdf5")
_t = time.time()
predictions = linear.predict([norm_inputs, inputs[1]], verbose=1, batch_size=1000)
linear_timing = time.time() - _t
linear_rmse = np.sqrt(
    np.mean(((4095*predictions-groundtruth))**2,axis=1)
    )

print(np.median(lstmmlp_rmse))
print(np.median(linear_rmse))

#%% Plot Median RMSE and timing (Fig. S7):

with open(save_folder + "/ODE_final_eval_records.pkl", "rb") as f:
    records_final = pickle.load(f)

plt.figure(dpi=300)
plt.bar(0, np.median(linear_rmse), color="orange", edgecolor="k")
for r, record in enumerate(records_final):
    plt.bar(2+r, np.median(record["rmse"]), color="purple", edgecolor="k")
plt.bar(6, np.median(lstmmlp_rmse), color="blue", edgecolor="k")
plt.ylabel("Root mean square error")
plt.xticks(
    ticks = [0, 3, 6],
    labels = ["Linear\nregression", "ODE models", "Our model"]
    )
figname = f"Models_comparison_RMSE"
plt.savefig(save_folder + figname + ".png", dpi=300)
plt.savefig(save_folder + figname + ".svg", dpi=300)
plt.savefig(save_folder + figname + ".pdf", dpi=300)
plt.show()

plt.figure(dpi=300)
plt.bar(0, linear_timing/1e5, color="orange", edgecolor="k")
for r, record in enumerate(records_final):
    plt.bar(2+r-.2, record["timing"]["state_estimation"]/1e5, color="xkcd:light purple", width=.4, edgecolor="k")
    plt.bar(2+r+.2, record["timing"]["prediction"]/1e5, color="purple", width=.4, edgecolor="k")
plt.bar(6-.2, encoder_timing/1e5, color="xkcd:light blue", width=.4, edgecolor="k")
plt.bar(6+.2, decoder_timing/1e5, color="blue", width=.4, edgecolor="k")
plt.yscale("log")
plt.ylabel("Computation time (s)")
plt.xticks(
    ticks = [0, 3, 6],
    labels = ["Linear\nregression", "ODE models", "Our model"]
    )
figname = f"Models_comparison_timing"
plt.savefig(save_folder + figname + ".png", dpi=300)
plt.savefig(save_folder + figname + ".svg", dpi=300)
plt.savefig(save_folder + figname + ".pdf", dpi=300)
plt.show()

#%% Plot single-cell ODE samples (Fig. S8)

for record in records_final:

    order = np.argsort(record["rmse"])
    indexes = [order[25_000], order[50_000], order[75_000]]

    model = ODEModel()
    model.load(save_folder + "/fitted_ode_model.json")
    
    model.technical_noise = record["technical_noise"]
    model.fixed_params["h1"] *= record["h1h2"]
    model.fixed_params["h2"] *= record["h1h2"]
    
    predictions, states, timing = model.predict(
        [x[indexes] for x in inputs], return_state=True, return_timing=True
        )
    
    for p, cell in enumerate(indexes):
        fig, ax1 = plt.subplots()
        stims = np.concatenate((inputs[0][cell,:,-1],inputs[1][cell,:]))
        xpast = np.arange(-inputs[0].shape[1],0,1)/12
        xfuture = np.arange(0, predictions.shape[1])/12
        xall = np.arange(-inputs[0].shape[1], predictions.shape[1])/12
        dcc.utilities.OptoPlotBackground(stims, ymax = 4095, x=xall)
        ax1.plot([-.5/12,-.5/12], [0, 4095], color=[.7, .7, .7])
        ax1.plot(xall, np.concatenate((inputs[0][cell,:,0], groundtruth[cell,:])), "k")
        ax1.plot(xpast, states[p,:,2],"b--")
        ax1.plot(xfuture, predictions[p],"b")
        plt.ylabel("Fluorescence (a.u.)")
        plt.ylim(0,4095)
        ax2 = ax1.twinx()
        ax2.plot(xpast, states[p,:,1],"--", color=[.5,.5,.5])
        plt.ylabel("Responsiveness (a.u.)")
        plt.xlim([-6,2-1/12])
        plt.xlabel("Time (hours)")
        figname = f"ODE_samples_{(p+1)*25}percentile_h1h2{record['h1h2']:1.00e}_R{record['technical_noise']:1.00e}"
        plt.savefig(save_folder + figname + ".png", dpi=300)
        plt.savefig(save_folder + figname + ".svg", dpi=300)
        plt.savefig(save_folder + figname + ".pdf", dpi=300)
        plt.show()

#%% Plot single-cell linear regression samples (Fig. S8)

order = np.argsort(linear_rmse)
indexes = [order[25_001], order[50_000], order[75_000]]
predictions = linear.predict(
    [norm_inputs[indexes], inputs[1][indexes]], verbose=1, batch_size=1000
    )

for p, cell in enumerate(indexes):
    fig, ax1 = plt.subplots()
    stims = np.concatenate((inputs[0][cell,:,-1],inputs[1][cell,:]))
    xpast = np.arange(-inputs[0].shape[1],0,1)/12
    xfuture = np.arange(0, predictions.shape[1])/12
    xall = np.arange(-inputs[0].shape[1], predictions.shape[1])/12
    dcc.utilities.OptoPlotBackground(stims, ymax = 4095, x=xall)
    ax1.plot([-.5/12,-.5/12], [0, 4095], color=[.7, .7, .7])
    ax1.plot(xall, np.concatenate((inputs[0][cell,:,0], groundtruth[cell,:])), "k")
    ax1.plot(xfuture, predictions[p]*4095,"b")
    plt.ylabel("Fluorescence (a.u.)")
    plt.ylim(0,4095)
    plt.xlim([-6,2-1/12])
    plt.xlabel("Time (hours)")
    figname = f"LINEAR_samples_{(p+1)*25}percentile"
    plt.savefig(save_folder + figname + ".png", dpi=300)
    plt.savefig(save_folder + figname + ".svg", dpi=300)
    plt.savefig(save_folder + figname + ".pdf", dpi=300)
    plt.show()

#%% Plot single-cell LSTM-MLP samples (Fig. S8)

order = np.argsort(lstmmlp_rmse)
indexes = [order[25_000], order[50_000], order[75_000]]
predictions = lstmmlp.predict(
    [norm_inputs[indexes], inputs[1][indexes]], verbose=1, batch_size=1000
    )

for p, cell in enumerate(indexes):
    fig, ax1 = plt.subplots()
    stims = np.concatenate((inputs[0][cell,:,-1],inputs[1][cell,:]))
    xpast = np.arange(-inputs[0].shape[1],0,1)/12
    xfuture = np.arange(0, predictions.shape[1])/12
    xall = np.arange(-inputs[0].shape[1], predictions.shape[1])/12
    dcc.utilities.OptoPlotBackground(stims, ymax = 4095, x=xall)
    ax1.plot([-.5/12,-.5/12], [0, 4095], color=[.7, .7, .7])
    ax1.plot(xall, np.concatenate((inputs[0][cell,:,0], groundtruth[cell,:])), "k")
    ax1.plot(xfuture, predictions[p]*4095,"b")
    plt.ylabel("Fluorescence (a.u.)")
    plt.ylim(0,4095)
    plt.xlim([-6,2-1/12])
    plt.xlabel("Time (hours)")
    figname = f"LSTMLP_samples_{(p+1)*25}percentile"
    plt.savefig(save_folder + figname + ".png", dpi=300)
    plt.savefig(save_folder + figname + ".svg", dpi=300)
    plt.savefig(save_folder + figname + ".pdf", dpi=300)
    plt.show()

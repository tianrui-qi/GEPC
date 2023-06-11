# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:19:27 2023

@author: jeanbaptiste
"""
import tensorflow as tf
number_of_cores = 8
tf.config.threading.set_inter_op_parallelism_threads(number_of_cores)
tf.config.threading.set_intra_op_parallelism_threads(number_of_cores)

import os
import copy
import time
import uuid
import sys
import json

import numpy as np

# Make sure the proper package is used:
sys.path.insert(0,'./../')
import deepcellcontrol as dcc

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

#%% Parameters
# Load default params:
params = copy.deepcopy(dcc.config.defaults)

# Update parameters with stored JSON
if len(sys.argv) > 1:
    with open(sys.argv[-1], "r") as f:
        params.update(json.load(f))

# Folder to save training, evaluation, control results:
params["save_folder"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_simulated_{uuid.uuid4()}"
save_path = params["models_folder"]+ "/" + params["save_folder"]

# params["save_folder"] = "./assets/simulated/data/CcaSR_gillespie/2023-03-16_20-53-20_simulated_c89afa3b-0fb9-4be6-bbca-8ab745f20fcf/"
# params["past_steps"] = [36, 144]
# params["horizon"] = 8*12
# params["features"] = ["fluo1", "stims"]
# params["training_parameters"]["epochs"] = 10 # 100
# params["batch_size"] = 1000
# params["loss"] = "Huber"
# params["models_folder"] = "" # TODO: fix this in timeseries.batch_train_eval

# Load cell parameters for record-keeping:
with open(params["datasets_folder"] + "/model_parameters.json", "r") as f:
     params["cell parameters"] = json.load(f)

# Save to file:
dcc.utilities.csvrecord(params, params["centralized_records"])
with open(save_path + '/training_parameters.json','w') as f:
    json.dump(params,f, indent=4)

print(f'Save path: {save_path}')

#%% Generate Dataset object from saved data:

training_set = dcc.data.Datasets_cnn(
    [params["datasets_folder"]+"/"+params['training_sets'][0]],
    formatter = dcc.data.LSTMFormatter(params["features"]),
    parameters = params
    )

training_set.test_ratio = 0.1 # Fraction of samples that are left out of training
# Actually load data and normalize:
training_set.load()
training_set.normalize() # This step now takes a few minutes the first time it is called
training_set.data_type='normalized_dataset'

#%% Train
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Training prediction network\n" + "="*100)

# Init model:
network = dcc.models.lstm_cnn(params)

# Initialize network
model_checkpoint = ModelCheckpoint(
    save_path+'/cnn_model.hdf5',
    monitor='loss',
    verbose=1, 
    save_best_only=True
    )
early_stopping = EarlyStopping(
    monitor='loss', mode='min', verbose=1, patience=100
    )
callbacks = [model_checkpoint, early_stopping]

history = network.fit_generator(
    training_set,
    steps_per_epoch=params["training_parameters"]["steps_per_epoch"],
    epochs=params["training_parameters"]["epochs"],
    callbacks=callbacks
    )
network.load_weights(save_path+'/cnn_model.hdf5')


#%% Evaluate predictor:
print("="*100 + f"\n{time.strftime('%Y-%m-%d_%H-%M-%S')}: Evaluating prediction network\n" + "="*100)
os.makedirs(save_path + "/evaluation/", exist_ok = True)

def gaussian_histogram(realizations):
    "Compile histogram image with a gaussian kernel"
    
    img = np.zeros((realizations.shape[1],params["cnn_bins"]), dtype=np.float64)
    for r, real in enumerate(realizations):
        img += dcc.data.gaussian_img(real, params["cnn_bins"])
    
    return img/realizations.shape[0]

# Load evaluation data:
eval_folder = params["datasets_folder"]+"/evaluation_set/"
past = np.load(eval_folder+"/past_fluo.npy")
future = np.load(eval_folder+"/futures_fluo.npy")
stims = np.load(eval_folder+"/stims.npy")
cut = past.shape[1]

# Predict probability landscapes:
predictions = network.predict(
    (
        np.stack((past[:,:cut]/4095, stims[:,:cut]), axis=-1),
        stims[:,cut:cut+params["horizon"]]
        )
    )

# Plot results and save them to disk:
# for c in range(min(past.shape[0], 100)):
for c in range(10):
    
    plt.figure(figsize=(8,14.5), dpi = 300)
    
    plt.subplot(3,1,1)
    x = [t/12. for t in range(stims.shape[1]+1)]
    dcc.utilities.OptoPlotBackground(stims[c], ymax=4095, x=x)
    x = [t/12. for t in range(cut)]
    plt.plot(x, past[c], color="b", alpha=1, lw=2)
    x = [t/12. for t in range(cut,stims.shape[1])]
    for f in future[c]:
        plt.plot(x, f, color="b", alpha=.05, lw=.5)
    plt.xlabel("time (hours)")
    plt.ylabel("fluorescence")
    plt.ylim(0,4095)
    plt.xlim(0,stims.shape[1]/12)
    plt.title(f"sample {c}")
    
    plt.subplot(3,2,3)
    toplot = predictions[c,:,:,0].copy()
    toplot[toplot<1e-4] = np.nan
    plt.imshow(
        np.transpose(toplot),vmin=1e-4, vmax=.05
        )
    plt.gca().invert_yaxis()
    plt.xlabel("timepoints")
    plt.ylabel("fluorescence")
    plt.title("Prediction")
    
    plt.subplot(3,2,4)
    histim = gaussian_histogram(future[c])
    toplot = histim[:, :params["horizon"]-1].copy()
    toplot[toplot<1e-4] = np.nan
    plt.imshow(np.transpose(toplot), vmin=1e-4, vmax=.05)
    plt.gca().invert_yaxis()
    plt.xlabel("timepoints")
    plt.title("Actual")
    
    plt.subplot(3,1,3)
    plt.plot(predictions[c,-1,:,0], label = "Prediction")
    plt.plot(histim[params["horizon"]-1,:], label="Actual")
    plt.title("Distribution at last timepoint")
    plt.xlabel("fluorescence")
    plt.legend()
    
    plt.savefig(save_path + f"/evaluation/cnn_eval_sample_{c}.png")
    plt.show()

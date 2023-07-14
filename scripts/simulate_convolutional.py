# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:19:27 2023

@author: jeanbaptiste
"""
import os
import copy

import numpy as np

import deepcellcontrol as dcc

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

#%% Parameters
# Load default params:
params = copy.deepcopy(dcc.config.defaults)
params["save_folder"] = "D:/deepcellcontrol/assets/autoactivation/"
params["past_steps"] = [36, 144]
params["horizon"] = 8*12
params["features"] = ["fluo1", "stims"]
params["training_parameters"]["epochs"] = 100
params["batch_size"] = 1000
params["loss"] = "Huber"
params["models_folder"] = "" # TODO: fix this in timeseries.batch_train_eval

os.makedirs(params["save_folder"], exist_ok = True)

#%% Generate Dataset object from saved data:

training_set = dcc.data.Datasets_cnn(
    [params["save_folder"]+"training_set/"],
    formatter = dcc.data.LSTMFormatter(params["features"]),
    parameters = params
    )

training_set.test_ratio = 0.1 # Fraction of samples that are left out of training
# Actually load data and normalize:
training_set.load()
training_set.normalize() # This step now takes a few minutes the first time it is called
training_set.data_type='normalized_dataset'

#%% Illustrate samples:

batch = next(training_set)

#%%
s = 72
# s+=1

past = batch[0][0][s]
future_stims = batch[0][1][s]
groundtruth = batch[1][s]

x = np.arange(-past.shape[0], 0)/12
dcc.utilities.OptoPlotBackground(past[:,-1], x=x)
plt.plot(x, past[:,0],'k')
# plt.xlim(-3,0)
plt.savefig(params["save_folder"] + "/SI_figs/formatting_s{s:03d}_past.svg")
plt.show()

x = np.arange(0, future_stims.shape[0])/12
dcc.utilities.OptoPlotBackground(future_stims, x=x)
plt.savefig(params["save_folder"] + "/SI_figs/formatting_s{s:03d}_past.svg")
plt.show()

plt.imshow(np.transpose(groundtruth), cmap="inferno")
plt.gca().invert_yaxis()
# plt.title(s)
plt.savefig(params["save_folder"] + "/fig4_groundtruth.svg")
plt.show()



#%% Train

# Init model:
network = dcc.models.lstm_cnn(params)

# Initialize network
model_checkpoint = ModelCheckpoint(
    params["save_folder"]+'/cnn_model.hdf5',
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
network.load_weights(params["save_folder"]+'/cnn_model.hdf5')


#%% Evaluate:

def gaussian_histogram(realizations):
    "Compile histogram image with a gaussian kernel"
    
    img = np.zeros((realizations.shape[1],params["cnn_bins"]), dtype=np.float64)
    for r, real in enumerate(realizations):
        img += dcc.data.gaussian_img(real, params["cnn_bins"])
    
    return img/realizations.shape[0]

# Load evaluation data:
eval_folder = params["save_folder"]+"/evaluation_set/"
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
for c in range(min(past.shape[0], 100)):
    
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
    
    # plt.savefig(eval_folder+f"/cnn_eval_sample_{c}.png")
    plt.show()

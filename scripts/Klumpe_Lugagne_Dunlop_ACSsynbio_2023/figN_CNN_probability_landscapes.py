# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:29:22 2023

@author: jeanbaptiste
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

import tensorflow as tf
import tensorflow.keras

import deepcellcontrol as dcc


save_folder = "D:/deepcellcontrol/assets/autoactivation/"
with open(save_folder+"/cnn_params.json", "r") as f:
    params = json.load(f)

light_blue = [.1, .3, 1]

def gaussian_histogram(realizations):
    "Compile histogram image with a gaussian kernel"
    
    img = np.zeros((realizations.shape[1],params["cnn_bins"]), dtype=np.float64)
    for r, real in enumerate(realizations):
        img += dcc.data.gaussian_img(real, params["cnn_bins"])
    
    return img/realizations.shape[0]

def landscape_plot(landscape, stims):
    
    stims = list(stims)
    stims.append(stims[-1])
    dcc.utilities.OptoPlotBackground(
        stims, ymin=-4, ymax=-.5, x=np.arange(len(stims))
        )
    plt.imshow(np.transpose(landscape), vmax=.05, cmap="inferno")
    plt.gca().invert_yaxis()
    
    ylabels = [0, 1000, 2000, 3000, 4000]
    yticks = [l*landscape.shape[0]/4095 for l in ylabels]
    xticks = list(range(0,landscape.shape[0],24))
    xlabels = [l/12 for l in xticks]
    plt.yticks(yticks, ylabels)
    plt.xticks(xticks, xlabels)
    
    plt.ylim(-4, landscape.shape[1])
    plt.xlim(0,landscape.shape[0])
    
    plt.xlabel("time (h)")
    plt.ylabel("fluorescence (a.u.)")
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

def realizations_plot(past, future, stims, prediction):
    
    cut = past.shape[0]
    
    x = np.arange(-cut, -cut+stims.shape[0]+1)/12
    dcc.utilities.OptoPlotBackground(stims, ymax=4095, x=x)
    
    x = np.arange(-cut, 0)/12
    plt.plot(x, past, color="k", alpha=1, lw=2)
    
    x = np.arange(0,-cut+stims.shape[0])/12
    for f in future:
        plt.plot(x, f, color="grey", alpha=.05, lw=1)
    
    plt.plot(x, prediction, color=light_blue, alpha=1, lw=2)
    
    plt.plot([-.5/12,-.5/12],[0, 4095], 'k--', lw=.5)
    plt.xlabel("time (hours)")
    plt.ylabel("fluorescence")
    plt.ylim(0,4095)
    plt.xlim(-3,(-cut+stims.shape[0])/12)
    
    

#%% Load evaluation data and predict landscapes:

eval_folder = save_folder+"/evaluation_set/"
past = np.load(eval_folder+"/past_fluo.npy")
future = np.load(eval_folder+"/futures_fluo.npy")
stims = np.load(eval_folder+"/stims.npy")
cut = past.shape[1]

cnn_network = tf.keras.models.load_model(save_folder+"/cnn_model.hdf5")
mlp_network = tf.keras.models.load_model(save_folder+"/mlp_model.hdf5")

mlp_predictions = mlp_network.predict(
    (
        np.stack((past[:,:cut]/4095, stims[:,:cut]), axis=-1),
        stims[:,cut:cut+params["horizon"]]
        ),
    verbose = 1
    )

# Predict probability landscapes:
cnn_predictions = cnn_network.predict(
    (
        np.stack((past[:,:cut]/4095, stims[:,:cut]), axis=-1),
        stims[:,cut:cut+params["horizon"]]
        ),
    verbose = 1
    )

#%% Plot

samples = [44, 96]

# Plot results and save them to disk:
for c in samples:
    
    realizations_plot(past[c], future[c], stims[c], mlp_predictions[c]*4095)
    plt.show()
    
    landscape_plot(cnn_predictions[c,:,:,0], stims[c, cut:cut+params["horizon"]])
    plt.title("Prediction")
    plt.show()
    
    histim = gaussian_histogram(future[c])
    landscape_plot(histim, stims[c, cut:cut+params["horizon"]])
    plt.title("Actual")
    plt.show()
    
    plt.plot(histim[params["horizon"]-1,:], 'k', label="Actual")
    plt.plot(cnn_predictions[c,-1,:,0], color=light_blue, label="Prediction")
    plt.title("Distribution at last timepoint")
    plt.xlabel("fluorescence")
    plt.legend()
    plt.show()
    
landscape_plot(cnn_predictions[c,:,:,0], stims[c, cut:cut+params["horizon"]])
plt.gca().set_visible(False)
plt.colorbar(extend="max", label="Probability")
plt.show()

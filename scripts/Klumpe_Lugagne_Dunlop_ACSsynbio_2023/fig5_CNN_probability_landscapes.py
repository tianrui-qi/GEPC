# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:29:22 2023

@author: jeanbaptiste
"""
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
import tensorflow as tf
import tensorflow.keras
from tqdm import tqdm

import deepcellcontrol as dcc


matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = 14

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

def landscape_plot(landscape, stims, vmax=.05):
    
    stims = list(stims)
    stims.append(stims[-1])
    dcc.utilities.OptoPlotBackground(
        stims, ymin=-4, ymax=-.5, x=np.arange(len(stims))
        )
    plt.imshow(np.transpose(landscape), vmax=vmax, cmap="inferno")
    plt.gca().invert_yaxis()
    
    ylabels = [0, 1000, 2000, 3000, 4000]
    yticks = [l*landscape.shape[0]/4095 for l in ylabels]
    xticks = list(range(0,landscape.shape[0]+1,24))
    xlabels = [int(l/12) for l in xticks]
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

#%% Generate gaussian histograms for future realizations and save to disk:
# This should only be run once (slow)

histograms = np.zeros((future.shape[0], future.shape[2], 96), dtype=np.float64)
for c, realizations in tqdm(enumerate(future)):
    histograms[c] = gaussian_histogram(realizations)

np.save(eval_folder+"/futures_fluo_histograms.npy", histograms)

#%% Plot

histograms = np.load(eval_folder+"/futures_fluo_histograms.npy")

samples = [44, 96]

# Plot results and save them to disk:
for c in samples:
    
    realizations_plot(past[c], future[c], stims[c], mlp_predictions[c]*4095)
    plt.savefig(save_folder + f"/fig4_sample{c:03d}_realizations.svg")
    plt.show()
    
    landscape_plot(cnn_predictions[c,:,:,0], stims[c, cut:cut+params["horizon"]])
    plt.title("Prediction")
    plt.savefig(save_folder + f"/fig4_sample{c:03d}_prediction.svg")
    plt.show()
    
    # histim = gaussian_histogram(future[c])
    landscape_plot(histograms[c], stims[c, cut:cut+params["horizon"]])
    plt.title("Actual")
    plt.savefig(save_folder + f"/fig4_sample{c:03d}_actual.svg")
    plt.show()
        
    plt.plot(np.mean(histograms[c,-12:,:], axis=0), 'k', label="Actual")
    plt.plot(np.mean(cnn_predictions[c,-12:,:,0], axis=0), color=light_blue, label="Prediction")
    plt.title("Distribution at last timepoint")
    plt.xlabel("fluorescence")
    plt.legend()
    plt.xlim(0,96)
    plt.ylim(0,.06)
    plt.savefig(save_folder + f"/fig4_sample{c:03d}_lastdistro.svg")
    plt.show()
    
landscape_plot(cnn_predictions[c,:,:,0], stims[c, cut:cut+params["horizon"]])
plt.gca().set_visible(False)
plt.colorbar(extend="max", label="Probability")
plt.savefig(save_folder + "/fig4_colorbar.svg")
plt.show()

#%% Compute Jensen-Shannon distance

dist = np.empty(cnn_predictions.shape[0:2])
distmlp = np.empty(cnn_predictions.shape[0:2])
for c in range(cnn_predictions.shape[0]):
    print(c)
    pred = cnn_predictions[c,:,:,0]
    actual = histograms[c]
    mlp_pred = dcc.data.gaussian_img(mlp_predictions[c]*4096, params["cnn_bins"])
    distmlp[c] = distance.jensenshannon(np.transpose(mlp_pred), np.transpose(actual), base=2)
    dist[c] = distance.jensenshannon(np.transpose(pred), np.transpose(actual), base=2)
    # for t in range(96):
    #     distmlp[c, t] = np.nanmean(kl_div(mlp_pred[t], actual[t]))
    #     dist[c, t] = np.nanmean(kl_div(pred[t], actual[t]))

dcc.utilities.plotq(dist, color="orange")
dcc.utilities.plotq(distmlp, color="b")
plt.xlim(0,8)
plt.xlabel("time (hours)")
plt.ylabel("Jensen-Shannon distance")
plt.ylim(0,1)
plt.grid("both", axis="y")
plt.savefig(save_folder + "/fig4_jensenshannon.svg", dpi=300)

#%% Dip test & histogram
import diptest

dips, pvals = [], []
for c in range(1000):
    distro = np.mean(histograms[c,params["horizon"]-12:,:], axis = 0)*10000
    data = []
    # Sample distro with uniform noise to avoid integer effects:
    for x, prob in enumerate(distro):
        data += [np.random.random(size=int(prob))+x]
    dip, pval = diptest.diptest(np.concatenate(data))
    # landscape_plot(histograms[c], stims[c, cut:cut+params["horizon"]])
    # plt.title(f"Sample {c} - dip={dip:.3g} - p={pval:.3g}")
    # plt.show()
    dips+=[dip]
    pvals+=[pval]
    

#%% Dip values illustrations
# dip_refs = np.logspace(np.log10(min(dips)),np.log10(max(dips)),6)
dip_thresh = [.003, .006]


dip_refs = [.0018,dip_thresh[0],(dip_thresh[0]+dip_thresh[1])/2,dip_thresh[1],.03]
dip_refs_c = []
for ref in dip_refs:
    dip_refs_c += [(np.abs(np.array(dips) - ref)).argmin()]

plt.hist(
    dips,
    np.logspace(np.log10(min(dips)),np.log10(max(dips)),30),
    histtype="step",
    color="k"
    )
plt.xscale("log")
yl = plt.ylim()
for ref in dip_refs:
    plt.plot([ref, ref], yl, "k--", lw=.2)
plt.ylim(yl)
plt.xlabel("Dip index")
plt.ylabel("count")
plt.savefig(save_folder + "SI_figs/diptest_histogram.svg", dpi=300)
plt.show()

xlabels = [0,1000,2000,3000,4000]
for c in dip_refs_c:
    plt.plot(np.mean(histograms[c,-12:,:], axis=0), 'k', label="Actual")
    plt.title(f"Sample {c} - dip={dips[c]:.3g}")   
    plt.xticks([x*96/4095 for x in xlabels], xlabels)
    plt.xlabel("fluorescence")
    plt.ylabel("frequency")
    plt.savefig(save_folder + f"SI_figs/diptest_dipsample_{c:03d}.svg", dpi=300)
    plt.show()
 
#%% Bimodal vs Unimodal errors, first hour vs last hour
from scipy.stats import spearmanr, pearsonr

x_bi, y_bi, x_uni, y_uni, w_bi, w_uni = [], [], [], [], [], []
for c in range(1000):
    if dips[c]>=.006:
        x_bi.append(np.mean(dist[c,:12]))
        y_bi.append(np.mean(dist[c,-12:]))
        w_bi.append(np.mean(dist[c,:]))
    elif dips[c]<=.003:
        x_uni.append(np.mean(dist[c,:12]))
        y_uni.append(np.mean(dist[c,-12:]))
        w_uni.append(np.mean(dist[c,:]))

colors = ["k", "r"]

plt.scatter(x=x_bi, y=y_bi, alpha=.5, color = "r", zorder = 100, label="bimodal")
plt.scatter(x=x_uni, y=y_uni, alpha=.5, color = "k", zorder = 10, label="unimodal")
plt.xlabel("Jensen-Shannon distance, first hour")
plt.ylabel("Jensen-Shannon distance, last hour")
plt.legend()
plt.gca().set_aspect('equal', 'box')
plt.grid("both", "both")
plt.savefig(save_folder + "/fig4_state_estim_propagates.svg", dpi=300)

# Pearson coeffs:
res_bi = pearsonr(x_bi, y_bi)
res_uni = pearsonr(x_uni, y_uni)

#%% Worst bimodal prediction plot:

c = np.argmax(np.mean(dist, axis=1) * dips>.006)
d = np.mean(dist, axis=1)[c]

landscape_plot(cnn_predictions[c,:,:,0], stims[c, cut:cut+params["horizon"]])
plt.title(f"Sample {c} - JS {d:.3f} - Prediction")
plt.savefig(save_folder + f"/SI_figs/worstsample_prediction.svg")
plt.show()

landscape_plot(histograms[c], stims[c, cut:cut+params["horizon"]])
plt.title(f"Sample {c} - JS {d:.3f} - Actual")
plt.savefig(save_folder + f"/SI_figs/worstsample_actual.svg")
plt.show()

plt.plot(np.mean(histograms[c,-12:,:], axis=0), 'k', label="Actual")
plt.plot(np.mean(cnn_predictions[c,-12:,:,0], axis=0), color=light_blue, label="Prediction")
plt.title("Distribution at last timepoint")
plt.xlabel("fluorescence")
plt.legend()
plt.xlim(0,96)
plt.ylim(0,.06)
plt.savefig(save_folder + f"/SI_figs/worstsample_lastdistro.svg")
plt.show()

#%% Gaussian kernel image formatting illustration
import copy
params = copy.deepcopy(dcc.config.defaults)
params["save_folder"] = "D:/deepcellcontrol/assets/autoactivation/"
params["past_steps"] = [36, 144]
params["horizon"] = 8*12
params["features"] = ["fluo1", "stims"]
params["models_folder"] = "" # TODO: fix this in timeseries.batch_train_eval

training_set = dcc.data.Datasets(
    [params["save_folder"]+"training_set/"],
    formatter = dcc.data.LSTMFormatter(params["features"]),
    parameters = params
    )

training_set.test_ratio = 0.1 # Fraction of samples that are left out of training
# Actually load data and normalize:
training_set.load()
training_set.normalize() # This step now takes a few minutes the first time it is called
training_set.data_type='normalized_dataset'

batch = next(training_set)

for s in range(100):
    ticks = [0, 1000, 2000, 3000, 4000]
    
    past = batch[0][0][s]
    future_stims = batch[0][1][s]
    groundtruth = batch[1][s]
    
    
    plt.figure(figsize=(12,4), dpi=300)
    plt.subplot(1,3,1)
    x = np.arange(-past.shape[0], 0)/12
    dcc.utilities.OptoPlotBackground(past[:,-1], x=x, ymax=4095)
    plt.plot(x, past[:,0]*4095,'k')
    plt.xlim(-3,-1/12)
    plt.ylim(0,4095)
    plt.savefig(params["save_folder"] + "/SI_figs/formatting_s{s:03d}_past.svg")
    plt.show()
    
    plt.subplot(1,3,2)
    x = np.arange(0, future_stims.shape[0])/12
    dcc.utilities.OptoPlotBackground(future_stims, x=x, ymax=4095)
    plt.plot(x, groundtruth*4095)
    plt.ylim(0,4095)
    plt.xlim(0,8)
    plt.yticks(ticks = ticks, labels=['']*len(ticks))
    plt.savefig(params["save_folder"] + "/SI_figs/formatting_s{s:03d}_past.svg")
    plt.show()
    
    plt.subplot(1,3,3)
    landscape_plot(dcc.data.gaussian_img(groundtruth[:,0]*4095, 96), future_stims, vmax=None)
    plt.colorbar()
    plt.ylim(0,96)
    plt.xlabel()
    plt.title(s)
    plt.suptitle(s)
    plt.savefig(params["save_folder"] + f"/SI_figs/formatting_s{s:03d}.svg")
    plt.show()

#%% Plot models:
import copy
from tensorflow.keras.utils import plot_model

params = copy.deepcopy(dcc.config.defaults)
params["save_folder"] = "D:/deepcellcontrol/assets/autoactivation/"
params["past_steps"] = [36, 144]
params["horizon"] = 8*12
params["features"] = ["fluo1", "stims"]

network = dcc.models.lstm_cnn(params)

plot_model(
    network,
    to_file = params["save_folder"] + f"/SI_figs/cnn_model.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=96,
    )

# -*- coding: utf-8 -*-
"""
This script generates plots for Figure 2, and SI Figures 1 & 2

Created on Tue Jul 12 16:16:25 2022

@author: jeanbaptiste
"""
import json
import pickle
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

import deepcellcontrol as dcc

# Raw datasets path:
images_raw = "Y:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/"

# Save images to:
save_folder = "C:/Users/Administrator/jb/deepmpc_paper/figure2/"

# Trained models:
model_folder_dict = {
    12: "Z:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-39_98118f0c-60c8-42a0-b873-6749a2128406",
    24: "Z:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-35_b0d0b5c3-158d-4476-926b-75b2607d6154",
    36: "Z:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-35_2d5eec41-ec65-4f75-afb1-f526a2dbbbc0",
    48: "Z:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_20-14-35_a9392a1f-6163-49c4-bae0-9e0894d8198f"
    }

# Load datasets, model etc:
def load_model(model_folder):
    
    with open(model_folder + "/training_parameters.json","r") as f:
        params = json.load(f)

    params["datasets_folder"] = dcc.config.defaults["datasets_folder"]

    # Load dataset:
    dataset, eval_set = dcc.data.load_datasets(params)

    # Load best eval model:
    model = tf.keras.models.load_model(model_folder + "/model_besteval.hdf5")
    
    return params, dataset, eval_set, model

#%% Evaluate performance on all horizons:

for horizon, model_folder in model_folder_dict.items():
    params, dataset, eval_set, model = load_model(model_folder)

    metrics, eval_d = dcc.timeseries.evaluate(
        eval_set, model, batch_size=100_000, num_batches = 1, return_eval=True
        )
    with open(save_folder+f"eval_d_horizon{horizon}.pkl","wb") as f:
        pickle.dump(eval_d, f)


#%% Panel A - Kymograph
# Note: this panel will not work if you do not have access to the raw images 
# data. Also it requires DeLTA.

import sys
sys.path.append("D:/delta")
import delta

# Parameters:
params, dataset, eval_set, model = load_model(model_folder_dict[24])
subset = dataset.data[-1]
raw_dataset = subset["raw_dataset"]
cell_nb = 285
stims = raw_dataset["stims"][cell_nb]
interval = 4

# Figure out which pos and roi it is:
experiment = images_raw + os.path.basename(subset["experiment"])
pos_list = os.listdir(experiment+"/delta_results")
pos_list.sort()
cell_count = 0
# pos_nb = 10, roi_nb = 10
for pos_nb, pos_file in enumerate(pos_list):
    print(pos_nb)
    pos = delta.pipeline.load_position(experiment+"/delta_results/"+pos_file)
    cell_count+=len(pos.rois)
    if cell_count > cell_nb:
        break
roi_nb = cell_nb-(cell_count-len(pos.rois))
roi = pos.rois[roi_nb]

# Collect raw images:
img_stack = []
fluo_stack = []
for f in range(0,len(stims), interval):
    img_stack += [cv2.imread(
        experiment+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
        cv2.IMREAD_ANYDEPTH
        )]
    fluo_stack += [cv2.imread(
        experiment+f"/pos{pos_nb+1:04d}/chan02_frame{f+1:06d}.tif",
        cv2.IMREAD_ANYDEPTH
        )]
drift = (pos.drift_values[0][0::interval], pos.drift_values[1][0::interval])
fluo_stack, _ = delta.utilities.driftcorr(np.array(fluo_stack), drift=drift)
img_stack, _ = delta.utilities.driftcorr(np.array(img_stack), drift=drift)

kymograph_fluo = []
kymograph_img = []
for f in range(0,len(stims), interval):
    
    # Mother cell contour:
    mother = roi.label_stack[f] == roi.lineage.cellnumbers[f][0]+1
    kernel = np.ones((3,3),np.uint8)
    mother = cv2.dilate(mother.astype(np.uint8),kernel,iterations = 2) # More visible
    cnt = delta.utilities.find_contours(mother.astype(np.uint8))
    
    # Colored chamber image (fluo):
    frame = img_stack[int(f/interval)]
    frame = delta.utilities.rangescale(frame, [0,255]).astype(np.uint8)
    chamber_img = delta.utilities.cropbox(frame, roi.box)
    chamber_img = np.repeat(chamber_img[:,:,np.newaxis], 3, axis=2)
    # Add in mother cell contour:
    chamber_img = cv2.drawContours(chamber_img, cnt, 0, [255,255,255], thickness=4)
    # Append
    kymograph_img += [chamber_img]
    
    # Colored chamber image (fluo):
    frame = fluo_stack[int(f/interval)]
    chamber_img = delta.utilities.cropbox(frame, roi.box)
    chamber_img = dcc.utilities.color_img(chamber_img, vmin=0.01, vmax=.9)
    chamber_img = (chamber_img*255).astype(np.uint8)
    # Add in mother cell contour:
    chamber_img = cv2.drawContours(chamber_img, cnt, 0, [255,255,255], thickness=4)
    # Append
    kymograph_fluo += [chamber_img]

# Concatenate into a single image, save to disk:
kymograph_fluo = np.concatenate(kymograph_fluo, axis=1)
cv2.imwrite(save_folder+"Panel_A_kymographfluo.tif", kymograph_fluo[:,:,::-1])
kymograph_img = np.concatenate(kymograph_img, axis=1)
cv2.imwrite(save_folder+"Panel_A_kymographphase.tif", kymograph_img[:,:,::-1])

#%% Panel B - Dataset sample

# Parameters:
params, dataset, eval_set, model = load_model(model_folder_dict[24])
subset = dataset.data[-1]
raw_dataset = subset["raw_dataset"]
cell_nb = 285
stims = raw_dataset["stims"][cell_nb]

# X vector (time points)
x = np.arange(0, raw_dataset["stims"][cell_nb].shape[0], dtype = np.float32)/12

plt.figure(figsize=(6, 12), dpi=300)

# Fluo & stims plot:
plt.subplot(3,1,1)

# Plot stimulations:
dcc.utilities.OptoPlotBackground(
    stims,
    x = x,
    ymin = 0,
    ymax = 4095,
    )

# Plot fluorescence:
plt.plot(x,raw_dataset["fluo1"][cell_nb],"k",label="Mother")
plt.ylabel("Fluorescence")
plt.xlim([x[0], x[-1]])
plt.ylim(0, 4095)

# Mean chamber fluorescence
plt.subplot(9,1,4)
plt.plot(x,raw_dataset["chamber_mean_fluo1"][cell_nb],"k",label="Mean")
plt.ylabel("Mean chamber Fluo.")
plt.xlim([x[0], x[-1]])
plt.ylim(0, 4095)
plt.grid(which="both", axis="both")

# Chamber fluorescence std dev:
plt.subplot(9,1,5)
plt.plot(x,raw_dataset["chamber_std_fluo1"][cell_nb],"k")
plt.ylabel("Std. Fluo")
plt.xlim([x[0], x[-1]])
plt.grid(which="both", axis="both")

# Area:
plt.subplot(9,1,6)
plt.plot(x,raw_dataset["area"][cell_nb],"k",label="area")
plt.ylabel('area, pixels', color='k')
plt.xlim([x[0], x[-1]])
plt.grid(which="both", axis="both")

# Cells in chamber
plt.subplot(9,1,7)
plt.step(x,raw_dataset["cell_count"][cell_nb],"k")
plt.ylabel("Cells in chamber")
plt.xlim([x[0], x[-1]])
plt.yticks(list(range(*[int(x) for x in plt.ylim()])))
plt.grid(which="both", axis="both")

# Neighbor stims
xx = np.repeat(x,2)[1:]
yy = np.repeat(raw_dataset["neighbor_stims"][cell_nb],2)[:-1]
plt.subplot(9,1,8)
ax = plt.gca()
ax.set_facecolor('xkcd:pale rose')
plt.fill_between(xx,yy*2,"k", facecolor='xkcd:light mint')
plt.ylabel("Neighbor stims")
plt.xlim([x[0], x[-1]])
plt.xlabel("time (hours)")
plt.ylim(0, 2)

# Image sharpness
plt.subplot(9,1,9)
plt.plot(x,raw_dataset["sharpness"][cell_nb],"k")
plt.ylabel("Sharpness")
plt.xlim([x[0], x[-1]])
plt.grid(which="both", axis="both")

plt.savefig(save_folder+"Panel_B_cellfeatures.png", dpi=300)
plt.savefig(save_folder+"Panel_B_cellfeatures.svg", dpi=300)
plt.savefig(save_folder+"Panel_B_cellfeatures.pdf", dpi=300)
plt.show()

#%% Load 2-hour horizon evaluation:

with open(save_folder+f"eval_d_horizon24.pkl","rb") as f:
    eval_d = pickle.load(f)

# RMSE over prediction horizon:
rmse = np.sqrt(
    np.mean((4095*(eval_d["prediction"]-eval_d["groundtruth"]))**2,axis=1)
    )
rmse_order = np.argsort(rmse)

def prediction_plot(eval_d, pred_nb):
    
    # Single-cell past features input:
    features = eval_d["input"][0][pred_nb]
    # Future stimulations
    future_stims = eval_d["input"][1][pred_nb]
    # Fluorescence is feature 1 by convention:
    fluo = features[-36:,0] # Also we only show the past 3 hours
    # X axis vectors (in hours):
    x = np.arange(-len(fluo),len(future_stims),1)/12
    x_future = np.arange(0,len(future_stims),1)/12
    # Concatenate the plot:
    fluo = np.concatenate((fluo, eval_d["groundtruth"][pred_nb]))
    stims = np.concatenate((features[-36:,-1], future_stims))

    # Plot:
    color = [float(i)/255 for i in [15, 104, 245]]
    dcc.utilities.OptoPlotBackground(stims, x=x, ymin = 0, ymax = 4095,)
    plt.plot((0,0),(0, 4095), linestyle=":", color="gray")
    plt.plot(x, fluo*4095, color = "k")
    plt.plot(x_future, eval_d["prediction"][pred_nb]*4095, color = color)
    plt.xlim((x[0],x[-1]))
    plt.ylim((0,4095))
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlabel("Time (hours)")

#%% Panel D - RMSE distro

plt.figure()

# Log-spaced bins:
nbins = 100
bins = np.logspace(0, 4, nbins + 1, base=10)
# Plot hist:
n, _, _ = plt.hist(rmse, bins=bins)
# Plot 25th, median, and 75th %ile:
qs = np.quantile(rmse, [.25, .5, .75])
for q in qs:
    plt.plot([q, q], [0, 4500], color="gray", zorder=-1)

plt.xscale("log")
plt.xlim([5, 5000])
plt.ylim([0, 4500])
plt.xlabel("Root mean square error (a.u.)")
plt.ylabel("Count")
plt.savefig(save_folder+"Panel_E_RMSEdistro.png", dpi=300)
plt.savefig(save_folder+"Panel_E_RMSEdistro.svg", dpi=300)
plt.show()

#%% Panel E - 25th percentile

plt.figure()
prediction_plot(eval_d, rmse_order[25_007])
plt.savefig(save_folder+"Panel_E_25thperc.png", dpi=300)
plt.savefig(save_folder+"Panel_E_25thperc.svg", dpi=300)
plt.show()

#%% Panel E - 50th percentile

plt.figure()
prediction_plot(eval_d, rmse_order[50_012])
plt.savefig(save_folder+"Panel_E_median.png", dpi=300)
plt.savefig(save_folder+"Panel_E_median.svg", dpi=300)
plt.show()

#%% Panel E - 75th percentile

prediction_plot(eval_d, rmse_order[75_002])
plt.savefig(save_folder+"Panel_E_75thperc.png", dpi=300)
plt.savefig(save_folder+"Panel_E_75thperc.svg", dpi=300)
plt.show()

#%% SI Fig 1 - 95th percentile

plt.figure(figsize=(8,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    prediction_plot(eval_d, rmse_order[95_000+i])
plt.savefig(save_folder+"SI_Fig_95th_perc.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_95th_perc.svg", dpi=300)
plt.show()

#%% Panel F - Amount of past timeseries data

params, dataset, eval_set, model = load_model(model_folder_dict[24])
model.save(save_folder+"/model24.hdf5")

plt.figure()

rmse_all = []
for past_steps in range(1,1+4*12):
    
    print(past_steps)
    model = tf.keras.models.load_model(save_folder+"/model24.hdf5")
    
    eval_set.past_steps = past_steps
    metrics, eval_d = dcc.timeseries.evaluate(
        eval_set, model, batch_size=100_000, num_batches = 1, return_eval=True
        )
    rmse = np.sqrt(np.mean((eval_d["prediction"]-eval_d["groundtruth"])**2))
    rmse_all.append(rmse)

rmse_all = np.array(rmse_all)
np.save(save_folder+"past_steps_rmse.npy", rmse_all)

color = [float(i)/255 for i in [15, 104, 245]]
x = np.arange(1,1+4*12,1)/12
plt.plot(x, rmse_all*4095, linewidth=3, color = color)
plt.grid(which="both", axis="y")
plt.xlabel("past (hours)")
plt.ylabel("Root mean square error (a.u.)")
plt.savefig(save_folder+"Panel_J.png", dpi=300)
plt.savefig(save_folder+"Panel_J.svg", dpi=300)
plt.show()

#%% Panel G - RMSE per horizon

colors = {
    12: [116, 167, 247],
    24: [15, 104, 245],
    36: [2, 67, 171],
    48: [1, 33, 84]
    }

plt.figure()

for horizon in model_folder_dict:
    
    with open(save_folder+f"eval_d_horizon{horizon}.pkl","rb") as f:
        eval_d = pickle.load(f)

    # RMSE over samples:
    rmse = np.sqrt(
        np.mean(np.square((eval_d["prediction"]-eval_d["groundtruth"])*4095),axis=0)
        )
    # rmse = np.mean(np.abs((eval_d["prediction"]-eval_d["groundtruth"])*4095),axis=0)
    
    x = np.arange(0,len(rmse),1)/12
    
    color = [float(i)/255 for i in colors[horizon]]
    label = f"{int(horizon/12)}-hour"
    plt.plot(x, rmse, zorder=50-horizon, color = color, linewidth=3, label=label)

# plt.grid(which="both", axis="both")
# plt.yscale("log")
plt.xlabel("Horizon (hours)")
plt.ylabel("Root mean square error (a.u.)")
plt.legend(title="Horizon")
plt.savefig(save_folder+"Panel_I.png", dpi=300)
plt.savefig(save_folder+"Panel_I.svg", dpi=300)
plt.show()

#%% SI Fig 2 - Other horizons evaluations & RMSEs

horizon_list = [12, 36, 48]

plt.figure(figsize=(8,12), dpi=300)
for h, horizon in enumerate(horizon_list):
    
    with open(save_folder+f"eval_d_horizon{horizon}.pkl","rb") as f:
        eval_d = pickle.load(f)
    
    # RMSE over prediction horizon:
    rmse = np.sqrt(
        np.mean((4095*(eval_d["prediction"]-eval_d["groundtruth"]))**2,axis=1)
        )
    rmse_order = np.argsort(rmse)
    
    # RMSE distro
    plt.subplot(4,3,h+1)
    # Log-spaced bins:
    nbins = 100
    bins = np.logspace(0, 4, nbins + 1, base=10)
    # Plot hist:
    n, _, _ = plt.hist(rmse, bins=bins)
    # Plot 25th, median, and 75th %ile:
    qs = np.quantile(rmse, [.25, .5, .75])
    for q in qs:
        plt.plot([q, q], [0, 5000], color="gray", zorder=-1)
    plt.xscale("log")
    plt.xlim([5, 5000])
    plt.ylim([0, 5000])
    plt.xlabel("Root mean square error (a.u.)")
    plt.ylabel("Count")
    
    #25th percentile
    plt.subplot(4,3,3+h+1)
    prediction_plot(eval_d, rmse_order[25_000])

    #50th percentile
    plt.subplot(4,3,6+h+1)
    prediction_plot(eval_d, rmse_order[50_000])

    #75th percentile
    plt.subplot(4,3,9+h+1)
    prediction_plot(eval_d, rmse_order[75_000])


plt.savefig(save_folder+"SI_Fig_X_OtherHorizonsEval.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_X_OtherHorizonsEval.svg", dpi=300)
plt.show()

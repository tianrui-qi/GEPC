# -*- coding: utf-8 -*-
"""
This script generates plots for Figure 4, SI Fig 14, and Movie S3

Created on Fri Sep  2 14:03:54 2022

@author: jeanbaptiste
"""
import json
import pickle
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import deepcellcontrol as dcc

# Experiments (zenodo archive)
experiments_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/experiments/"

# Save images to:
save_folder = "D:/papers/deepmpc/figure4/"

# experiments path:
movie_folders = (
    experiments_folder + "2022-05-28_DeepMPC_2001_1",
    experiments_folder + "2022-06-01_DeepMPC_2001_2",
    experiments_folder + "2022-06-04_DeepMPC_2001_3"
    )

um_ppixel = 120/(1851-57) # measured distance per pixel

#%% Load and reconstruct 2001 movie

movie_shape = (80,125)
cutoff = 32*12

# Load data:
objectives = np.load(movie_folders[0] + "/whole_movie_objectives.npy")
shuffling = np.load(movie_folders[0] + "/whole_movie_shuffling.npy")
deshuffling = np.load(movie_folders[0] + "/whole_movie_deshuffling.npy")

# Place data into movie:
whole_movie = np.zeros(
    shape=(movie_shape[0]*movie_shape[1],cutoff), dtype = np.float32
    )
obj_movie = whole_movie.copy()
inputs_movie = whole_movie.copy()
pixels_to_xp = np.zeros(shape=(movie_shape[0]*movie_shape[1],2), dtype = int)
for xp_ind, xpf in enumerate(movie_folders):
    
    # Load local objectives, shuffling / deshuffling:
    local_obj = np.load(xpf + "/local_objectives.npy")
    local_shuffle = np.load(xpf + "/local_shuffling.npy")
    local_deshuffle = np.load(xpf + "/local_deshuffling.npy")
    obj_movie[local_shuffle] = local_obj[:,:cutoff]
    
    # Cells fluorescence:
    cells_fluo = np.load(xpf + "/cells_fluo.npy")
    whole_movie[local_shuffle] = cells_fluo
    
    # Optogenetic stimulations:
    cells_stims = np.load(xpf + "/cells_stims.npy")
    inputs_movie[local_shuffle] = np.clip(cells_stims,0,1)
    
    # Record pixel <-> xp correspondance
    pixels_to_xp[local_shuffle,0] = xp_ind
    pixels_to_xp[local_shuffle,1] = np.arange(local_obj.shape[0])

# De-shuffle movie:
whole_movie = np.reshape(whole_movie,movie_shape+(cutoff,))
obj_movie = np.reshape(obj_movie,movie_shape+(cutoff,))
inputs_movie = np.reshape(inputs_movie,movie_shape+(cutoff,))
pixels_to_xp = np.reshape(pixels_to_xp,movie_shape+(2,))

#%% Panel A - Obj & Fluo movie kymogrpahs

interval = 24

obj_kymograph = []
cells_kymograph = []
frame_nbs = np.arange(24-1, cutoff, interval)
for f in frame_nbs:
    obj_kymograph.append(
        dcc.utilities.color_img(obj_movie[:,:,f], vmin=.05, cmap=dcc.utilities.graymap)
        )
    cells_kymograph.append(
        dcc.utilities.color_img(whole_movie[:,:,f], vmin=.05, cmap=dcc.utilities.gfpmap)
        )

obj_kymograph = np.concatenate(obj_kymograph, axis=0)
cells_kymograph = np.concatenate(cells_kymograph, axis=0)
 
obj_kymograph = (obj_kymograph*255).astype(np.uint8)
cells_kymograph = (cells_kymograph*255).astype(np.uint8)

# Plot objectives kymograph
plt.figure()
plt.imshow(obj_kymograph)
plt.axis("off")
plt.savefig(save_folder+"Panel_A_obj.png", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_A_obj.svg", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_A_obj.pdf", dpi=300, bbox_inches='tight')
cv2.imwrite(save_folder+"Panel_A_obj.tif", obj_kymograph[:,:,::-1])
plt.show()

# Plot cells kymograph
plt.figure()
plt.imshow(cells_kymograph)
plt.axis("off")
plt.savefig(save_folder+"Panel_A_cells.png", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_A_cells.svg", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_A_cells.pdf", dpi=300, bbox_inches='tight')
cv2.imwrite(save_folder+"Panel_A_cells.tif", cells_kymograph[:,:,::-1])
plt.show()

#%% Reload cells data, keep all cells data:

objectives = []
fluorescence = []
area = []
stims = []
for xp_ind, xpf in enumerate(movie_folders):
    
    # Load local objectives:
    objectives.append(np.load(xpf + "/local_objectives.npy")[:,:cutoff])
    
    # Cells fluorescence:
    fluorescence.append(np.load(xpf + "/cells_fluo.npy")[:,:cutoff])
    
    # Stims:
    stims.append(np.load(xpf + "/cells_stims.npy")[:,:cutoff])
    
    # Load cells area:
    cells_area = []
    with open(xpf+"/fallback_control_parameters.json", "r") as f:
        control_parameters = json.load(f)
    with open(xpf+"/mothers.pkl", "rb") as f:
        mothers = pickle.load(f)
    for s in mothers:
        for p in s:
            for c in p:
                cells_area.append(
                    c[:cutoff,control_parameters["features"].index("area")]
                    )
    cells_area = np.array(cells_area)
    cells_area = -3_000 * np.log10(1-cells_area) # De-normalize
    area.append(cells_area)

objectives = np.concatenate(objectives, axis=0)
fluorescence = np.concatenate(fluorescence, axis=0)
area = np.concatenate(area, axis=0)
stims = np.concatenate(stims, axis=0)

fluorescence[fluorescence<100] = np.nan
area[area<100] = np.nan

def compute_growth(fluo, area):
    
    # Run through time:
    growth = []
    for t in range(len(area)-1):
    
        cont_flag = False
    
        # If current area is None or too small, append a NaN (empty chamber)
        if area[t] is None or area[t] < 100 or fluo[t] is None or fluo[t] < 100:
            fluo[t] = np.nan
            growth.append(np.nan)
            continue
        
        # Same thing for time t+1:
        if area[t+1] is None or area[t+1] < 100 or fluo[t+1] is None or fluo[t+1] < 100:
            fluo[t+1] = np.nan
            growth.append(np.nan)
            continue
        
        # Otherwise compute growth as delta_area / area:
        growth.append((area[t+1] - area[t])/area[t])
    
    # Filter out divisions and glitches:
    growth = np.array(growth)
    growth[growth<-.2] = np.nan
    growth[growth>.3] = np.nan
    
    # Convert to 1/hour:
    growth*=12
    
    return growth, fluo, area

growth = np.zeros_like(fluorescence)[:,:-1]
for c in range(fluorescence.shape[0]):
    _g, _f, _a = compute_growth(
        fluorescence[c], area[c]
        )
    growth[c] = _g
    fluorescence[c] = _f
    area[c] = _a

# Smoothed growth:
window = (6, 6)
avg_growth = np.zeros_like(growth)
for f in range(cutoff-1):
        
        f_min = f-window[0]
        if f_min < 0: f_min = 0
        f_max = f+window[1]
        if f_max > cutoff: f_max = cutoff
        
        t_growth = np.nanmean(growth[:,f_min:f_max], axis=1)
        avg_growth[:,f] = t_growth


#%% Panel B - Error over time

x = np.arange(384,)/12

plt.subplot(2,1,1)
av_obj = np.mean(obj_movie,axis=(0,1))
av_obj[:36] = np.nan
plt.plot(x, av_obj, "k")
abs_error = np.abs((whole_movie-obj_movie)).reshape(-1,384)
abs_error[:,:36] = np.nan
dcc.utilities.plotq(abs_error, color="purple")
rmse = np.sqrt(np.mean((abs_error)**2,axis=0))
rmse[:36] = np.nan
plt.plot(x,rmse)
mae = np.mean(abs_error,axis=0)
mae[:36] = np.nan
plt.plot(x,mae)
plt.xlim([0,383/12])
plt.xticks(list(range(0,34,2)))

plt.subplot(2,1,2)
plt.fill_between([0,383/12], [1,1], facecolor=[0,1,0])
av_inputs = np.mean(inputs_movie,axis=(0,1))
plt.fill_between(x, 1-av_inputs, facecolor=[1,0,0], edgecolor=None)
plt.xlim([0,383/12])
plt.ylim([0,1])
plt.xticks(list(range(0,34,2)))

plt.savefig(save_folder+f"Panel_B_imecourses.svg", dpi=300)


#%% Panel C - Error vs derivative of objective


x = np.diff(obj_movie, axis=2)[:,:,72:].flatten()
y = abs_error[:,73:].flatten()
nbins = 10
edges = np.concatenate(
    (
         -np.flip(np.logspace(np.log10(5), np.log10(500), nbins)),
         np.logspace(np.log10(5), np.log10(500), nbins),
         )
    )

# Compute error per diff bin:
error_dist = []
rmse_dist = []
mean_dist = []
emin, emax = -np.inf, edges[0]
mask = np.logical_and(x>=emin, x<emax)
yy = y[mask]
error_dist.append(np.quantile(yy, q=[.25,.5,.75]))
mean_dist.append(np.mean(yy))
rmse_dist.append(np.sqrt(np.mean(yy**2)))
for e in range(len(edges)-1):
    emin, emax = edges[e:e+2]
    mask = np.logical_and(x>=emin, x<emax)
    yy = y[mask]
    
    error_dist.append(np.quantile(yy, q=[.25,.5,.75]))
    mean_dist.append(np.mean(yy))
    rmse_dist.append(np.sqrt(np.mean(yy**2)))
emin, emax = edges[-1], np.inf
mask = np.logical_and(x>=emin, x<emax)
yy = y[mask]
error_dist.append(np.quantile(yy, q=[.25,.5,.75]))
mean_dist.append(np.mean(yy))
rmse_dist.append(np.sqrt(np.mean(yy**2)))

# Plot:
xplot = []
xplot.append(-3)
for e in range(len(edges)-1):
    emin, emax = edges[e:e+2]
    if emin < 0 and emax < 0:
        xpos = (emin-emax)/2 + emax
    elif emin < 0:
        xplot.append(0)
        continue
    else:
        xpos = (emax-emin)/2 + emin
    xplot.append(np.sign(xpos)*np.log10(np.abs(xpos)))
xplot.append(3)
plt.plot(xplot,[i[1] for i in error_dist], color="purple")
plt.fill_between(
    xplot,
    [i[0] for i in error_dist],
    [i[2] for i in error_dist],
    color="purple",
    alpha=.2,
    )
plt.plot(xplot, mean_dist, color="orange")
plt.plot(xplot, rmse_dist, color="blue")
plt.xticks(
    np.concatenate(
        (
            [-3],
            -np.flip(np.log10(np.arange(100,1000,100))),
            -np.flip(np.log10(np.arange(10,100,10))),
            -np.flip(np.log10(np.arange(5,10,1))),
            np.log10(np.arange(5,10,1)),
            np.log10(np.arange(10,100,10)),
            np.log10(np.arange(100,1000,100)),
            [3]
            )
        ),
    ["-1000"] + [""]*8 + ["-100"] + [""]*8 + ["-10"] + [""]*10 + ["10"] + [""]*8 + ["100"] + [""]*8 + ["1000"]
    )
# plt.xticks(np.arange(0,1,.25))
# plt.xlim([0,1])
plt.savefig(save_folder+f"Panel_C_diff.svg", dpi=300)
plt.show()

#%% Panel D - Filamentation over time:

plt.figure()
x = np.arange(0, area.shape[1], 1)/12
plt.plot(x, np.nanmean(area*um_ppixel**2>6, axis=0))
plt.xlim([0,x[-1]])
plt.ylabel(r"fraction of cells > 6 um^2")
yl = plt.ylim()
plt.ylim([0, yl[1]])
plt.xlabel("time (hours)")
plt.savefig(save_folder+"Panel_D_areafraction.png", dpi=300)
plt.savefig(save_folder+"Panel_D_areafraction.svg", dpi=300)
plt.savefig(save_folder+"Panel_D_areafraction.pdf", dpi=300)
plt.show()

#%% Panel D - Growth rate over time

plt.figure()
x = np.arange(0, growth.shape[1], 1)/12
plt.plot(x, np.nanmedian(growth, axis=0))
plt.xlim([0,x[-1]])
plt.ylim([-.1, 1.5])
plt.ylabel("growth rate (1/hour)")
plt.xlabel("time (hours)")
plt.savefig(save_folder+"Panel_D_growthtime.png", dpi=300)
plt.savefig(save_folder+"Panel_D_growthtime.svg", dpi=300)
plt.savefig(save_folder+"Panel_D_growthtime.pdf", dpi=300)
plt.show()

#%% Panel E - Total growth hist:

plt.figure()
plt.hist(avg_growth.flatten(), bins=200)

# Plot quantiles
quants = np.nanquantile(avg_growth.flatten(), np.linspace(.1,.9,9))
yl = plt.ylim()
for q_bot in quants:
    plt.plot([q_bot, q_bot], yl, color="gray", zorder=-1, alpha=.2)

plt.ylim(yl)
plt.xlim([-1, 3])
plt.xlabel("growth rate (1/hour)")
plt.ylabel("count")
plt.savefig(save_folder+"Panel_E_growthdistro.png", dpi=300)
plt.savefig(save_folder+"Panel_E_growthdistro.svg", dpi=300)
plt.savefig(save_folder+"Panel_E_growthdistro.pdf", dpi=300)
plt.show()

#%% Panel F - RMSE per quantile

def quantile_rmse(x, sq_error, q_num=100):


    quants = np.nanquantile(x.flatten(), np.linspace(0,1,q_num+1))
    # To make sure the max point is included:
    quants[-1] +=.1

    rmse = []
    for q in range(q_num):
        
        q_bot = quants[q]
        q_top = quants[q+1]
        
        points = np.logical_and(x>=q_bot, x<q_top)
        # Ignore warm up phase:
        points[:,:36] = False
        
        rmse.append(np.sqrt(np.nanmean(sq_error[points])))
        
    return rmse

plt.figure()

sq_error = (objectives-fluorescence)**2
sq_error = sq_error[:,:-1]
sq_error[objectives[:,:-1]>800] = np.nan
rmse = quantile_rmse(avg_growth, sq_error)
plt.stairs(rmse, baseline=None, color="r", label="objectives < 800 a.u.")

sq_error = (objectives-fluorescence)**2
sq_error = sq_error[:,:-1]
sq_error[objectives[:,:-1]<1600] = np.nan
rmse = quantile_rmse(avg_growth, sq_error)
plt.stairs(rmse, baseline=None, color="g", label="objectives > 1600 a.u.")

sq_error = (objectives-fluorescence)**2
sq_error = sq_error[:,:-1]
rmse = quantile_rmse(avg_growth, sq_error)
plt.stairs(rmse, baseline=None, color = "b", label="All objectives")

plt.xlim([0,100])
plt.xlabel("Growth rate percentiles")
plt.ylabel("Root mean square error (a.u.)")
plt.legend()
plt.savefig(save_folder+"Panel_F_errorvsgrowth.png", dpi=300)
plt.savefig(save_folder+"Panel_F_errorvsgrowth.svg", dpi=300)
plt.savefig(save_folder+"Panel_F_errorvsgrowth.pdf", dpi=300)
plt.show()

#%% SI Fig. 14 - Panels A & B - Error and inputs kymographs + Distributions

def color_hist(values, bins, colors):
    
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    hist /= max(hist)
    for h, v in enumerate(hist):
        plt.fill_between(
            edges[h:h+2], [v, v], facecolor=colors[h], edgecolor=None
            )

error_kymograph = []
inputs_kymograph = []
inputs_frame = np.zeros((80,125,3), dtype=np.uint8)
error_frame = np.zeros((80,125,3), dtype=np.uint8)
ermap = cm.get_cmap("magma")
error_comp = lambda I: (np.log10(np.abs(I))-np.log10(10))/(np.log10(1000)-np.log10(10))
ticks = list(range(10,100,10)) + list(range(100,1000,100)) + list(range(1000,4095,1000))
# error_comp = lambda I: (np.abs(I)-20)/(1000-20)
for f in frame_nbs:
    error = error_comp(whole_movie[:,:,f]-obj_movie[:,:,f])
    error = np.clip(error, 0, 1)
    error = ermap(error)[:,:,:3]
    error_kymograph.append(error.copy())
    
    inputs_frame[:] = 0
    inputs_frame[:,:,1][inputs_movie[:,:,f]>0.5] = 255
    inputs_frame[:,:,0][inputs_movie[:,:,f]<0.5] = 255
    inputs_kymograph.append(inputs_frame.copy())
    
    bins = np.logspace(np.log10(.1), np.log10(4095), 30, base=10)
    colors = [ermap(error_comp(b)) for b in bins]
    color_hist(np.abs(whole_movie[:,:,f]-obj_movie[:,:,f]).flatten(), bins, colors)
    plt.xscale("log")
    plt.xlim([1,4095])
    plt.ylim([0, 1.1])
    plt.xticks(ticks, [""]*len(ticks))
    plt.savefig(
        save_folder+f"SI_Fig_14_2001_dist{f:03d}.svg", 
        dpi=300,
        bbox_inches='tight'
        )
    plt.show()

error_kymograph[0][:] = 0
# Concatenate kymographs into strips
inputs_kymograph = np.concatenate(inputs_kymograph, axis=0)
error_kymograph = np.concatenate(error_kymograph, axis=0)
inputs_kymograph = (inputs_kymograph).astype(np.uint8)
error_kymograph = (error_kymograph*255).astype(np.uint8)
cv2.imwrite(save_folder+"SI_Fig_14_2001_kymograph_error.tif", error_kymograph[:,:,::-1])
cv2.imwrite(save_folder+"SI_Fig_14_2001_kymograph_inputs.tif", inputs_kymograph[:,:,::-1])

plt.imshow(error_kymograph, cmap=ermap)
ticks = list(range(10,100,10)) + list(range(100,1000,100)) + [995]
ticks = [255*error_comp(x) for x in ticks]
labels = ["10"] + [""]*8 + ["100"] + [""]*8 + ["1000"]
cbar = plt.colorbar(extend="both")
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)
plt.savefig(save_folder+f"SI_Fig_14_2001_colormap.svg", dpi=300)
plt.show()


#%% SI Fig. 14 - Panel C - Gaussian Kernel Density Error v Objective

import numpy as np
from scipy.stats import gaussian_kde

abs_error = np.abs((whole_movie-obj_movie)).reshape(-1,384)
abs_error[:,:36] = np.nan
x = obj_movie[:,:,72:].flatten()
y = abs_error[:,72:].flatten()

# Downsample otherwise it takes ages:
x = x[::10]
y = np.log10(y[::10])

k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,0:np.log10(4095):y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Plot
ticks = list(range(1,10,1)) + list(range(10,100,10)) + list(range(100,1000,100)) + list(range(1000,4095,1000))
ticks = [np.log10(x) for x in ticks]
labels = [""]*9 + ["10"] + [""]*8 + ["100"] + [""]*8 + ["1000"] + [""]*3
plt.contourf(xi, yi, zi.reshape(xi.shape))
plt.yticks(ticks, labels)
plt.ylim(1,np.log10(2000))
plt.colorbar()
plt.savefig(save_folder+f"SI_Fig_14_2001_KDEerror.svg", dpi=300)
plt.show()

unique, counts = np.unique(obj_movie[:,:,72:].flatten(), return_counts=True)
plt.fill_between(unique, counts)
plt.xlim([500,2000])
plt.ylim([0,90_000])
plt.savefig(save_folder+f"SI_Fig_14_2001_ObjDist.svg", dpi=300)
plt.show()

#%% SI Movie 3 - 2001: A Space Odyssey
# Note: this requires DeLTA to run (commit 8ceb015).

plt.style.use('dark_background')
compiled = []
for f in range(0,cutoff):
    
    fig = plt.gcf()
    fig.set_size_inches(5.5, 2.5)
    fig.set_dpi(300)
    
    obj = dcc.utilities.color_img(obj_movie[:,:,f], vmin=.05, cmap=dcc.utilities.graymap)
    fluo = dcc.utilities.color_img(whole_movie[:,:,f], vmin=.05, cmap=dcc.utilities.gfpmap)
    
    obj = cv2.resize(
        obj,
        dsize = tuple([x*5 for x in obj.shape[:2]][::-1]),
        interpolation=cv2.INTER_NEAREST
        )
    fluo = cv2.resize(
        fluo,
        dsize = tuple([x*5 for x in fluo.shape[:2]][::-1]),
        interpolation=cv2.INTER_NEAREST
        )
    
    plt.subplot(1,2,1)
    plt.imshow(obj)
    plt.axis('off')
    plt.title("Objectives", fontsize=8)
    
    plt.subplot(1,2,2)
    plt.imshow(fluo)
    plt.axis('off')
    plt.title("Fluorescence", fontsize=8)
    
    # Time counter:
    plt.text(
        0.13,
        0.20,
        f"{int(f/12):02d}h {int(f*5%60):02d}min", 
        fontsize=8,
        transform=plt.gcf().transFigure
        )
    
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    X = X[:,:,:3]
    X = np.copy(X)
    
    X = np.concatenate((X[110:622, 180:800], X[110:622, 890:1510]), axis=1)
    compiled.append(X)
    
    plt.clf()
    plt.imshow(X)
    plt.show()
    plt.clf()


sys.path.append("D:/delta")
import delta
delta.utilities.vidwrite(compiled, save_folder + "SI_movie_3_2001SpaceOdyssey.mp4")

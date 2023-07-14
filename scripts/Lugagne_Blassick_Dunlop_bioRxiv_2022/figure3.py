# -*- coding: utf-8 -*-
"""
This script generates plots for Figure 3, SI Figs 9-13, and Movies S1 and S2

Some of the plots can only be generated with access to raw microscopy image data
that is not on the zenodo archive.

Created on Tue Jul 12 16:16:25 2022

@author: jeanbaptiste
"""
import pickle
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

import deepcellcontrol as dcc

# Experiments (zenodo archive)
experiments_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/experiments/"

# Trained models (zenodo archive):
models_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/models/"

# Raw microscopy images (not on zenodo):
raw_data = "Z:/data/Microscope/jeanbaptiste/deepmpc/control/"

# Save images to:
save_folder = "D:/papers/deepmpc/figure3/"

def load_cells(mothers, partition):
    #Load cells belonging to one of the 12h, 24h, 36h, 48h horizion partitions
    
    cells = []
    global_nb = []
    pos_nb = 0
    for s in mothers:
        for p in s:
            if pos_nb in partition:
                cells += [np.array(p)]
                global_nb += [[[pos_nb for _ in range(len(p))], np.arange(len(p))]]
            pos_nb+=1
    cells = np.concatenate(cells, axis=0)
    global_nb = np.concatenate(global_nb, axis=1)
    return cells, global_nb

#%% Load Horizons test experiment

horizon_exp = experiments_folder+"2022-05-07_DeepMPC_horizons_tests/"
with open(horizon_exp+"mothers.pkl", "rb") as f:
    mothers = pickle.load(f)
    
with open(horizon_exp+"positions_partitioning.pkl", "rb") as f:
    ppt = pickle.load(f)

with open(horizon_exp+"controller_model_folders.pkl", "rb") as f:
    model_folders = pickle.load(f)
for k, v in model_folders.items():
    model_folders[k] = models_folder + os.path.basename(v)

#%% Panel A - DeepMPC illustration 3 strategies

# Parameters:
cutoff = 228
present = 14*12
cell_nb = 110
x = np.arange(0, cutoff, 1)/12
colors = {
    "weak": [float(i)/255 for i in [116, 167, 247]],
    "best": [float(i)/255 for i in [15, 104, 245]],
    "strong": [float(i)/255 for i in [2, 67, 171]],
    }

# Control objective:
objective = dcc.utilities.sine_objective(offset=1250)
objective = objective[:cutoff-36]

# Load cells corresponding to 2-hour horizon controller:
cells, global_nb = load_cells(mothers, ppt["horizon_24"])
cells = cells[:,:cutoff,:]

plt.figure()

# Plot stimulations:
dcc.utilities.OptoPlotBackground(
    cells[cell_nb, :present, -1],
    x = x[:present],
    ymin = 0,
    ymax = 4095,
    )

# Plot objective:
plt.plot(x[36:],objective,linestyle=":",color="#808080",label="Objective")

# Plot fluorescence:
plt.plot(x[:present],cells[cell_nb,:present,0]*4095,"k",label="Mother")

# Instanciate controller:
controller = dcc.control.SplitLSTMMPC(
    model_file = model_folders["horizon_24"] + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=24, iterations=25, particles=40
        )
    )

# Run feedback:
_  = controller.feedback(
    cells[[cell_nb], :present, :],
    [objective[present-36:present-36+24]/4095]
    )
best_strategy = controller.strategy[0]
best_prediction = controller.show_predict(
    cells[[cell_nb], :present, :], controller.strategy
    )
dcc.utilities.OptoPlotBackground(
    best_strategy, x = x[present:present+24], ymin = 1/3*4095, ymax = 2/3*4095
    )

# Random "stronger" strategy based on best:
stronger_strategy = np.random.uniform(0,1, size=24)<(1-np.mean(best_strategy)*.90)
stronger_prediction = controller.show_predict(
    cells[[cell_nb], :present, :], np.array(stronger_strategy)[np.newaxis]
    )
dcc.utilities.OptoPlotBackground(
    stronger_strategy, x = x[present:present+24], ymin = 2/3*4095, ymax = 4095
    )

# Random "weaker" strategy based on best:
weaker_strategy = np.random.uniform(0,1, size=24)<np.mean(best_strategy)*.5
weaker_prediction = controller.show_predict(
    cells[[cell_nb], :present, :], np.array(weaker_strategy)[np.newaxis]
    )
dcc.utilities.OptoPlotBackground(
    weaker_strategy, x = x[present:present+24], ymin = 0/3*4095, ymax = 1/3*4095
    )

# Plot predictions for each case:
plt.plot(x[present:present+24], stronger_prediction[0]*4095, color=colors["strong"])
plt.plot(x[present:present+24], best_prediction[0]*4095, color=colors["best"])
plt.plot(x[present:present+24], weaker_prediction[0]*4095, color=colors["weak"])

plt.ylabel("Fluorescence (a.u.)")
plt.xlabel("time (hours)")
plt.xlim([x[0], x[-1]])
plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
plt.ylim(0, 4095)

plt.savefig(save_folder+"Panel_A.png", dpi=300)
plt.savefig(save_folder+"Panel_A.svg", dpi=300)
plt.savefig(save_folder+"Panel_A.pdf", dpi=300)

plt.show()

#%% Prediction Timing
# To run with %%time cell magic

parallel = 83
latent = controller.encoder.predict(cells[:parallel, :16*12, :], verbose=1)

for _ in range(25):
    b = controller.model.predict(
        [
            np.repeat(latent[0],40,axis=0),
            np.repeat(latent[1],40,axis=0),
            np.ones((parallel*40,24),dtype=np.float32)
        ],
        verbose=1, batch_size=2000
        )

#%% SI Fig. 9 - bPSO evaluation
import os
import time

past = cells[:, :present, :]

rmse = {}
timing = {}
bitdiff = {}

oneshot_rmse = {}
oneshot_timing = {}
oneshot_bitdiff = {}


particles_nbs = [2, 5, 10, 20, 40, 60, 100]
iteration_nbs = [1, 2, 5, 10, 25, 50, 100]


for horizon in [12, 24, 36, 48]:
    
    rmse[horizon] = {}
    timing[horizon] = {}
    bitdiff[horizon] = {}
    
    # Redef objectives to current horizon:
    objectives = [objective[present-36:present-36+horizon]/4095]*cells.shape[0]
    objectives = np.array(objectives)
    
    # Instanciate reference controller:
    model_file = model_folders[f"horizon_{horizon}"] + '/model.hdf5'
    controller = dcc.control.SplitLSTMMPC(
        model_file = model_file,
        strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
            horizon=horizon, iterations=100, particles=200
            )
        )
    
    # Run reference control:
    _  = controller.feedback(past, objectives)
    best_strategies = controller.strategy
    best_predictions = controller.show_predict(past, best_strategies)
    
    # Run bPSO:
    for particles in particles_nbs:
        timing[horizon][particles] = []
        rmse[horizon][particles] = []
        bitdiff[horizon][particles] = []
        
        for iterations in iteration_nbs:
            
            print((horizon, iterations, particles))
            
            # Instanciate controller:
            controller = dcc.control.SplitLSTMMPC(
                model_file = model_file,
                strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
                    horizon=horizon, iterations=iterations, particles=particles
                    )
                )
    
            # Time and run feedback control for current settings:
            t0 = time.perf_counter()
            _  = controller.feedback(past, objectives)
            timing[horizon][particles].append(time.perf_counter()-t0)
            
            # Compute difference between reference control and current controller:
            strategies = controller.strategy
            bitdiff[horizon][particles].append(
                np.mean(np.logical_xor(strategies, best_strategies))
                )
            
            # Compute RMSE between controller prediction and objective:
            predictions = controller.show_predict(past, strategies)
            rmse[horizon][particles].append(
                np.mean(np.sqrt(np.mean((predictions-objectives)**2,axis=1)))
                )
        
    # Run one-shot optimizer:
    model_file = model_folders[f"horizon_{horizon}"] + '/model.hdf5'
    controller = dcc.control.SplitLSTMMPC(
        model_file = model_file,
        strategy_optimizer=dcc.control.OneShotOptimizer(
            particles=1000, horizon=horizon
        )
    )

    # Time and run feedback control for current settings:
    t0 = time.perf_counter()
    _ = controller.feedback(past, objectives)
    oneshot_timing[horizon] = time.perf_counter()-t0

    # Compute difference between reference control and current controller:
    strategies = controller.strategy
    oneshot_bitdiff[horizon] = np.mean(
        np.logical_xor(strategies, best_strategies)
    )

    # Compute RMSE between controller prediction and objective:
    predictions = controller.show_predict(past, strategies)
    oneshot_rmse[horizon] = np.mean(
        np.sqrt(np.mean((predictions-objectives)**2, axis=1))
    )

#%% Plot the bPSO results:
plt.figure(figsize=(8,12), dpi=300)
for h, horizon in enumerate([12, 24, 36, 48]):
    
    # RMSE:
    plt.subplot(4,3,1+h*3)
    for particles in particles_nbs:
        plt.plot(
            iteration_nbs,
            [r*4095 for r in rmse[horizon][particles]]
            )
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([2.8e2, 1e3])
    plt.grid("both","both")
    xl = plt.xlim()
    plt.plot(xl, [oneshot_rmse[horizon]*4095]*2,'k--')
    plt.xlim(xl)
    
    # Difference
    plt.subplot(4,3,2+h*3)
    for particles in particles_nbs:
        plt.plot(iteration_nbs, [b*horizon for b in bitdiff[horizon][particles]])
    plt.xscale("log")
    plt.ylim([0, 25])
    plt.grid("both","both")
    xl = plt.xlim()
    plt.plot(xl, [oneshot_bitdiff[horizon]*horizon]*2,'k--')
    plt.xlim(xl)
    
    # Comp time
    plt.subplot(4,3,3+h*3)
    for particles in particles_nbs:
        plt.plot(
            iteration_nbs,
            [1000*t/past.shape[0] for t in timing[horizon][particles]]
            )
    plt.xscale("log")
    plt.ylim([0, 80])
    plt.grid("both","both")
    xl = plt.xlim()
    plt.plot(xl, [oneshot_timing[horizon]*1000/past.shape[0]]*2,'k--')
    plt.xlim(xl)


plt.savefig(save_folder+"SI_Fig_9_bPSOeval.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_9_bPSOeval.svg", dpi=300)
plt.savefig(save_folder+"SI_Fig_9_bPSOeval.pdf", dpi=300)
plt.show()

#%% Panel B - Control Population

plt.figure()

plt.fill_between([0,3],[2500, 2500], color="#eeeeee", zorder=-10)

# Plot objective:
plt.plot(x[36:],objective,linestyle="--",color="#808080",label="Objective")

# Plot 25-75 percentile:
dcc.utilities.plotq(cells[:,:,0]*4095, color="g")

plt.ylabel("Fluorescence (a.u.)")
plt.xlabel("Time (hours)")
plt.xlim([x[0], x[-1]])
plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
plt.ylim(0, 2500)

plt.savefig(save_folder+"Panel_B_controlpop.png", dpi=300)
plt.savefig(save_folder+"Panel_B_controlpop.svg", dpi=300)
plt.savefig(save_folder+"Panel_B_controlpop.pdf", dpi=300)
plt.show()

#%% Panel C - Single-cell trajectories

cell_colors = ("#ff9955", "#55ddff", "#aa87de")
obj_array = np.repeat(objective[np.newaxis],cells.shape[0], axis=0)
cells_rmse = np.sqrt(np.mean(np.square(cells[:,36:,0]*4095-obj_array),axis=1))
rmse_order = np.argsort(cells_rmse)

plt.figure()

# Single cell trajectories:
plt.subplot(3,1, (1,2))
plt.fill_between([0,3],[2500, 2500], color="#eeeeee", zorder=-10)
plt.plot(x[36:],objective,linestyle="--",color="#808080",label="Objective")
for c in range(3):
    cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
    plt.plot(
        x,
        cells[cell_nb,:,0]*4095,
        label=f"{25*(c+1)}%-ile", 
        color=cell_colors[c],
        lw= 2
        )

plt.ylabel("Fluorescence (a.u.)")
plt.xlim([x[0], x[-1]])
plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int), labels=[])
plt.ylim(0, 2500)
plt.legend()

# Optogenetic stimulations:
plt.subplot(3,1,3)
for c in range(3):
    cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
    dcc.utilities.OptoPlotBackground(
        cells[cell_nb,:,-1], x = x, ymin = c-.5, ymax = c+.5
        )

plt.ylabel("Stimulations")
plt.xlabel("Time (hours)")
plt.xlim([x[0], x[-1]])
plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
plt.ylim(-.5, 2.5)
plt.yticks([0,1,2],["25%", "50%", "75%"])

plt.savefig(save_folder+"Panel_C_controlsingle.png", dpi=300)
plt.savefig(save_folder+"Panel_C_controlsingle.svg", dpi=300)
plt.savefig(save_folder+"Panel_C_controlsingle.pdf", dpi=300)

plt.show()

#%% SI Movie 1 - Pre-load single cell fluo movies
# Note: this will not work if you do not have access to the raw images 
# Note: this requires DeLTA to run (commit 8ceb015).

sys.path.append("D:/delta")
import delta

# Necessary on some systems:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

cell_movies = []

ctrl_obj = dcc.utilities.sine_objective(offset=1250)

for c in range(3):

    cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
    
    raw_xpf = raw_data + os.path.basename(os.path.normpath(horizon_exp))
    
    with open(horizon_exp+"/roi_boxes.pkl","rb") as f:
        roi_boxes = pickle.load(f)
    seg_model = tf.keras.models.load_model(horizon_exp+"/delta_segmentation.hdf5")
    
    pos_nb, roi_nb = global_nb[:, cell_nb]
    pos = delta.pipeline.load_position(
        raw_xpf+f"/delta_positions/Pos{pos_nb:06d}.pkl"
        )
    
    # Load images:
    img_stack = []
    fluo_stack = []
    for f in range(0, cutoff):
        img_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        fluo_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan02_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        print(f)
    
    # Drift correction:
    img_stack, drift_values = delta.utilities.driftcorr(
        np.array(img_stack), template=pos.drifttemplate, box=pos.driftcorbox
        )
    fluo_stack, _ = delta.utilities.driftcorr(
        np.array(fluo_stack), drift=drift_values
        )
    
    # Segment images:
    seg_input = []
    for frame in img_stack:
        frame = delta.utils.cropbox(frame, pos.rois[roi_nb].box)
        frame = delta.utils.rangescale(frame, (0,1))
        frame = cv2.resize(frame, (32, 256))
        seg_input.append(frame[:,:,np.newaxis])
    seg_stack = seg_model.predict(np.array(seg_input), verbose=1)
    seg_stack = (seg_stack>.5).astype(np.uint8)
    seg_stack = seg_stack[:,:,:,0]
    for f, seg in enumerate(seg_stack):
        seg_stack[f,:,:] = delta.utilities.opencv_areafilt(seg, min_area = 100)
    
    # Get mother contour:
    mother_cnt = []
    dsize = (
        pos.rois[roi_nb].box["xbr"] - pos.rois[roi_nb].box["xtl"],
        pos.rois[roi_nb].box["ybr"] - pos.rois[roi_nb].box["ytl"]
        )
    for mask in seg_stack:
        mask = cv2.resize(mask,dsize)
        contours = delta.utilities.find_contours((mask>0.5).astype(np.uint8))
        mother_cnt += [sorted(contours,key=lambda cnt: min(cnt[:,:,1]))[0]]
    
    # Crop out fluo frames and draw mother contour:
    cellmovie = []
    for f, frame in enumerate(fluo_stack):
        chamber_img = delta.utilities.cropbox(frame, pos.rois[roi_nb].box)
        chamber_img = dcc.utilities.color_img(chamber_img, vmin=0.05)
        chamber_img = (chamber_img*255).astype(np.uint8)
        chamber_img = cv2.drawContours(
            chamber_img, [mother_cnt[f]], 0, [255,255,255], thickness=2
            )
        cellmovie.append(chamber_img)
    cell_movies.append(cellmovie)

#%% SI Movie 1 - Unroll strategies, create movie

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from matplotlib.lines import Line2D

# # Instanciate controller:
controller = dcc.control.SplitLSTMMPC(
    model_file = model_folders["horizon_24"] + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=24, iterations=25, particles=40
        )
    )

texts = ["25th percentile", "Median", "75th percentile"]
cells_imgs = [[210,640],[695,1115],[1175,1600]] # Paste positions

compiled = []
for f in range(36,cutoff):
    
    print(f"TIMEPOINT {f}")
    
    fig = plt.figure(figsize=(6,6), dpi=300)

    # Run through 3 cells:
    for c in range(3):
        
        plt.subplot(3,1,c+1)
        
        if f < 36:
            plt.fill_between([0,3],[4500,4500],color="gray", alpha=.1)
        
        # Plot objective:
        plt.plot(
            x[36:],objective,linestyle="--",color="#808080",label="Objective"
            )
        
        # What cell to plot? (based on RMSE distro percentiles)
        cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
        
        # Plot optogenetic stims background:
        dcc.utilities.OptoPlotBackground(
            cells[cell_nb,:f,-1], x = x[:f], ymax = 4095
            )
        
        # If control has started:
        if 36 <= f < 227:
            # Get best strategy & prediction for it from controller:
            _  = controller.feedback(
                cells[[cell_nb], :f, :],
                [ctrl_obj[f-36:f-36+24]/4095]
                )
            best_strategy = controller.strategy[0]
            best_prediction = controller.show_predict(
                cells[[cell_nb], :f, :], controller.strategy
                )
            best_strategy[0] = cells[cell_nb, f+1, -1] # bPSO doesn't always converge to the same thing
            
            # Plot strategy:
            dcc.utilities.OptoPlotBackground(
                best_strategy, x = np.arange(f, f+24)/12, ymax = 4095, alpha=.5
                )
            
        # Plot past/future divide:
        plt.plot([x[f]-.5/12, x[f]-.5/12], [0, 2500], color="gray", lw=2)
        plt.plot(x[:f], cells[cell_nb,:f,0]*4095, color=cell_colors[c], lw=3)
        
        # If control:
        if 36 <= f < 227:
            # Plot controller prediction:
            pred = [cells[cell_nb,f-1,0]*4095] + [x for x in best_prediction[0]*4095]
            predx = np.arange(f-1, f+24)/12
            predx[0] = (f-.5)/12
            plt.plot(predx, pred, color=cell_colors[c], lw=3, alpha=.7)
        
        # Misc:
        plt.plot(x[f]-.5/12, cells[cell_nb,f,0]*4095, color=cell_colors[c], lw=3, marker='.', markersize=10)
        plt.text(21, 1250, texts[c], ha="center", va="center", rotation=-90, size="large", color=cell_colors[c])
        
        # Labelling etc:
        if c == 1:
            plt.ylabel("Fluorescence (a.u.)")
        plt.xlim([x[0], x[-1]])
        if c==2:
            plt.xlabel("Time (hours)")
            plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
        else:
            plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int),labels=[])
        plt.ylim(0, 2500)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if c == 0:
            ax.annotate(
                '', xy=(0, 1.025), xycoords='axes fraction', xytext=(3/19, 1.025),
                arrowprops=dict(arrowstyle="-", color="gray", alpha=.3, lw=2)
                )
            plt.text(.05, 2650, "No control", color="gray", alpha=.6, size="medium")
    
    
    # Convert figure to numpy array image:
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    X = X[:,:,:3]
    X = np.copy(X)
    plt.clf()
    
    # Paste in the microscopy single cell movies:
    for c in range(3):
        xmax = 1720
        y1 = cells_imgs[c]
        cell_frame = cell_movies[c][f]
        dsize = (
            y1[1]-y1[0], 
            int(cell_frame.shape[1]*(y1[1]-y1[0])/cell_frame.shape[0])
            )
        cell_frame = cv2.resize(cell_frame, dsize = dsize[::-1])
        X[y1[0]:y1[1], xmax-cell_frame.shape[1]:xmax] = cell_frame
        
    # Padding & cropping:
    X = np.pad(X,((0,0),(0,50),(0,0)), constant_values=255)
    X = np.copy(X[125:1725])
    
    compiled.append(X)
    
    plt.imshow(X)
    plt.show()
    plt.clf()
    

# Scale down images (getting compression issues otherwise)
res = []
for X in compiled:
    res.append(cv2.resize(X,(1388,1200), interpolation=cv2.INTER_AREA))

# Write movie to disk:
import delta
delta.utilities.vidwrite(res, save_folder + "SI_movie_1_unroll_twittercut.mp4", crf=20)

#%% SI Fig. 10 - Other horizons control resuts plots

plt.figure(figsize=(18,12), dpi=300)

for h, horizon in enumerate([12, 36, 48]):
    
    cells, _ = load_cells(mothers, ppt[f"horizon_{horizon}"])
    cells = cells[:,:cutoff,:]
    
    plt.subplot(3,3,h+1)
    # Plot objective:
    plt.plot(x[36:],objective,color="#808080",label="Objective")
    
    # Plot 25-75 percentile and median:
    dcc.utilities.plotq(cells[:,:,0]*4095, color="g")
    
    plt.fill_between([0,3],[2500, 2500], color="#eeeeee", zorder=-10)
    
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlim([x[0], x[-1]])
    plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
    plt.ylim(0, 2500)
    plt.legend(("Objective", "Median", "25-75%-iles"))
    
    
    
    obj_array = np.repeat(objective[np.newaxis],cells.shape[0], axis=0)
    cells_rmse = np.sqrt(np.mean(
        np.square(cells[:,36:,0]*4095-obj_array),
        axis=1
        ))
    rmse_order = np.argsort(cells_rmse)
    
    plt.subplot(3,3,h+4)
    plt.fill_between([0,3],[2500, 2500], color="#eeeeee", zorder=-10)
    # Plot objective:
    plt.plot(x[36:],objective,color="#808080",label="Objective")
    for c in range(3):
        cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
        plt.plot(x, cells[cell_nb,:,0]*4095, label=f"{25*(c+1)}%-ile", color=cell_colors[c])
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlim([x[0], x[-1]])
    plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
    plt.ylim(0, 2500)
    plt.legend()
    
    # plt.subplot(6,1,5)
    plt.subplot(6,3,h+13)
    # Plot 3  cells:
    for c in range(3):
        cell_nb = rmse_order[int((c+1)*len(rmse_order)/4)]
        dcc.utilities.OptoPlotBackground(
            cells[cell_nb,:,-1], x = x, ymin = c, ymax = c+1
            )
    plt.ylabel("Percentile")
    plt.xlabel("Time (hours)")
    plt.xlim([x[0], x[-1]])
    plt.yticks(ticks=[.5, 1.5, 2.5], labels=["25th", "50th", "75th"])
    plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
    plt.ylim(0, 3)


plt.savefig(save_folder+"SI_Fig_10_OtherHorizonsControl.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_10_OtherHorizonsControl.svg", dpi=300)
plt.savefig(save_folder+"SI_Fig_10_OtherHorizonsControl.pdf", dpi=300)

plt.show()

#%% SI Fig. 11 - Horizons RMSE over time

cutoff = 228
x = np.arange(0, cutoff, 1)/12
objective = dcc.utilities.sine_objective(offset=1250)
objective = objective[:cutoff-36]

colors = {12: "#1f77b4", 24: "#ff7f0e", 36: "#2ca02c", 48: "#d62728"}

plt.figure()

for horizon in (12, 24, 36, 48):
    
    # Load cells corresponding to horizon:
    cells, _ = load_cells(mothers, ppt[f"horizon_{horizon}"])
    cells = cells[:,:cutoff,:]
    
    obj_array = np.repeat(objective[np.newaxis],cells.shape[0], axis=0)
    #     )
    time_rmse = np.sqrt(np.mean(
        np.square(cells[:,36:,0]*4095-obj_array),
        axis=0
        ))

    plt.plot(x[36:], time_rmse, label=f"{int(horizon/12)} hour", color=colors[horizon])

plt.ylabel("Root mean square error (a.u.)")
plt.xlabel("time (hours)")
plt.xlim([x[0], x[-1]])
plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
plt.grid(which="both", axis="y")
plt.legend(title="Horizon")

plt.savefig(save_folder+"SI_Fig_11_RMSEtime.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_11_RMSEtime.svg", dpi=300)
plt.savefig(save_folder+"SI_Fig_11_RMSEtime.pdf", dpi=300)

plt.show()

#%% SI Fig. 12 - Horizons RMSE distribution

plt.figure()

cells_rmse_tot = []
for horizon in (12, 24, 36, 48):
    
    # Load cells corresponding to horizon:
    cells, _ = load_cells(mothers, ppt[f"horizon_{horizon}"])
    cells = cells[:,:cutoff,:]
    
    # Compute RMSE per cell through time:
    obj_array = np.repeat(objective[np.newaxis],cells.shape[0], axis=0)
    cells_rmse = np.sqrt(np.mean(
        np.square(cells[:,36:,0]*4095-obj_array),
        axis=1
        ))
    log_rmse = np.log10(cells_rmse)
    
    # Base violin plot:
    parts = plt.violinplot(
        positions = [horizon/12],
        dataset = log_rmse,
        showextrema=False,
        widths=.8,
        points=50,
        )
    
    # Customize it:
    parts['bodies'][0].set_facecolor(colors[horizon])
    parts['bodies'][0].set_edgecolor('black')
    parts['bodies'][0].set_alpha(.9)
    parts['bodies'][0].zorder = 10
    plt.plot(
        [horizon/12, horizon/12],
        [min(log_rmse), max(log_rmse)],
        color='k',
        lw=1,
        zorder=11
        )
    
    # Plot quartiles, median, and mean:
    q1, median, q3 = np.percentile(log_rmse, [25, 50, 75])
    plt.plot([horizon/12, horizon/12], [q1, q3], color='#808080', lw=5, zorder=12)
    plt.plot([horizon/12], median, 'D', markersize=2.5, color='k', zorder= 13)
    plt.plot([horizon/12], np.mean(log_rmse), 'o', markersize=3, color='w', zorder= 14)

# Ticks, grid, axes etc
plt.xticks([1,2,3,4])
yticks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]
plt.yticks(ticks=np.log10(yticks), labels=yticks)
plt.grid(which="both", axis="y")
plt.xlabel("Horizon (hours)")
plt.ylabel("Root mean square error (a.u.)")

plt.savefig(save_folder+"SI_Fig_12_RMSEdistros.png", dpi=300)
plt.savefig(save_folder+"SI_Fig_12_RMSEdistros.svg", dpi=300)
plt.savefig(save_folder+"SI_Fig_12_RMSEdistros.pdf", dpi=300)

plt.show()

#%% Load and shape sinewaves movie

movie_shape = (100,100)
cutoff = 19*12
movie_folders = (
    experiments_folder + "2022-05-09_DeepMPC_sinemovie_1",
    experiments_folder + "2022-05-11_DeepMPC_sinemovie_2",
    experiments_folder + "2022-05-24_DeepMPC_sinemovie_3"
    )

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

#%% Panel D - Sinewaves kymograph

interval = 24 # 2 hours
cell_ind = ((15,15), (75, 50)) # Example cells
cell_colors = ("#ff9955", "#aa87de")

# Load kymograph images:
obj_kymograph = []
cells_kymograph = []
frame_nbs = np.arange(24, cutoff, interval)
for f in frame_nbs:
    obj_kymograph.append(
        dcc.utilities.color_img(obj_movie[:,:,f], vmin=.05, cmap=dcc.utilities.graymap)
        )
    cells_kymograph.append(
        dcc.utilities.color_img(whole_movie[:,:,f], vmin=.05, cmap=dcc.utilities.gfpmap)
        )

# Concatenate kymographs into strips
obj_kymograph = np.concatenate(obj_kymograph, axis=1)
cells_kymograph = np.concatenate(cells_kymograph, axis=1)
obj_kymograph = (obj_kymograph*255).astype(np.uint8)
cells_kymograph = (cells_kymograph*255).astype(np.uint8)

# Plot objectives kymograph
plt.figure()
plt.imshow(obj_kymograph)
for pix, color in zip(cell_ind, cell_colors):
    for shift in range(len(frame_nbs)):
        plt.plot(
            pix[1]+shift*100, # Numpy indexing is y x
            pix[0],
            '.',
            markersize=1,
            color=color
            )
plt.axis("off")
plt.savefig(save_folder+"Panel_E_obj.png", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_E_obj.svg", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_E_obj.pdf", dpi=300, bbox_inches='tight')
cv2.imwrite(save_folder+"Panel_E_obj.tif", obj_kymograph[:,:,::-1])
plt.show()

# Plot cells kymograph
plt.figure()
plt.imshow(cells_kymograph)
for pix, color in zip(cell_ind, cell_colors):
    for shift in range(len(frame_nbs)):
        plt.plot(
            pix[1]+shift*100,
            pix[0],
            '.',
            markersize=1,
            color=color
            )
plt.axis("off")
plt.savefig(save_folder+"Panel_E_cells.png", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_E_cells.svg", dpi=300, bbox_inches='tight')
plt.savefig(save_folder+"Panel_E_cells.pdf", dpi=300, bbox_inches='tight')
cv2.imwrite(save_folder+"Panel_E_cells.tif", cells_kymograph[:,:,::-1])
plt.show()

#%% Panel E&F - Single cell kymograph and trajectory plots
# Note: this panel will not work if you do not have access to the raw images 
# Note: this panel requires DeLTA to run (commit 8ceb015).

sys.path.append("D:/delta")
import delta

# Necessary on some systems:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


interval = 6
panels = ("E", "F")

for c, pix in enumerate(cell_ind):
    
    # XP & cell indexes:
    xp_ind = pixels_to_xp[pix[0], pix[1], 0]
    cell_nb = pixels_to_xp[pix[0], pix[1], 1]
    
    # Load relevant data:
    xpf = movie_folders[xp_ind]
    local_obj = np.load(xpf + "/local_objectives.npy")
    xp_cells_fluo = np.load(xpf + "/cells_fluo.npy")
    xp_cells_stims = np.load(xpf + "/cells_stims.npy")
    raw_xpf = raw_data + os.path.basename(os.path.normpath(xpf))
    
    with open(xpf+"/roi_boxes.pkl","rb") as f:
        roi_boxes = pickle.load(f)
    seg_model = tf.keras.models.load_model(xpf+"/delta_segmentation.hdf5")
    
    # Plot cell stims and fluorescence:
    plt.figure()
    x = np.arange(0, cutoff, dtype = np.float32) / 12
    dcc.utilities.OptoPlotBackground(
        xp_cells_stims[cell_nb],
        x = x,
        ymin = 0,
        ymax = 4095,
        )
    plt.plot(x,xp_cells_fluo[cell_nb],"k")
    plt.plot(
        x[36:],
        local_obj[cell_nb][36:cutoff],
        linestyle=":",
        color="#808080",
        label="Objective"
        )
    plt.ylabel("Fluorescence (a.u.)")
    plt.xlabel("time (hours)")
    plt.xlim([x[0], x[-1]])
    plt.xticks(ticks=np.arange(0,x[-1],2,dtype=int))
    plt.ylim(0, 4095)
    
    plt.savefig(save_folder+f"Panel_{panels[c]}_plot.png", dpi=300)
    plt.savefig(save_folder+f"Panel_{panels[c]}_plot.svg", dpi=300)
    plt.savefig(save_folder+f"Panel_{panels[c]}_plot.pdf", dpi=300)
    plt.show()
    
    
    ## Kymograph:
    # Figure out position and roi:
    _cell = 0
    total_pos = 0
    for s, series in enumerate(roi_boxes):
        for p, pos in enumerate(series):
            for r, _roi in enumerate(pos):
                if _cell == cell_nb:
                    pos_nb = total_pos
                    roi_nb = r
                _cell += 1
            total_pos+=1
    pos = delta.pipeline.load_position(
        raw_xpf+f"/delta_positions/Pos{pos_nb:06d}.pkl"
        )
    
    # Load images:
    img_stack = []
    fluo_stack = []
    for f in range(0, cutoff, interval):
        img_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        fluo_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan02_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        print(f)
    
    # Drift correction:
    img_stack, drift_values = delta.utilities.driftcorr(
        np.array(img_stack), template=pos.drifttemplate, box=pos.driftcorbox
        )
    fluo_stack, _ = delta.utilities.driftcorr(
        np.array(fluo_stack), drift=drift_values
        )
    
    # Segment images:
    seg_input = []
    for frame in img_stack:
        frame = delta.utils.cropbox(frame, pos.rois[roi_nb].box)
        frame = delta.utils.rangescale(frame, (0,1))
        frame = cv2.resize(frame, (32, 256))
        seg_input.append(frame[:,:,np.newaxis])
    seg_stack = seg_model.predict(np.array(seg_input), verbose=1)
    seg_stack = (seg_stack>.5).astype(np.uint8)
    seg_stack = seg_stack[:,:,:,0]
    for f, seg in enumerate(seg_stack):
        seg_stack[f,:,:] = delta.utilities.opencv_areafilt(seg, min_area = 100)
    
    # Get mother contour:
    mother_cnt = []
    dsize = (
        pos.rois[roi_nb].box["xbr"] - pos.rois[roi_nb].box["xtl"],
        pos.rois[roi_nb].box["ybr"] - pos.rois[roi_nb].box["ytl"]
        )
    for mask in seg_stack:
        mask = cv2.resize(mask,dsize)
        contours = delta.utilities.find_contours((mask>0.5).astype(np.uint8))
        mother_cnt += [sorted(contours,key=lambda cnt: min(cnt[:,:,1]))[0]]
    
    # Crop out fluo frames and draw mother contour:
    kymograph = []
    for f, frame in enumerate(fluo_stack):
        chamber_img = delta.utilities.cropbox(frame, pos.rois[roi_nb].box)
        chamber_img = dcc.utilities.color_img(chamber_img, vmin=0.05)
        chamber_img = (chamber_img*255).astype(np.uint8)
        chamber_img = cv2.drawContours(
            chamber_img, [mother_cnt[f]], 0, [255,255,255], thickness=2
            )
        kymograph.append(chamber_img)
    kymograph = np.concatenate(kymograph, axis=1)
    
    # Plot kymograph:
    plt.figure()
    plt.imshow(kymograph)
    plt.show()
    cv2.imwrite(
        save_folder+f"Panel_{panels[c]}_kymograph.tif", kymograph[:,:,::-1]
        )

#%% SI Fig. 13 - Panels A & B - Error and inputs kymographs + Distributions

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
inputs_frame = np.zeros((100,100,3), dtype=np.uint8)
error_frame = np.zeros((100,100,3), dtype=np.uint8)
ermap = cm.get_cmap("magma")
error_comp = lambda I: (np.log10(np.abs(I))-np.log10(10))/(np.log10(1000)-np.log10(10))
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
    plt.savefig(
        save_folder+f"SI_Fig_13_sinewaves_dist{f:03d}.svg", 
        dpi=300, 
        bbox_inches='tight'
        )
    plt.show()

error_kymograph[0][:] = 0
# Concatenate kymographs into strips
inputs_kymograph = np.concatenate(inputs_kymograph, axis=1)
error_kymograph = np.concatenate(error_kymograph, axis=1)
inputs_kymograph = (inputs_kymograph).astype(np.uint8)
error_kymograph = (error_kymograph*255).astype(np.uint8)
cv2.imwrite(save_folder+"SI_Fig_13_sinewaves_kymograph_error.tif", error_kymograph[:,:,::-1])
cv2.imwrite(save_folder+"SI_Fig_13_sinewaves_kymograph_inputs.tif", inputs_kymograph[:,:,::-1])


plt.imshow(error_kymograph, cmap=ermap)
ticks = list(range(10,100,10)) + list(range(100,1000,100)) + [995]
ticks = [255*error_comp(x) for x in ticks]
labels = ["10"] + [""]*8 + ["100"] + [""]*8 + ["1000"]
cbar = plt.colorbar(extend="both")
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)
plt.savefig(save_folder+f"SI_Fig_13_sinewaves_colormap.svg", dpi=300)
plt.show()

#%% Panel G - Error timecourses

x = np.arange(228,)/12

plt.subplot(2,1,1)
av_obj = np.mean(obj_movie,axis=(0,1))
av_obj[:36] = np.nan
plt.plot(x, av_obj, "k--")
abs_error = np.abs((whole_movie-obj_movie)).reshape(-1,228)
abs_error[:,:36] = np.nan
dcc.utilities.plotq(abs_error, color="purple")
rmse = np.sqrt(np.mean((abs_error)**2,axis=0))
rmse[:36] = np.nan
plt.plot(x,rmse)
mae = np.mean(abs_error,axis=0)
mae[:36] = np.nan
plt.plot(x,mae)
plt.xlim([0,227/12])
plt.xticks(list(range(0,20,2)))

plt.subplot(2,1,2)
plt.fill_between([0,227/12], [1,1], facecolor=[0,1,0])
av_inputs = np.mean(inputs_movie,axis=(0,1))
plt.fill_between(x, 1-av_inputs, facecolor=[1,0,0], edgecolor=None)
plt.xlim([0,227/12])
plt.ylim([0,1])
plt.xticks(list(range(0,20,2)))

plt.savefig(save_folder+f"Panel_G_sinewaves_timecourses.svg", dpi=300)

#%% SI Fig. 13 - Panel C - Gaussian Kernel Density Error v Objective

import numpy as np
from scipy.stats import gaussian_kde

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
plt.savefig(save_folder+f"SI_Fig_13_sinewaves_KDEerror.svg", dpi=300)

#%% SI Fig. 13 - Panel D - Error v Phase

# Compute phase
phase = np.zeros((100, 100, 228-36))
for i in range(100):
    for j in range(100):
        lower = next((k for k,x in enumerate(obj_movie[i,j,36:]) if x<=1250))
        cross  = next((k for k,x in enumerate(obj_movie[i,j,36+lower:]) if x>1250))
        delay = 1-(lower+cross)*5/(8*60)
        phase[i,j,:] = delay + np.arange(0,228-36)*5/(8*60)
phase = np.mod(phase,1)

x = phase[:,:,37:].flatten()
y = abs_error[:,73:].flatten()
edges = np.linspace(0, 1, 50)

# Compute error per phase bin:
error_dist = []
rmse_dist = []
mean_dist = []
for e in range(len(edges)-1):
    emin, emax = edges[e:e+2]
    mask = np.logical_and(x>=emin, x<emax)
    yy = y[mask]
    
    error_dist.append(np.quantile(yy, q=[.25,.5,.75]))
    mean_dist.append(np.mean(yy))
    rmse_dist.append(np.sqrt(np.mean(yy**2)))

# Plot:
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(np.linspace(0,1,100), 750*np.sin(np.linspace(0,1,100)*2*np.pi)+1250, "k", zorder = -100)
ax1.plot(edges[:-1],[i[1] for i in error_dist], color="purple")
ax1.fill_between(
    edges[:-1],
    [i[0] for i in error_dist],
    [i[2] for i in error_dist],
    color="purple",
    alpha=.2,
    )
ax1.plot(edges[:-1], mean_dist, color="orange")
ax1.plot(edges[:-1], rmse_dist, color="blue")
plt.xticks(np.arange(0,1,.25))
plt.xlim([0,1])
plt.savefig(save_folder+f"SI_Fig_13_sinewaves_phase.svg", dpi=300)
plt.show()

#%% SI Movie 2 - Pre-load cell movies
# Note: this will not work if you do not have access to the raw images 
# Note: this requires DeLTA to run (commit 8ceb015).

sys.path.append("D:/delta")
import delta

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


cell_ind = ((15,15), (75, 50))
fig_movie = []
plt.style.use('dark_background')

# Retrieve cell fluo, stims, objective, and movie:
cell_data = [{},{}]
for c, pix in enumerate(cell_ind):
    
    # XP & cell indexes:
    xp_ind = pixels_to_xp[pix[0], pix[1], 0]
    cell_nb = pixels_to_xp[pix[0], pix[1], 1]
    
    # Load relevant data:
    xpf = movie_folders[xp_ind]
    local_obj = np.load(xpf + "/local_objectives.npy")
    xp_cells_fluo = np.load(xpf + "/cells_fluo.npy")
    xp_cells_stims = np.load(xpf + "/cells_stims.npy")
    raw_xpf = raw_data + os.path.basename(os.path.normpath(xpf))
    
    cell_data[c]["stims"] = xp_cells_stims[cell_nb]
    cell_data[c]["fluo"] = xp_cells_fluo[cell_nb]
    cell_data[c]["objective"] = local_obj[cell_nb][36:cutoff]
    
    with open(xpf+"/roi_boxes.pkl","rb") as f:
        roi_boxes = pickle.load(f)
    seg_model = tf.keras.models.load_model(xpf+"/delta_segmentation.hdf5")
    
    # Figure out position and roi:
    _cell = 0
    total_pos = 0
    for s, series in enumerate(roi_boxes):
        for p, pos in enumerate(series):
            for r, _roi in enumerate(pos):
                if _cell == cell_nb:
                    pos_nb = total_pos
                    roi_nb = r
                _cell += 1
            total_pos+=1
    pos = delta.pipeline.load_position(
        xpf+f"/delta_positions/Pos{pos_nb:06d}.pkl"
        )
    
    # Load images:
    img_stack = []
    fluo_stack = []
    for f in range(0, cutoff):
        img_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        fluo_stack += [cv2.imread(
            raw_xpf+f"/pos{pos_nb+1:04d}/chan02_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        print(f)
    
    # Drift correction:
    img_stack, drift_values = delta.utilities.driftcorr(
        np.array(img_stack), template=pos.drifttemplate, box=pos.driftcorbox
        )
    fluo_stack, _ = delta.utilities.driftcorr(
        np.array(fluo_stack), drift=drift_values
        )
    
    # Segment images:
    seg_input = []
    for frame in img_stack:
        frame = delta.utils.cropbox(frame, pos.rois[roi_nb].box)
        frame = delta.utils.rangescale(frame, (0,1))
        frame = cv2.resize(frame, (32, 256))
        seg_input.append(frame[:,:,np.newaxis])
    seg_stack = seg_model.predict(np.array(seg_input), verbose=1)
    seg_stack = (seg_stack>.5).astype(np.uint8)
    seg_stack = seg_stack[:,:,:,0]
    for f, seg in enumerate(seg_stack):
        seg_stack[f,:,:] = delta.utilities.opencv_areafilt(seg, min_area = 100)
    
    # Get mother contour:
    mother_cnt = []
    dsize = (
        pos.rois[roi_nb].box["xbr"] - pos.rois[roi_nb].box["xtl"],
        pos.rois[roi_nb].box["ybr"] - pos.rois[roi_nb].box["ytl"]
        )
    for mask in seg_stack:
        mask = cv2.resize(mask,dsize)
        contours = delta.utilities.find_contours((mask>0.5).astype(np.uint8))
        mother_cnt += [sorted(contours,key=lambda cnt: min(cnt[:,:,1]))[0]]
    
    # Crop out fluo frames and draw mother contour:
    cellmovie = []
    for f, frame in enumerate(fluo_stack):
        chamber_img = delta.utilities.cropbox(frame, pos.rois[roi_nb].box)
        chamber_img = dcc.utilities.color_img(chamber_img, vmin=0.05)
        chamber_img = (chamber_img*255).astype(np.uint8)
        chamber_img = cv2.drawContours(
            chamber_img, [mother_cnt[f]], 0, [255,255,255], thickness=2
            )
        cellmovie.append(chamber_img)
    cell_data[c]["movie"] = cellmovie

#%% SI Movie 2 - Concentric sinewaves

from matplotlib.patches import ConnectionPatch


plt.style.use('dark_background')
compiled = []

for f in range(0, whole_movie.shape[2]):
    
    fig = plt.gcf()
    fig.set_size_inches(9, 2.5)
    fig.set_dpi(300)
    
    # Gray obj movie:
    conc_frame = dcc.utilities.color_img(
        obj_movie[:,:,f], vmin=.05, cmap=dcc.utilities.graymap
        )
    plt.subplot(1,3,1)
    plt.imshow(np.fliplr(conc_frame))
    plt.axis('off')
    plt.title("Objectives", fontsize=10)
    
    # Colored concentric movie:
    conc_frame = dcc.utilities.color_img(
        whole_movie[:,:,f], vmin=.05, cmap=dcc.utilities.gfpmap
        )
    mov_ax = plt.subplot(1,3,2)
    plt.imshow(np.fliplr(conc_frame))
    plt.axis('off')
    plt.title("Fluorescence", fontsize=10)
    
    # Time counter:
    plt.text(
        0.14,
        0.05,
        f"{int(f/12):02d}h {int(f*5%60):02d}min", 
        fontsize=10,
        transform=plt.gcf().transFigure
        )
    
    # Single-cell plots:
    for c, pix in enumerate(cell_ind):
        
        ax = plt.subplot(2,3,3*(c+1))
        # Opto stims:
        dcc.utilities.OptoPlotBackground(
            cell_data[c]["stims"][:f],
            x = np.arange(f)/12,
            ymin = -200,
            ymax = 100,
            )
        # Objective:
        plt.plot(
            np.arange(36,cutoff)/12,
            cell_data[c]["objective"][:cutoff-36],
            color="w",
            linewidth=1
            )
        # FLuo:
        plt.plot(np.arange(f)/12, cell_data[c]["fluo"][:f], color=[0,1,.4], lw=2)
        # Labelling etc:
        plt.ylim(-200,2400)
        plt.xlim(0,cutoff/12)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # White connecting lines 
        con = ConnectionPatch(
            xyA=(100-pix[1],pix[0]), xyB=(-0.1,.5), coordsA="data", coordsB="axes fraction",
            axesA=mov_ax, axesB=ax, color="w", linewidth=1
            )
        mov_ax.add_artist(con)
        
    # Figure to numpy array image:
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    X = X[:,:,:3]
    X = np.copy(X)
    plt.clf()
    
    # Paste in the cell microscopy movies
    xmax = 1800
    y1 = [60,345]
    cell_frame = cell_data[0]["movie"][f][:,:]
    dsize = (
        y1[1]-y1[0], 
        int(cell_frame.shape[1]*(y1[1]-y1[0])/cell_frame.shape[0])
        )
    cell_frame = cv2.resize(cell_frame, dsize = dsize[::-1])
    X[y1[0]:y1[1], xmax-cell_frame.shape[1]:xmax] = cell_frame
    
    y1 = [380,660]
    cell_frame = cell_data[1]["movie"][f][:,:]
    dsize = (
        y1[1]-y1[0], 
        int(cell_frame.shape[1]*(y1[1]-y1[0])/cell_frame.shape[0])
        )
    cell_frame = cv2.resize(cell_frame, dsize = dsize[::-1])
    X[y1[0]:y1[1], xmax-cell_frame.shape[1]:xmax] = cell_frame
    
    # Crop:
    X = np.concatenate((X[:, 300:960], X[:, 1080:2500]), axis=1)
    
    
    plt.imshow(X)
    plt.show()
    plt.clf()
    
    compiled.append(X)

# Write movie to disk:
compiled = np.array(compiled)
delta.utilities.vidwrite(compiled, save_folder + "SI_movie_2_concentric.mp4")

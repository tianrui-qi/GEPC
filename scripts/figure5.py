# -*- coding: utf-8 -*-
"""
This script generates plots for Figure 5, SI Fig. 8-10, and Movie S4

Created on Thu Aug 11 11:45:33 2022

@author: jeanbaptiste
"""
import json
import pickle
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Experiments (zenodo archive)
experiments_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/experiments/"

# Raw microscopy images (not on zenodo):
raw_data = "Z:/data/Microscope/jeanbaptiste/deepmpc/control/"

# Save images to:
save_folder = "D:/papers/deepmpc/figure5/"

# experiments path:
experiments = [
    "2022-07-26_DeepMPC_tetA_1/",
    "2022-08-01_DeepMPC_tetA_1/",
    "2022-08-02_DeepMPC_tetA_1/",
    ]

def compute_growth(fluo, area):
    
    # Run through time:
    growth = []
    for t in range(len(area)-1):
    
        cont_flag = False
    
        # If current area is None or too small, append a NaN (empty chamber)
        if area[t] is None or area[t] < 100 or fluo[t] is None or fluo[t] < 350:
            fluo[t] = np.nan
            growth.append(np.nan)
            continue
        
        # Same thing for time t+1:
        if area[t+1] is None or area[t+1] < 100 or fluo[t+1] is None or fluo[t+1] < 350:
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

#%% Load data:

antibio_time = 9*12
cutoff = antibio_time+60

xp_data = []
for xp, xpname in enumerate(experiments):
    
    print(xpname)
    xp_data.append({})
    
    # Cells fluo and objectives are already saved to numpy arrays::
    xp_data[xp]["fluo"] = np.load(
        experiments_folder+xpname+"cells_fluo.npy"
        )[:,:cutoff]
    xp_data[xp]["fluo"][xp_data[xp]["fluo"]==0] = np.nan
    xp_data[xp]["objectives"] = np.load(
        experiments_folder+xpname+"objectives.npy"
        )[:,:cutoff]
    
    # Cell area:
    with open(experiments_folder+xpname+"fallback_control_parameters.json", "r") as f:
        control_parameters = json.load(f)
    with open(experiments_folder+xpname+"mothers.pkl", "rb") as f:
        mothers = pickle.load(f)
    cells_area = []
    for s in mothers:
        for p in s:
            for c in p:
                cells_area.append(c[:cutoff,control_parameters["features"].index("area")])
    cells_area = np.array(cells_area)
    cells_area = -3_000 * np.log10(1-cells_area) # De-normalize
    xp_data[xp]["area"] = cells_area
    
    # Compute growth and update fluo and area:
    cellnb = xp_data[xp]["fluo"].shape[0]
    xp_data[xp]["growth"] = np.zeros_like(
        xp_data[xp]["fluo"], shape=(cellnb, cutoff-1)
        )
    for c in range(cellnb):
        _g, _f, _a = compute_growth(
            xp_data[xp]["fluo"][c], xp_data[xp]["area"][c]
            )
        xp_data[xp]["growth"][c] = _g
        xp_data[xp]["fluo"][c] = _f
        xp_data[xp]["area"][c] = _a


# Smoothed growth:
window = (6, 6)
for xp in xp_data:
    
    xp["avg_growth"] = np.zeros_like(xp["growth"])
    
    for f in range(cutoff-1):
        
        f_min = f-window[0]
        if f_min < 0: f_min = 0
        f_max = f+window[1]
        if f_max > cutoff: f_max = cutoff
        
        t_growth = np.nanmean(xp["growth"][:,f_min:f_max], axis=1)
        xp["avg_growth"][:,f] = t_growth

#%% Panel B - Control results:

plt.figure()
obj_colors = ["#ff6961", "#ffb347", "#fff056", "#bfef66", "#79dcb9"]
obj_levels = [1, 800, 1200, 1800, 4095]

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    x = np.arange(0,cutoff,1)/12
    
    plt.plot([3, cutoff/12], [obj_lvl, obj_lvl], "--", color="grey", zorder = 90)
    plt.plot([antibio_time/12, antibio_time/12],[0,4095],"--", color="grey", zorder = 90)
    
    all_fluo = []
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        fluo = xp["fluo"][obj_cells]
        all_fluo.append(fluo)
        
        plt.plot(x,np.nanmedian(fluo,axis=0), color=color, zorder = 100)
        
    all_fluo = np.concatenate(all_fluo, axis=0)

plt.xlabel("time (hours)")
plt.ylabel("Fluorescence (a.u.)")
plt.ylim([0,4095])
plt.savefig(save_folder+"/Panel_B_control.png", dpi=300)
plt.savefig(save_folder+"/Panel_B_control.svg", dpi=300)
plt.savefig(save_folder+"/Panel_B_control.pdf", dpi=300)

plt.show()

#%% Panel C - growth rates over time

plt.figure()
plt.plot([antibio_time/12, antibio_time/12],[0,1.4],"--", color="grey")

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    for xp in xp_data:
    
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        growth = np.nanmedian(xp["growth"][obj_cells], axis=0)
        growth = savgol_filter(growth, 15, 2)
        x = np.arange(0,len(growth),1)/12
        plt.plot(x, growth, color=color)

plt.xlabel("time (hours)")
plt.ylabel("Growth rate (1/hour)")

plt.savefig(save_folder+"/Panel_C_growth.png", dpi=300)
plt.savefig(save_folder+"/Panel_C_growth.svg", dpi=300)
plt.savefig(save_folder+"/Panel_C_growth.pdf", dpi=300)
plt.show()

#%% Panel D - growth rate distributions over time

timepoints = [12, 102, 114, 126, 138, 150, 162]
growth_threshold = .3

plt.figure(figsize=(6, 3.5), dpi=300)

for obj_lvl, color in zip(obj_levels[::-1], obj_colors[::-1]):
    
    pos=0
    tick_pos = []
    tick_labels = []
    
    for f in timepoints:
        pos+=1
    
        # Compile growth for objective level throughout replicate XPs:
        obj_growth = []
        for xp in xp_data:
            obj_cells = xp["objectives"][:,-1] == obj_lvl
            growth = xp["avg_growth"][obj_cells][:,f]
            growth = growth[~np.isnan(growth)]
            obj_growth.append(growth)
        obj_growth = np.concatenate(obj_growth)
        
        # Filter out top/bottom 0.1%:
        qs = np.quantile(obj_growth,[.001, .999])
        obj_growth = obj_growth[obj_growth>qs[0]]
        obj_growth = obj_growth[obj_growth<qs[1]]
        
        # Violin plot for timepoint:
        parts = plt.violinplot(
            positions = [pos],
            dataset = obj_growth,
            showextrema=False,
            widths=.7,
            points=200,
            )
        parts['bodies'][0].set_alpha(1)
        parts['bodies'][0].set_facecolor("None")
        parts['bodies'][0].set_edgecolor(color)
        parts['bodies'][0].set_linewidth(2)
        parts['bodies'][0].zorder = 10
        tick_pos+=[pos]
        tick_labels+=[f/12.0]
    

# Misc:
plt.plot([0.5,pos+.5], [growth_threshold]*2, '#bbbbbb', zorder = 0, linewidth=3)
plt.plot([2.5,2.5], [-2, 3], color="xkcd:light blue", zorder = 0, linewidth=3)

# Labelling:
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([-2,3])
plt.xlim([0, 8])
plt.xticks(tick_pos, tick_labels)
plt.xlabel("Time (hours)")
plt.ylabel("Growth rate (1/hour)")

plt.tight_layout()
plt.savefig(save_folder+f"/Panel_D_distros.png", dpi=300)
plt.savefig(save_folder+f"/Panel_D_distros.svg", dpi=300)
plt.savefig(save_folder+f"/Panel_D_distros.pdf", dpi=300)
plt.show()

#%% Panel E - Percentage of growing cells

growth_threshold = .3

plt.figure()
for obj_lvl, color in zip(obj_levels, obj_colors):
    
    # Compile growth for objective level throughout replicate XPs:
    obj_growth = []
    for xp in xp_data:
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        growth = xp["avg_growth"][obj_cells]
        obj_growth.append(growth)
    obj_growth = np.concatenate(obj_growth, axis=0)
    
    x = np.arange(0,cutoff-1,1)/12
    ratio = np.nanmean(obj_growth>=growth_threshold, axis=0)
    plt.plot(x, ratio, color = color)
    print(f"Obj {obj_lvl} end fraction: {np.nanmean(ratio[-12:])}")

plt.plot([antibio_time/12, antibio_time/12],[0,1],"--", color="grey", zorder=0)
plt.ylim([0,1])
plt.ylabel("Survival rate")
plt.xlabel("time (hours)")
plt.savefig(save_folder+"/Panel_E_survival_rate.png", dpi=300)
plt.savefig(save_folder+"/Panel_E_survival_rate.svg", dpi=300)
plt.savefig(save_folder+"/Panel_E_survival_rate.pdf", dpi=300)
plt.show()

#%% SI Fig. 16 - All cells growth histogram

growth = []
for xp in xp_data:
    growth.append(xp["avg_growth"])
growth = np.concatenate(growth, axis=0)

plt.figure()
plt.hist(growth.flatten(), density=True, bins=100)
plt.xlim([-2, 3])
yl = plt.ylim()
plt.plot([.3, .3], yl, "gray")
plt.text(.35, yl[1]*.9, "Healthy", color="gray")
plt.text(.22, yl[1]*.9, "Dying", color="gray", ha = "right")
plt.ylim(yl)
plt.xlabel("growth rate (1/hour)")
plt.ylabel("count (normalized)")

plt.savefig(save_folder+"/SI_Fig_16_All_cells_growth.png", dpi=300)
plt.savefig(save_folder+"/SI_Fig_16_All_cells_growth.svg", dpi=300)
plt.savefig(save_folder+"/SI_Fig_16_All_cells_growth.pdf", dpi=300)
plt.show()


#%% SI Fig. 17 - Growing/Dying Kymographs
sys.path.append("D:/delta")
import delta

growth_threshold = .3
interval=3

selected_cells = {
    1: {"growing": [2, 1918], "stopped": [2, 1766]},
    800: {"growing": [0, 58], "stopped": [2, 1465]},
    1200: {"growing": [0, 689], "stopped": [1, 771]},
    1800: {"growing": [1, 1075], "stopped": [1, 1753]},
    4095: {"growing": [1, 1853], "stopped": [0, 1233]},
    }

def get_kymograph(xpf, cell_nb, interval=3):
    
    with open(xpf+"/roi_boxes.pkl","rb") as f:
        roi_boxes = pickle.load(f)
    
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
    for f in range(0, cutoff, interval):
        img_stack += [cv2.imread(
            xpf+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
    
    # Drift correction:
    img_stack, drift_values = delta.utilities.driftcorr(
        np.array(img_stack), template=pos.drifttemplate, box=pos.driftcorbox
        )
    
    kymograph_img = []
    seg_input = []
    for f in range(img_stack.shape[0]):
        
        # Crop:
        chamber_img = delta.utilities.cropbox(img_stack[f], pos.rois[roi_nb].box)
        
        # Appent to segmentation inputs stack:
        seg_frame = delta.utils.rangescale(chamber_img, (0,1))
        seg_frame = cv2.resize(seg_frame, (32, 256))
        seg_input.append(seg_frame[:,:,np.newaxis])
        
        # Append to RGB kymograph:
        chamber_img = (chamber_img.astype(np.float64)-np.min(img_stack[f]))/np.ptp(img_stack[f])
        chamber_img = (chamber_img*255).astype(np.uint8)
        chamber_img = np.repeat(chamber_img[:,:,np.newaxis], 3, axis=2)
        kymograph_img.append(chamber_img)
    
    return kymograph_img

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    x, c = selected_cells[obj_lvl]["growing"]
    print(f"Obj {obj_lvl} healthy: XP {x}, cell {c}")
    
    # Get cell kymograph:
    kymograph_img = get_kymograph(raw_data+experiments[x], c, interval)
    _width = kymograph_img[0].shape[1]
    line = int(antibio_time/interval)*_width
    kymograph_img = np.concatenate(kymograph_img, axis=1)
    cv2.imwrite(save_folder+f"/kymographs/kymograph_obj{obj_lvl:04d}_growing_img.png", kymograph_img)
    
    # Show it:
    plt.figure()
    plt.imshow(kymograph_img)
    plt.plot([line]*2, [0,kymograph_img.shape[0]],'--', color="grey")
    ticks = np.arange(0, cutoff/12, 2)
    plt.xticks(ticks*_width*(12/interval), ticks)
    plt.title(obj_lvl)
    plt.ylabel("Dying")
    plt.show()
    
    x, c = selected_cells[obj_lvl]["stopped"]
    print(f"Obj {obj_lvl} dead: XP {x}, cell {c}")
    
    # Get cell kymograph:
    kymograph_img = get_kymograph(raw_data+experiments[x], c, interval)
    _width = kymograph_img[0].shape[1]
    line = int(antibio_time/interval)*_width
    kymograph_img = np.concatenate(kymograph_img, axis=1)
    cv2.imwrite(save_folder+f"/kymographs/kymograph_obj{obj_lvl:04d}_dying_img.png", kymograph_img)
    
    # Show it:
    plt.imshow(kymograph_img)
    plt.plot([line]*2, [0,kymograph_img.shape[0]],'--', color="grey")
    ticks = np.arange(0, cutoff/12, 2)
    plt.xticks(ticks*_width*(12/interval), ticks)
    plt.title(obj_lvl)
    plt.ylabel("Growing")
    plt.show()


#%% SI Fig. 18 - Alive/dead cells mean growth rates:

growth_threshold = .3

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    # Compile growth for objective level throughout replicate XPs:
    obj_growth = []
    for xp in xp_data:
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        growth = xp["growth"][obj_cells]
        obj_growth.append(growth)
    obj_growth = np.concatenate(obj_growth, axis=0)
    
    # Split into alive and dead at each timepoint, compute median:
    alive_growth = []
    dead_growth = []
    for f in range(obj_growth.shape[1]):
        growth_f = obj_growth[:,f]
        growth_f = growth_f[~np.isnan(growth_f)]
        alive_growth += [np.median(growth_f[growth_f>=growth_threshold])]
        dead_growth += [np.median(growth_f[growth_f<growth_threshold])]
    
    # Plot both:
    x = np.arange(0,cutoff-1,1)/12
    plt.plot(x, alive_growth, color = color)
    plt.plot(x, dead_growth, color = color)

plt.plot([antibio_time/12, antibio_time/12],[-0.2,1.3],"--", color="grey", zorder=0)
plt.ylabel("Growth rate (1/hour)")
plt.xlabel("time (hours)")
plt.grid(axis="y")
plt.savefig(save_folder+"/SI_Fig_18_split_growth_rate.png", dpi=300)
plt.savefig(save_folder+"/SI_Fig_18_split_growth_rate.svg", dpi=300)
plt.savefig(save_folder+"/SI_Fig_18_split_growth_rate.pdf", dpi=300)
plt.show()


#%% SI Movie 4 - Growth violin plots

growth_threshold = .3

movie = []
for f in range(cutoff-1):
    
    
    fig = plt.figure(dpi=300)
    pos=0
    
    for obj_lvl, color in zip(obj_levels, obj_colors):
        
        pos+=1
    
        # Compile growth for objective level throughout replicate XPs:
        obj_growth = []
        for xp in xp_data:
            obj_cells = xp["objectives"][:,-1] == obj_lvl
            growth = xp["avg_growth"][obj_cells][:,f]
            growth = growth[~np.isnan(growth)]
            obj_growth.append(growth)
        obj_growth = np.concatenate(obj_growth)
        
        # Filter out top/bottom 0.25%:
        qs = np.quantile(obj_growth,[.0025, .9975])
        obj_growth = obj_growth[obj_growth>qs[0]]
        obj_growth = obj_growth[obj_growth<qs[1]]
        
        # Violin plot for timepoint:
        parts = plt.violinplot(
            positions = [pos],
            dataset = obj_growth,
            showextrema=False,
            widths=.8,
            points=200,
            )
        parts['bodies'][0].set_facecolor(color)
        parts['bodies'][0].set_edgecolor(color)
        parts['bodies'][0].set_linewidth(1)
        parts['bodies'][0].set_alpha(1)
        parts['bodies'][0].zorder = 10
        
    # Misc:
    plt.plot([0.5,pos+.5], [growth_threshold]*2, color="#bbbbbb", zorder = 0, linewidth=2)
    
    #Labelling:
    plt.ylim([-1.75,2.5])
    plt.xlim([0, pos+1])
    plt.xticks([1,2,3,4,5], ["Constant\nred", "800\nobjective", "1200\nobjective", "1800\n objective", "Constant\ngreen"])
    if f >= antibio_time:
        plt.text(3, 2.5, "+Tetracycline", color="xkcd:light blue", ha="center", zorder = 100, size="x-large", weight="bold")
    plt.ylabel("Growth rate ($hour^{-1}$)")
    plt.yticks([-1,0,1,2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Clock:
    plt.text(-0.6, -2.1, f"{int(f/12):02d}h {5*(f%12):02d}m", size="large",
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.25'))
    
    # Figure to numpy image array:
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    X = X[:,:,:3]
    movie.append(np.copy(X))
    
    plt.show()
    plt.clf()

# Write movie to disk:
import delta
delta.utilities.vidwrite(movie, save_folder + "SI_movie_4_growthviolins.mp4")

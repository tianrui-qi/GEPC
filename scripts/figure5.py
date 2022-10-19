# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:45:33 2022

@author: jeanbaptiste
"""
import json
import pickle
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import savgol_filter

import deepcellcontrol as dcc

# experiments path:
xp_path = "Y:/data/Microscope/jeanbaptiste/deepmpc/control/"
experiments = [
    "2022-07-26_DeepMPC_tetA_1/",
    "2022-08-01_DeepMPC_tetA_1/",
    "2022-08-02_DeepMPC_tetA_1/",
    ]

# Save images to:
save_folder = "C:/Users/Administrator/jb/deepmpc_paper/figure5/"

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
    xp_data[xp]["fluo"] = np.load(xp_path+xpname+"cells_fluo.npy")[:,:cutoff]
    xp_data[xp]["fluo"][xp_data[xp]["fluo"]==0] = np.nan
    xp_data[xp]["objectives"] = np.load(xp_path+xpname+"objectives.npy")[:,:cutoff]
    
    # Cell area:
    with open(xp_path+xpname+"fallback_control_parameters.json", "r") as f:
        control_parameters = json.load(f)
    with open(xp_path+xpname+"mothers.pkl", "rb") as f:
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


#%% Smoothed growth:

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

#%% Plot objective results:

# obj_colors = ["#e0bbd2", "#b76b99", "#b33b72", "#8e3563", "#572649"]
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
    plt.savefig(save_folder+"/PanelA_control.png", dpi=300)
    plt.savefig(save_folder+"/PanelA_control.svg", dpi=300)
    plt.savefig(save_folder+"/PanelA_control.pdf", dpi=300)

plt.show()
#%% Plot growth rates over time:

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

plt.savefig(save_folder+"/PanelB_growth.png", dpi=300)
plt.savefig(save_folder+"/PanelB_growth.svg", dpi=300)
plt.savefig(save_folder+"/PanelB_growth.pdf", dpi=300)
plt.show()

#%% Plot growth rate distributions for each experiment and objective level:

timepoints = [12, 102, 114, 126, 138, 150, 162]
growth_threshold = .3

for obj_lvl, color in zip(obj_levels, obj_colors):
    
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
        
        # Violin plot for timepoint:
        parts = plt.violinplot(
            positions = [pos],
            dataset = obj_growth,
            showextrema=False,
            widths=.8,
            points=200,
            )
        parts['bodies'][0].set_facecolor(color)
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][0].set_alpha(1)
        parts['bodies'][0].zorder = 10
        tick_pos+=[pos]
        tick_labels+=[f/12.0]
    
    
    plt.plot([0,pos+1], [growth_threshold]*2, 'k--', zorder = 0, linewidth=.5)
    plt.ylim([-3,4])
    plt.xticks(tick_pos, tick_labels)
    plt.xlabel("Time (hours)")
    plt.ylabel("Growth rate (1/hour)")
    plt.grid(axis="y")
    plt.savefig(save_folder+f"/PanelD_distros/PanelC_obj{obj_lvl:04d}.png", dpi=300)
    plt.savefig(save_folder+f"/PanelD_distros/PanelC_obj{obj_lvl:04d}.svg", dpi=300)
    plt.savefig(save_folder+f"/PanelD_distros/PanelC_obj{obj_lvl:04d}.pdf", dpi=300)
    plt.show()

#%% Plot alive/dead ratios over time:

growth_threshold = .3

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
    print(np.nanmean(ratio[-12:]))

plt.plot([antibio_time/12, antibio_time/12],[0,1],"--", color="grey", zorder=0)
plt.ylim([0,1])
plt.ylabel("Survival rate")
plt.xlabel("time (hours)")
plt.savefig(save_folder+"/survival_rate.png", dpi=300)
plt.savefig(save_folder+"/survival_rate.svg", dpi=300)
plt.savefig(save_folder+"/survival_rate.pdf", dpi=300)
plt.show()

#%% Alive/dead cells mean growth rates:

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
plt.savefig(save_folder+"/split_growth_rate.png", dpi=300)
plt.savefig(save_folder+"/split_growth_rate.svg", dpi=300)
plt.savefig(save_folder+"/split_growth_rate.pdf", dpi=300)
plt.show()

#%% Kymographs
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
    
    
    # # Segment images:
    # seg_model = tf.keras.models.load_model(xpf+"/delta_segmentation.hdf5")
    # seg_stack = seg_model.predict(np.array(seg_input), verbose=1)
    # seg_stack = (seg_stack>.5).astype(np.uint8)
    # seg_stack = seg_stack[:,:,:,0]
    # for f, seg in enumerate(seg_stack):
    #     seg_stack[f,:,:] = delta.utilities.opencv_areafilt(seg, min_area = 100)
    
    # # Add contour to kymograph:
    # for chamber_img, seg_mask in zip(kymograph_img, seg_stack):
    #     seg_mask = cv2.resize(seg_mask,chamber_img.shape[1::-1])
    #     contours = delta.utilities.find_contours((seg_mask>0.5).astype(np.uint8))
    #     mother_cnt = [sorted(contours,key=lambda cnt: min(cnt[:,:,1]))[0]]
    #     chamber_img = cv2.drawContours(
    #         chamber_img, mother_cnt, 0, [255,255,255], thickness=1
    #         )
    
    return kymograph_img

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    # Compile growth for objective level throughout replicate XPs:
    # obj_growth = []
    # xp_attrib = []
    # cell_nb = []
    # for _x, xp in enumerate(xp_data):
    #     obj_cells = xp["objectives"][:,-1] == obj_lvl
    #     growth = xp["avg_growth"][obj_cells]
    #     obj_growth.append(growth)
    #     xp_attrib.append(np.full(growth.shape[0], _x))
    #     cell_nb.append(
    #         np.arange(0,xp["avg_growth"].shape[0],1,dtype=int)[obj_cells]
    #         )
    # obj_growth = np.concatenate(obj_growth, axis=0)
    # xp_attrib = np.concatenate(xp_attrib, axis=0)
    # cell_nb = np.concatenate(cell_nb, axis=0)
    
    # Get random healthy cell:
    # pick = np.random.choice(np.where(obj_growth[:,-6]>.4)[0])
    # cell_growth = obj_growth[pick,:]
    # x = xp_attrib[pick]
    # c = cell_nb[pick]
    x, c = selected_cells[obj_lvl]["growing"]
    print(f"Obj {obj_lvl} healthy: XP {x}, cell {c}")
    
    # Get cell kymograph:
    kymograph_img = get_kymograph(xp_path+experiments[x], c, interval)
    _width = kymograph_img[0].shape[1]
    line = int(antibio_time/interval)*_width
    kymograph_img = np.concatenate(kymograph_img, axis=1)
    cv2.imwrite(save_folder+f"/kymographs/kymograph_obj{obj_lvl:04d}_healthy_img.png", kymograph_img)
    
    # Show it:
    plt.imshow(kymograph_img)
    plt.plot([line]*2, [0,kymograph_img.shape[0]],'--', color="grey")
    ticks = np.arange(0, cutoff/12, 2)
    plt.xticks(ticks*_width*(12/interval), ticks)
    plt.title(obj_lvl)
    plt.show()
    
    # Same for "dead" cell:
    # pick = np.random.choice(
    #     np.where(
    #         np.logical_and(obj_growth[:,cutoff-5*12]>.4, obj_growth[:,-6]<.2))[0]
    #     )
    # cell_growth = obj_growth[pick,:]
    # x = xp_attrib[pick]
    # c = cell_nb[pick]
    
    x, c = selected_cells[obj_lvl]["stopped"]
    print(f"Obj {obj_lvl} dead: XP {x}, cell {c}")
    
    # Get cell kymograph:
    kymograph_img = get_kymograph(xp_path+experiments[x], c, interval)
    _width = kymograph_img[0].shape[1]
    line = int(antibio_time/interval)*_width
    kymograph_img = np.concatenate(kymograph_img, axis=1)
    cv2.imwrite(save_folder+f"/kymographs/kymograph_obj{obj_lvl:04d}_dead_img.png", kymograph_img)
    
    # Show it:
    plt.imshow(kymograph_img)
    plt.plot([line]*2, [0,kymograph_img.shape[0]],'--', color="grey")
    ticks = np.arange(0, cutoff/12, 2)
    plt.xticks(ticks*_width*(12/interval), ticks)
    plt.title(obj_lvl)
    plt.show()

#%% SI Figure XX: Split fluorescence control results:


def plotq(fluo, q=.5, color="blue"):
    
    x = np.arange(0,len(fluo[0]),1)/12
    plt.plot(x,np.nanmedian(fluo,axis=0), color=color, zorder = 100)
    plt.fill_between(
        x,
        np.nanquantile(fluo,.5-q/2,axis=0),
        np.nanquantile(fluo,.5+q/2,axis=0),
        color=color,
        alpha=.3,
        )

plt.figure(figsize=(6, 9), dpi=300)
_o = 0
for obj_lvl, color in zip(obj_levels, obj_colors):
    
    x = np.arange(0,cutoff,1)/12
    
    all_fluo = []
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        fluo = xp["fluo"][obj_cells]
        
        _o+=1
        plt.subplot(5,3,_o)
        if 1 < obj_lvl < 4095:
            plt.plot(
                [3, cutoff/12], [obj_lvl, obj_lvl], "--", color="grey", zorder = 90
                )
        plt.plot([antibio_time/12, antibio_time/12],[0,4095],color="xkcd:light blue", zorder = 90)
        plotq(fluo, color=color)

        if _o == 14:
            plt.xlabel("time (hours)")
        if _o == 7:
            plt.ylabel("Fluorescence (a.u.)")
        if _o >= 13:
            plt.xticks([0, 5, 10])
        else:
            plt.xticks([0, 5, 10], ['']*3)
        if _o in [1, 4, 7, 10, 13]:
            plt.yticks([0, 2000, 4000])
        else:
            plt.yticks([0, 2000, 4000], ['']*3)
        if _o <= 3:
            plt.title(f"Experiment #{_o}")
        plt.ylim([0,4095])
        plt.grid(axis="both")

plt.savefig(save_folder+"/SI_control_separate.png", dpi=300)
plt.savefig(save_folder+"/SI_control_separate.svg", dpi=300)
plt.savefig(save_folder+"/SI_control_separate.pdf", dpi=300)

plt.show()


#%% SI Figure XX: All cells growth histogram

growth = []
for xp in xp_data:
    growth.append(xp["avg_growth"])
growth = np.concatenate(growth, axis=0)

plt.hist(growth.flatten(), density=True, bins=100)
plt.xlim([-2, 3])
yl = plt.ylim()
plt.plot([.3, .3], yl, "gray")
plt.text(.35, yl[1]*.9, "Healthy", color="gray")
plt.text(.22, yl[1]*.9, "Dying", color="gray", ha = "right")
plt.ylim(yl)
plt.xlabel("growth rate (1/hour)")
plt.ylabel("count (normalized)")

plt.savefig(save_folder+"/All_cells_growth.png", dpi=300)
plt.savefig(save_folder+"/All_cells_growth.svg", dpi=300)
plt.savefig(save_folder+"/All_cells_growth.pdf", dpi=300)
plt.show()


#%% SI Movie XX - Growth violin plots

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
        
        qs = np.quantile(obj_growth,[.005, .995])
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
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][0].set_alpha(1)
        parts['bodies'][0].zorder = 10
        
    
    plt.plot([0,pos+1], [growth_threshold]*2, color="#bbbbbb", zorder = 0, linewidth=2)
    plt.ylim([-1.5,2.5])
    plt.xticks([1,2,3,4,5], ["Constant\nred", "800\nobjective", "1200\nobjective", "1800\n objective", "Constant\ngreen"])
    plt.text(-1, -2, f"{int(f/12):02d}h {5*(f%12):02d}m", size="x-large",
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.25'))
    if f >= antibio_time:
        plt.text(3, 2.25, "+Tetracycline", color="xkcd:light blue", ha="center", zorder = 100, size="x-large", weight="bold")
    plt.ylabel("Growth rate ($hour^{-1}$)")
    plt.yticks([-1,0,1,2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    X = X[:,:,:3]
    movie.append(np.copy(X))
    
    plt.show()
    plt.clf()

# Write movie to disk:
import delta
delta.utilities.vidwrite(movie, save_folder + "SI_movie_distributions.mp4")

#%% Split dead / growing cells:

threshold = 0.025
window = (6, 6)

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    nc = 2
    mix = GaussianMixture(
        n_components=nc,
        warm_start=True,
        means_init=np.linspace(0,.12,nc)[:,np.newaxis]
        )
    
    x = []
    dead_g = []
    surv_g = []
    for f in range(window[0],167-window[1]):
        time_span = list(range(f-window[0], f+window[1]))
        
        pos = 0
        o = 0
            
        obj_growth = []
        for xp in xp_data:
            
            # pos+=1
        
            obj_cells = xp["objectives"][:,-1] == obj_lvl
            # growth = np.nanmean(xp["growth"][obj_cells][:,time_span], axis=1)
            growth = xp["growth"][obj_cells][:,f]
            growth = growth[~np.isnan(growth)]
            obj_growth.append(growth)
        
        pos+=1
        obj_growth = np.concatenate(obj_growth)
        
        x+=[f/12]
        dead_g+=[np.median(obj_growth[obj_growth<threshold])]
        surv_g+=[np.median(obj_growth[obj_growth>=threshold])]
    
    plt.plot(x, dead_g, color=color)
    plt.plot(x, surv_g, color=color)
    

plt.plot([0,14], [threshold, threshold], 'k--', linewidth=.5)
plt.ylabel("growth rate (norm.)")
plt.xlabel("time (hours)")
plt.ylim([-0.03, 0.13])
plt.savefig(save_folder+"/mean_growth_bimodal.png", dpi=300)
plt.savefig(save_folder+"/mean_growth_bimodal.svg", dpi=300)
plt.savefig(save_folder+"/mean_growth_bimodal.pdf", dpi=300)
plt.show()


#%% Synchronize around death

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    obj_growth = []
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        growth = xp["avg_growth"][obj_cells]
        obj_growth.append(growth)
        
    obj_growth = np.concatenate(obj_growth, axis=0)

    # plt.plot( np.transpose(obj_growth), color=color, alpha=.02)
    # plt.show()

    
    first_dying = np.argmax(obj_growth<.3, axis=1)
    x = np.arange(0, obj_growth.shape[1], 1)/12
    death_centered = np.full(
        shape=(obj_growth.shape[0], obj_growth.shape[1]*2),
        fill_value = np.nan
        )
    c = 0
    for cell_growth, death_time in zip(obj_growth, first_dying):
        
        if death_time==0:
            continue
        
        # plt.plot(x - death_time/12, cell_growth, color="r", alpha=.01)
        
        t = obj_growth.shape[1]-death_time
        death_centered[c,t:t+len(cell_growth)] = cell_growth
        c+=1
        
    plt.title(obj_lvl)
    # plt.show()
    
    x = np.arange(-obj_growth.shape[1], obj_growth.shape[1], 1)/12
    plt.plot(x, np.nanmedian(death_centered, axis=0), color=color)
    
plt.plot([-15,15], [.3,.3], "gray")
plt.ylim([-.5, 1.5])
plt.xlim([-10, 5])
    # plt.show()
    

#%% Growth scatter plots

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    for f in range(9*12, 13*12-1):
    
        for xp in xp_data:
            
            obj_cells = xp["objectives"][:,-1] == obj_lvl
            freegrowth = np.nanmean(xp["growth"][obj_cells][:,8*12:9*12], axis=1)
            abgrowth = np.nanmean(xp["growth"][obj_cells][:,f:f+12], axis=1)
            pop_out = np.logical_or(np.isnan(freegrowth), np.isnan(abgrowth))
            freegrowth = freegrowth[~pop_out]
            abgrowth = abgrowth[~pop_out]
            plt.scatter(freegrowth, abgrowth, color = color, alpha=.2)
            # obj_growth.append(growth)
        
        plt.plot([-.2,3], [.025, .025], 'k--', linewidth=.5)
        plt.plot([.025, .025], [-.2,3], 'k--', linewidth=.5)
        plt.xlim([-.2,.3])
        plt.ylim([-.2,.3])
        plt.title(clock(f))
        plt.show()


#%%
import scipy.stats as st

f = 10*12

fluo_all = []
ab_all = []
for xp in xp_data:
    
    fluo = np.nanmean(xp["fluo"][:,8*12:9*12], axis=1)
    abgrowth = np.nanmean(xp["growth"][:,f:f+12], axis=1)
    pop_out = np.logical_or(np.isnan(fluo), np.isnan(abgrowth))
    fluo = fluo[~pop_out]
    abgrowth = abgrowth[~pop_out]
    fluo_all.append(fluo)
    ab_all.append(abgrowth)

fluo_all = np.concatenate(fluo_all,axis=0)
ab_all = np.concatenate(ab_all,axis=0)

plt.scatter(fluo_all, ab_all, alpha=.2)

# plt.hist(fluo_all, bins=100)

#%% 

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        fluo = xp["fluo"][obj_cells,:-1]
        growth = xp["avg_growth"][obj_cells]
        fluo[growth<.025] = np.nan
        plotq(fluo, color=color)
        

    plt.xlabel("time (hours)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.ylim([0,4095])
    # plt.savefig(save_path+"/fluo_tetA_objectives.png", dpi=300)
    plt.show()

plt.show()

#%%

all_growth = []
for xp in xp_data:
    
    all_growth.append(xp["avg_growth"].flatten())

all_growth = np.concatenate(all_growth)
all_growth = all_growth[~np.isnan(all_growth)]

parts = plt.violinplot(
    positions = [1],
    dataset = all_growth,
    showextrema=False,
    widths=.8,
    points=200,
    )

# Customize it:
parts['bodies'][0].set_facecolor(color)
parts['bodies'][0].set_edgecolor('black')
parts['bodies'][0].set_alpha(.9)
parts['bodies'][0].zorder = 10


#%% Control for survivors


for obj_lvl, color in zip(obj_levels, obj_colors):
    
    plt.plot([3, cutoff/12], [obj_lvl, obj_lvl], "--", color="grey")
    plt.plot([antibio_time/12, antibio_time/12],[0,4095],"--", color="grey")
    
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        fluo = xp["fluo"][obj_cells,:-1]
        growth = xp["avg_growth"][obj_cells]
        fluo[growth<.025] = np.nan
        plotq(fluo, color=color)
        

    plt.xlabel("time (hours)")
    plt.ylabel("Fluorescence (a.u.)")
    plt.ylim([0,4095])
    # plt.savefig(save_path+"/fluo_tetA_objectives.png", dpi=300)
    plt.show()

plt.show()

#%% Control performance as function of growth rate:
    
for xp in xp_data:
    
    xp["rmse"] = np.zeros_like(xp["growth"])
    
    fluo = xp["fluo"][:,:-1]
    obj = xp["objectives"][:, :-1]
    
    
    for f in range(cutoff-1):
        f_min = f-window[0]
        if f_min < 0: f_min = 0
        f_max = f+window[1]
        if f_max > cutoff: f_max = cutoff
        
        xp["rmse"][:, f] = np.sqrt(
            np.nanmean(np.square((fluo[:,f_min:f_max]-obj[:,f_min:f_max])), axis = 1)
            )


        # plotq(fluo, color=color)



def quantilex_partitiony(arr_x, arr_y, q = 10):
    
    q = np.linspace(0,1,q+1)
    x = np.nanquantile(arr_x, q)
    
    partition = []
    for _x in range(len(x)-1):
        
        indexes = np.logical_and(arr_x>=x[_x], arr_x<x[_x+1])
        partition.append(arr_y[indexes])
    
    return x, partition
        
        

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    rmse = []
    growth = []
    
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        rmse.append(xp["rmse"][obj_cells,36:].flatten())
        growth.append(xp["growth"][obj_cells,36:].flatten())
        
    rmse = np.concatenate(rmse, axis=0)
    growth = np.concatenate(growth, axis=0)
        
    x, partition = quantilex_partitiony(growth, rmse, q=10)
    plt.plot(x[:-1],[np.nanmean(p) for p in partition], '-o', color=color)

#%%

for obj_lvl, color in zip(obj_levels, obj_colors):
    
    rmse = []
    area = []
    
    for xp in xp_data:
        
        obj_cells = xp["objectives"][:,-1] == obj_lvl
        rmse.append(xp["rmse"][obj_cells,36:].flatten())
        area.append(xp["area"][obj_cells,36:-1].flatten())
        
    rmse = np.concatenate(rmse, axis=0)
    area = np.concatenate(area, axis=0)
        
    x, partition = quantilex_partitiony(area, rmse, q=10)
    plt.plot(x[:-1],[np.nanmean(p) for p in partition], '-o', color=color)


            
            
            
        
        # plt.scatter(growth.flatten(), rmse.flatten(), color=color, alpha=.02)
    
    # plt.show()
#     plt.xlabel("time (hours)")
#     plt.ylabel("Fluorescence (a.u.)")
#     plt.ylim([0,4095])
#     # plt.savefig(save_path+"/fluo_tetA_objectives.png", dpi=300)
#     plt.show()

# plt.show()
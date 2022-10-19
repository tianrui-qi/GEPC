# -*- coding: utf-8 -*-

import pickle
import sys
import os

sys.path.insert(0, "D:/delta")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import delta

import deepcellcontrol as dcc

# Sinewaves:
movie_folders = (
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-09_DeepMPC_sinemovie_1",
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-11_DeepMPC_sinemovie_2",
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-24_DeepMPC_sinemovie_3"
    )
movie_shape = (100,100)
cutoff = 19*12
save_folder = "D:/tmp/concentric/"
# moviecells = (
#     (20,80), (20,20), (80,20), (50,50),
#     (40,60), (60,40), (60,60), (40,40),
#     (30,70), (70,30), (70,70), (30,30),
#     (15,85), (85,15), (85,85), (15,15),
#     (45,55), (55,45), (55,55), (45,45),
#     )

# 2001:
# movie_folders = (
#     "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-28_DeepMPC_2001_1",
#     "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-06-01_DeepMPC_2001_2",
#     "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-06-04_DeepMPC_2001_3"
#     )
# movie_shape = (80,125)
# cutoff = 32*12
# save_folder = "D:/tmp/2001/"


# Retrieve global objectives & shuffling / deshuffling lists
xpf = movie_folders[0]
objectives = np.load(xpf + "/whole_movie_objectives.npy")
shuffling = np.load(xpf + "/whole_movie_shuffling.npy")
deshuffling = np.load(xpf + "/whole_movie_deshuffling.npy")

#%% Go through experiments and reconstruct movie:
whole_movie = np.zeros(
    shape=(movie_shape[0]*movie_shape[1],cutoff), dtype = np.float32
    )
obj_movie = whole_movie.copy()
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
    
    # Record pixel <-> xp correspondance
    pixels_to_xp[local_shuffle,0] = xp_ind
    pixels_to_xp[local_shuffle,1] = np.arange(local_obj.shape[0])

whole_movie = np.reshape(whole_movie,movie_shape+(cutoff,))
obj_movie = np.reshape(obj_movie,movie_shape+(cutoff,))
pixels_to_xp = np.reshape(pixels_to_xp,movie_shape+(2,))

# Save to disk:
np.save(save_folder + "whole_movie.npy", whole_movie)
np.save(save_folder + "obj_movie.npy", obj_movie)
np.save(save_folder + "pixels_to_xp.npy", whole_movie)

#%% Create RGB movie:
whole_movie_rgb = np.zeros(shape=whole_movie.shape+(3,), dtype=np.uint8)
obj_movie_rgb = whole_movie_rgb.copy()

colormap = dcc.utilities.gfpmap

for f in range(cutoff):
    
    whole_movie_rgb[:,:,f,:] = dcc.utilities.color_img(
        whole_movie[:,:,f], vmin=0.05, cmap = colormap
        )*255
    obj_movie_rgb[:,:,f,:] = dcc.utilities.color_img(
        obj_movie[:,:,f], vmin=0.05, cmap = colormap
        )*255
    plt.imshow(
        np.concatenate((obj_movie_rgb[:,:,f], whole_movie_rgb[:,:,f]),axis = 1)
        )
    plt.xlabel(f)
    plt.show()

whole_movie_rgb = np.moveaxis(whole_movie_rgb, 2, 0)
obj_movie_rgb = np.moveaxis(obj_movie_rgb, 2, 0)

delta.utilities.vidwrite(whole_movie_rgb, save_folder + "whole_movie.mp4")
delta.utilities.vidwrite(
    np.concatenate((obj_movie_rgb, whole_movie_rgb),axis=2),
    save_folder + "whole_movie_obj.mp4"
    )

#%% Extract movies for all cells of interest:

# Load delta just once since they are all the same:
delta.config.load_config(xpf+"/delta_config.json")

for c, pix in enumerate(moviecells):
    
    # XP & cell indexes:
    xp_ind = pixels_to_xp[pix[0], pix[1], 0]
    cell_nb = pixels_to_xp[pix[0], pix[1], 1]
    
    # Load relevant data:
    xpf = movie_folders[xp_ind]
    local_obj = np.load(xpf + "/local_objectives.npy")
    local_shuffle = np.load(xpf + "/local_shuffling.npy")
    local_deshuffle = np.load(xpf + "/local_deshuffling.npy")
    cells_fluo = np.load(xpf + "/cells_fluo.npy")
    with open(xpf+"/roi_boxes.pkl","rb") as f:
        roi_boxes = pickle.load(f)
    with open(xpf+"/drift.pkl","rb") as f:
        drift = pickle.load(f)
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
                    box = _roi
                    drift = drift[s][p]
                _cell += 1
            total_pos+=1
    pos = delta.pipeline.load_position(xpf+f"/delta_positions/Pos{pos_nb:06d}.pkl")
    pos.drift_values = np.flip(np.transpose(np.array(drift)[:,:,0]), axis=0)
    
    # Load images:
    img_stack = []
    fluo_stack = []
    for f in range(cutoff):
        img_stack += [cv2.imread(
            xpf+f"/pos{pos_nb+1:04d}/chan01_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        fluo_stack += [cv2.imread(
            xpf+f"/pos{pos_nb+1:04d}/chan02_frame{f+1:06d}.tif",
            cv2.IMREAD_ANYDEPTH
            )]
        print(f)
    
    # Drift correction:
    img_stack, pos.drift_values = delta.utilities.driftcorr(
        np.array(img_stack), template=pos.drifttemplate, box=pos.driftcorbox
        )
    fluo_stack, _ = delta.utilities.driftcorr(
        np.array(fluo_stack), drift=pos.drift_values
        )
    
    # Segment images:
    x = []
    for f, img in enumerate(img_stack):
        inputs, _ = pos.rois[roi_nb].get_segmentation_inputs(img)
        x += [inputs]
    x = np.concatenate(x,axis=0)
    seg_stack = seg_model.predict(x, verbose=1)
    seg_stack = (seg_stack>.5).astype(np.uint8)
    seg_stack = seg_stack[:,:,:,0]
    for f, seg in enumerate(seg_stack):
        seg_stack[f,:,:] = delta.utilities.opencv_areafilt(seg, min_area = 100)
    
    # Get mother contour:
    mother = []
    mother_stack = []
    dsize = (
        pos.rois[roi_nb].box["xbr"] - pos.rois[roi_nb].box["xtl"],
        pos.rois[roi_nb].box["ybr"] - pos.rois[roi_nb].box["ytl"]
        )
    for mask in seg_stack:
        mask = cv2.resize(mask,dsize)
        plt.imshow(mask)
        plt.show()
        contours = delta.utilities.find_contours((mask>0.5).astype(np.uint8))
        mother += [sorted(contours,key=lambda x: min(x[:,:,1]))[0]]
    
    # Crop out fluo frames:
    fluo_chamber = np.array([delta.utils.cropbox(x, pos.rois[roi_nb].box) for x in fluo_stack])
    
    # Save to disk:
    cell_folder = save_folder+f"/cellmovie_y{pix[0]:03d}x{pix[1]:03d}/"
    os.makedirs(cell_folder, exist_ok=True)
    with open(cell_folder+"mother_contour.pkl", "wb") as f:
        pickle.dump(mother, f)
    np.save(cell_folder + "fluo_chamber.npy", fluo_chamber)
    

#%% # Generate RGB movies

for c, pix in enumerate(moviecells):
    
    cell_folder = save_folder+f"/cellmovie_y{pix[0]:03d}x{pix[1]:03d}/"
    with open(cell_folder+"mother_contour.pkl", "rb") as f:
        mother = pickle.load(f)
    fluo_chamber = np.load(cell_folder + "fluo_chamber.npy")
    
    rgb = np.empty(shape = fluo_chamber.shape + (3,), dtype = np.uint8)
    rgb_mother = np.empty(shape = fluo_chamber.shape + (3,), dtype = np.uint8)
    for f, fluo_img in enumerate(fluo_chamber):

        rgb[f] = dcc.utilities.color_img(
            fluo_img, vmin=0.05
            )*255
        
        rgb_mother[f] = cv2.drawContours(
            rgb[f].copy(),
            mother,
            f,
            color=(255,255,255),
            thickness=1,
        )
        
        plt.imshow(rgb_mother[f])
        plt.show()
    
    np.save(cell_folder + "fluo_rgb.npy", rgb)
    np.save(cell_folder + "fluo_rgb_mother.npy", rgb_mother)

#%% Display cell movies:

cell_folder = save_folder+"/cellmovie_y015x015/"
rgb_mother = np.load(cell_folder+"fluo_rgb_mother.npy")

for f, frame in enumerate(rgb_mother):
    plt.imshow(frame)
    plt.xlabel(f)
    plt.show()

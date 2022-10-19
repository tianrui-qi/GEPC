# -*- coding: utf-8 -*-

import pickle

import numpy as np

import deepcellcontrol as dcc

movie_folders = (
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-09_DeepMPC_sinemovie_1",
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-11_DeepMPC_sinemovie_2",
    "Z:/data/Microscope/jeanbaptiste/deepmpc/control/2022-05-24_DeepMPC_sinemovie_3"
    )
movie_shape = (100,100)
cutoff = 19*12

# First, retrieve global objectives & shuffling / deshuffling lists
xpf = movie_folders[0]
objectives = np.load(xpf + "/whole_movie_objectives.npy")
shuffling = np.load(xpf + "/whole_movie_shuffling.npy")
deshuffling = np.load(xpf + "/whole_movie_deshuffling.npy")

# Then, go through experiments and reconstruct movie:
whole_movie = np.zeros(shape=(movie_shape[0]*movie_shape[1],cutoff), dtype = np.float32)
for xpf in movie_folders:
    
    # Load local objectives, shuffling / deshuffling:
    local_obj = np.load(xpf + "/local_objectives.npy")
    local_shuffle = np.load(xpf + "/local_shuffling.npy")
    local_deshuffle = np.load(xpf + "/local_deshuffling.npy")
    
    # Cells fluorescence:
    cells_fluo = np.load(xpf + "/cells_fluo.npy")
    
    whole_movie[local_shuffle] = cells_fluo

whole_movie = np.reshape(whole_movie,movie_shape+(cutoff,))

# Save to disk:
np.save("D:/tmp/concentric/whole_movie.npy", whole_movie)

# Create RGB movie:
whole_movie_rgb = np.zeros(shape=whole_movie.shape+(3,), dtype=np.uint8)

import time
import matplotlib.pyplot as plt
for f in range(cutoff):
    
    print(f)
    whole_movie_rgb[:,:,f,:] = dcc.utilities.color_img(whole_movie[:,:,f], vmin=0.05)*255
    plt.imshow(whole_movie_rgb[:,:,f])
    plt.show()
    time.sleep(.1)
     

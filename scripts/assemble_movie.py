# -*- coding: utf-8 -*-

import pickle

import numpy as np

movie_folders = (
    "E:/deepmpc/control/2022-05-09_DeepMPC_sinemovie_1/",
    )
movie_shape = (100,100)
cutoff = 19*12

# First, retrieve global objectives & shuffling / deshuffling lists
xpf = movie_folders[0]
objectives = np.load(xpf + "/whole_movie_objectives.npy")
shuffling = np.load(xpf + "/whole_movie_shuffling.npy")
deshuffling = np.load(xpf + "/whole_movie_deshuffling.npy")

# Then, go through experiments and reconstruct movie:
whole_movie = np.zeros(shape=movie_shape+(cutoff,), dtype = np.float32)
for xpf in movie_folders:
    
    # Load local objectives, shuffling / deshuffling:
    local_obj = np.load(xpf + "/local_objectives.npy")
    local_shuffle = np.load(xpf + "/local_shuffling.npy")
    local_deshuffle = np.load(xpf + "/local_deshuffling.npy")
    
    # Cells data:
    with open(xpf+"/mothers.pkl", "rb") as f:
        mothers = pickle.load(f)
    
    
    cell_nb = 0
    for s in mothers:
        for p in s:
            for mother in p:
                fluo = mother[:cutoff, 0]
                
                

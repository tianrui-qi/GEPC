# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:38:12 2022

@author: jeanbaptiste
"""

import os
import time
import pickle

import numpy as np

import deepcellcontrol as dcc

training_sets = "E:/deepmpc/trainingsets/"
experiments = os.listdir(training_sets)

for xp in experiments:

    raw_dataset = dcc.data.compile_dataset(training_sets + xp)

    dataset = {
        "analysis_date": time.strftime("%Y-%m-%d"),
        "experiment": training_sets + xp,
        "raw_dataset": raw_dataset
        }
    
    # Save dataset:
    savefolder = training_sets + xp + "/deepcellcontrol_dataset/"
    os.makedirs(savefolder, exist_ok = True)
    with open(savefolder + xp + "_dataset.pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    # Plot examples:
    for cell_nb in np.random.choice(raw_dataset["stims"].shape[0], 100, replace = False):
        dcc.data.single_cell_plot(
            raw_dataset,
            cell_nb,
            savefig = savefolder + f"cell_{cell_nb}"
            )

# otf = mothers_pkl[3][81-75]
# otd = dict()
# for k in dataset:
#     otd[k] = []
#     for c in otf:
#         if k in c:
#             otd[k] += [np.array(c[k])[:199,np.newaxis]]

# for k in dataset:
#     if len(otd[k]) > 0:
#         for c in range(len(otd[k])):
#             if np.any(otd[k][c]-dataset[k][c] > 1e-3):
#                 plt.plot(otd[k][c])
#                 plt.plot(dataset[k][c])
#                 plt.title((k,c))
#                 plt.show()

# k = "sharpness"
# c = 1
# print(otd[k][c]-dataset[k][c])
# print(otd[k][c]-dataset[k][c] > 1e-3)
# plt.plot(otd[k][c])
# plt.plot(dataset[k][c])
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:38:12 2022

@author: jeanbaptiste
"""

import os
import time
import pickle
import sys

import numpy as np

sys.path.insert(0,"/usr2/postdoc/jlugagne/delta")
sys.path.insert(0,"/project/dunlop/shared_python_packages/deepcellcontrol")
import deepcellcontrol as dcc

training_sets = "/ad/eng/research/eng_research_dunloplab/data/Microscope/jeanbaptiste/deepmpc/trainingsets/"
xp = sys.argv[1]

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

# training_sets=(
#     "2022-04-13_TrainingSet2_dataset.pkl",
#     "2022-04-19_TrainingSet4_dataset.pkl",
#     "2022-04-22_TrainingSet6_dataset.pkl",
#     "2022-04-24_TrainingSet8_dataset.pkl",
#     "2022-04-15_TrainingSet3_dataset.pkl",
#     "2022-04-21_TrainingSet5_dataset.pkl",
#     "2022-04-23_TrainingSet7_dataset.pkl",
#     )

# for ts in training_sets:
#     with open("D:/shared_packages/deepcellcontrol/assets/data/experimental/before_stims_shift/"+ts, "rb") as f:
#         dataset = pickle.load(f)
        
#     stims = dataset["raw_dataset"]["stims"]
    
#     stims = np.roll(stims,1, axis=1)
    
#     stims[:,0,:] = 0
    
#     dataset["raw_dataset"]["stims"] = stims
    
#     with open("D:/shared_packages/deepcellcontrol/assets/data/experimental/"+ts, "wb") as f:
#         pickle.dump(dataset,f)
    
    
    

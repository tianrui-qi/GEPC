# -*- coding: utf-8 -*-
"""
This script generates plots for Figure 1
Note that it uses microscopy images and can not be reproduced from the zenodo
archive data only.

Created on Tue Sep 13 16:19:56 2022

@author: jeanbaptiste
"""
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

import deepcellcontrol as dcc

# Where to save files:
save_folder = "D:/papers/deepmpc/figure1/"

# Path to raw images data (BU ENG NAS):
experiment = "Z:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/2022-04-13_TrainingSet2/"

pos = 29 # Position number
f = 99 # Frame nb

#%% Panel C - Mother Machine Images
# Note: this panel will not work if you do not have access to the raw images 

# Load trans image
iname = experiment + f"/pos{pos+1:04d}/chan01_frame{f+1:06d}.tif"
I = cv2.imread(iname, cv2.IMREAD_ANYDEPTH)
I = I.astype(np.float64)
I = 255*(I - np.min(I))/np.ptp(I)
I = I.astype(np.uint8)
I = np.repeat(I[:,:,np.newaxis], 3, axis=2)

plt.figure()
plt.imshow(I)
plt.show()

cv2.imwrite(save_folder + "/Panel_C_deltaex_orig.png", I[:,:,::-1])

# Load fluo image
iname = experiment + f"/pos{pos+1:04d}/chan02_frame{f+1:06d}.tif"
F = cv2.imread(iname, cv2.IMREAD_ANYDEPTH)
F = F.astype(np.float64)
F = (255*(F - np.min(F))/np.ptp(F)).astype(np.uint8)
F = (255*dcc.utilities.gfpmap(F)[:,:,:3]).astype(np.uint8)

plt.figure()
plt.imshow(F)
plt.show()

cv2.imwrite(save_folder + "/Panel_C_deltaex_origfluo.png", F[:,:,::-1])


#%% Panel D - cell contours
# Note: this panel will not work if you do not have access to the raw images 
# Note: this panel requires DeLTA to run (commit 8ceb015)

sys.path.append("D:/delta")
import delta

# Load delta position
fname = experiment + f"/delta_results/Pos{pos:06d}.pkl"
pos = delta.pipeline.load_position(fname)

# Draw contours in all chambers:
D = np.copy(I)
DF = np.copy(F)
for roi in pos.rois:
    
    offset = (
        int(roi.box["xtl"] + pos.drift_values[1][f]),
        int(roi.box["ytl"] + pos.drift_values[0][f])
        )
    
    cnt = delta.utilities.find_contours((roi.label_stack[f]>0).astype(np.uint8))
    for c in cnt:
        cv2.drawContours(D, [c], 0, (249,249,249), thickness = 2, offset=offset)
        cv2.drawContours(DF, [c], 0, (249,249,249), thickness = 2, offset=offset)

plt.figure()
plt.imshow(D)
plt.show()

plt.figure()
plt.imshow(DF)
plt.show()


cv2.imwrite(save_folder + "/Panel_D_deltaex.png", D[:,:,::-1])
cv2.imwrite(save_folder + "/Panel_D_deltaex_fluo.png", DF[:,:,::-1])
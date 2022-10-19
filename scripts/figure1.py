# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:19:56 2022

@author: jeanbaptiste
"""
import sys
sys.path.append("D:/delta")
import delta

import deepcellcontrol as dcc

import numpy as np
import cv2
import matplotlib.pyplot as plt


f = 99
fname = "Z:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/2022-04-13_TrainingSet2/delta_results/Pos000029.pkl"
pos = delta.pipeline.load_position(fname)

iname = f"Z:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/2022-04-13_TrainingSet2/pos0030/chan01_frame{f+1:06d}.tif"
I = cv2.imread(iname, cv2.IMREAD_ANYDEPTH)
I = I.astype(np.float64)
I = 255*(I - np.min(I))/np.ptp(I)
I = I.astype(np.uint8)
I = np.repeat(I[:,:,np.newaxis], 3, axis=2)



iname = f"Z:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/2022-04-13_TrainingSet2/pos0030/chan02_frame{f+1:06d}.tif"
F = cv2.imread(iname, cv2.IMREAD_ANYDEPTH)
F = F.astype(np.float64)
F = (255*(F - np.min(F))/np.ptp(F)).astype(np.uint8)
# F = (255*(F - 150)/500)
# F[F<0] = 0
# F = F.astype(np.uint8)
F = (255*dcc.utilities.gfpmap(F)[:,:,:3]).astype(np.uint8)
# F = dcc.utilities.color_img(F, vmin=.05, cmap=dcc.utilities.gfpmap)


D = np.copy(I)
DF = np.copy(F)

for roi in pos.rois:
    
    offset = (
        int(roi.box["xtl"] + pos.drift_values[1][f]),
        int(roi.box["ytl"] + pos.drift_values[0][f])
        )
    
    cnt = delta.utilities.find_contours((roi.label_stack[f]==1).astype(np.uint8))
    cv2.drawContours(D, cnt, 0, (229,128,255), thickness = 2, offset=offset)
    cv2.drawContours(DF, cnt, 0, (229,128,255), thickness = 2, offset=offset)
    
    cnt = delta.utilities.find_contours((roi.label_stack[f]>1).astype(np.uint8))
    for c in cnt:
        cv2.drawContours(D, [c], 0, (249,249,249), thickness = 2, offset=offset)
        cv2.drawContours(DF, [c], 0, (249,249,249), thickness = 2, offset=offset)
    
plt.imshow(DF)

cv2.imwrite("D:/deepmpc_paper/figure1/deltaex_orig.png", I[:,:,::-1])
cv2.imwrite("D:/deepmpc_paper/figure1/deltaex.png", D[:,:,::-1])
cv2.imwrite("D:/deepmpc_paper/figure1/deltaex_origfluo.png", F[:,:,::-1])
cv2.imwrite("D:/deepmpc_paper/figure1/deltaex_fluo.png", DF[:,:,::-1])
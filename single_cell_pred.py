#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:17:00 2021

@author: jeanbaptiste
"""

import core
import matplotlib.pyplot as plt
from core.models import lstm
import numpy as np

# Load dataset:
datasets_path = '/home/jeanbaptiste/data/deepcellcontrol/assets/data/'
dataset = core.data.Datasets(
    (
        datasets_path+'experimental/2021-09-20_Dataset_2/dataset.pkl',
        datasets_path+'experimental/2021-09-23_Dataset_3/dataset.pkl',
        datasets_path+'experimental/2021-09-28_Dataset_4/dataset.pkl',
        datasets_path+'experimental/2021-09-30_Dataset_5/dataset.pkl',
        ),
    features=('fluos','length','stims')
    )
dataset.load()
dataset.normalize()
dataset.data_type='normalized_dataset'
dataset.mode='testing'

# Initialize network:
network = lstm(
    past_steps=dataset.past_steps,
    horizon=dataset.horizon,
    features=len(dataset.features),
    learning_rate=1e-2,
    latent_dim=64
    )
network.load_weights('/home/jeanbaptiste/data/deepmpc/models/lstm/2021-10-07_15-16-41/model.hdf5')

#%%

data = dataset
data.batch_size=1
xval, yval = next(data)
yhat = network.predict(xval,verbose=1)
time = np.arange(-data.past_steps, data.horizon, 1)/12


stims = np.concatenate(
    (xval[0][0,:,[f=='stims' for f in data.features].index(True)], xval[1][0,:,0]),
    axis=0
    )
fluo = np.concatenate(
    (xval[0][0,:,[f=='fluos' for f in data.features].index(True)], yval[0,:,0]),
    axis=0
    )
core.utils.OptoPlotBackground(
    stims,
    x=time,
    ymin=0,
    ymax=1
    )
plt.plot(time,fluo,'xkcd:black')
plt.plot(
    time[data.past_steps-1:],
    np.insert(yhat[0,:],0,fluo[data.past_steps-1],axis=0)
    )
plt.plot(
    [time[data.past_steps-1], time[data.past_steps-1]],
    [0,1],
    '--k',
    linewidth=.5)

plt.ylabel('GFP (norm.)')
plt.xlim(time[0],time[-1])
plt.ylim(0,1)
plt.xlabel('time (hours)')
rnum = np.random.randint(1e6)
plt.savefig('./assets/figures/experimental/timeseries_forecasting/lstm_random_sample_%06d.png'%(rnum),dpi=300)
plt.savefig('./assets/figures/experimental/timeseries_forecasting/lstm_random_sample_%06d.pdf'%(rnum),dpi=300)
plt.savefig('./assets/figures/experimental/timeseries_forecasting/lstm_random_sample_%06d.svg'%(rnum),dpi=300)
plt.show()


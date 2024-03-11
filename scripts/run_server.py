# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:09:14 2022

@author: jeanbaptiste
"""

import time
import json
import sys

sys.path.insert(0, "/projectnb/dunlop/JB/deepcellcontrol")
import deepcellcontrol as dcc

models_folder = "/projectnb/dunlop/JB/deepcellcontrol/assets/models/"
# model_folder = models_folder+"2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2/model.hdf5"
model_file = models_folder+"2024-01-30_17-36-02_4fc0cb07-7dda-4580-92fd-26550476275a/model_besteval.hdf5"
log_folder = f"/projectnb/dunlop/JB/deepcellcontrol/assets/onlineserverlogs/{time.strftime('%Y-%m-%d_%H-%M-%S')}/"
horizon = 24
iterations = 25
particles = 40
block = True

# Init controller:
controller = dcc.control.SplitLSTMMPC(
    model_file = model_file,
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=horizon, iterations=iterations, particles=particles
        )
    )

# Init server:
server = dcc.server.Server(controller, device = "GPU")
server.batch_size = 10 # Just to be safe
server.start()

# Init TCP/IP server:
onlineserver = dcc.server.SocketServer(server, port= 7555, savelog=log_folder)
onlineserver.start()

with open(log_folder+"parameters.json", "w") as f:
    parameters = {
        "trained_model": model_file,
        "horizon": horizon,
        "iterations": iterations,
        "particles": particles
        }
    json.dump(parameters, f)

while block:
    time.sleep(1)
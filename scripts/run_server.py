# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:09:14 2022

@author: jeanbaptiste
"""

import time
import json
import sys

sys.path.append("D:/shared_packages/deepcellcontrol")
import deepcellcontrol as dcc

res_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2"
log_folder = f"D:/onlineserverlogs/{time.strftime('%Y-%m-%d_%H-%M-%S')}/"
horizon = 24
iterations = 25
particles = 40
block = True

controller = dcc.control.SplitLSTMMPC(
    model_file = res_folder + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=horizon, iterations=iterations, particles=particles
        )
    )

server = dcc.server.Server(controller, device = "GPU")
server.start()

onlineserver = dcc.server.SocketServer(server, savelog=log_folder)
onlineserver.start()

with open(log_folder+"parameters.json", "w") as f:
    parameters = {
        "trained_model_folder": res_folder,
        "horizon": horizon,
        "iterations": iterations,
        "particles": particles
        }
    json.dump(parameters, f)

while block:
    time.sleep(1)
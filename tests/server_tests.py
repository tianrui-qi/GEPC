# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:18:10 2022

@author: jeanbaptiste
"""

#%% Server

import time
import json

import deepcellcontrol as dcc

res_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2"
log_folder = f"D:/onlineserverlogs/{time.strftime('%Y-%m-%d_%H-%M-%S')}/"
horizon = 24
iterations = 25
particles = 40

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

#%% Client

import numpy as np
import time

import deepcellcontrol as dcc

dummy_dispatcher = lambda output, meta: print(
    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - index {meta['index']} received: {np.array(output, dtype=np.uint8)}"
    )

# Set up local fallback server:
res_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2"
controller = dcc.control.SplitLSTMMPC(
    model_file = res_folder + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=24, iterations=25, particles=40
        )
    )
fallback_server = dcc.server.Server(controller, device = "GPU")
fallback_server.start()

# To test without it:
# fallback_server = None

# Set up client:
client = dcc.server.DistantServer("DESKTOP-A5D6QR1", fallback_server=fallback_server)
client.start()

# Stream control inputs:
for index in range(100_000):
    inputs = np.random.uniform(size=(28,144,8)).astype(np.float32)
    objectives = np.random.uniform(size=(28,24)).astype(np.float32)
    client.queue.put(((inputs, objectives), dict(index=index), dummy_dispatcher))
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - index {index} sent")
    time.sleep(1)
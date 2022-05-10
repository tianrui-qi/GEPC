# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:18:10 2022

@author: jeanbaptiste
"""

import deepcellcontrol as dcc

res_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/2022-05-07_21-52-16_f651d065-2d7b-4f55-8dee-10e48d6e79b2"

controller = dcc.control.SplitLSTMMPC(
    model_file = res_folder + '/model.hdf5',
    strategy_optimizer=dcc.control.BinaryParticleSwarmOptimizer(
        horizon=24, iterations=25, particles=40
        )
    )

server = dcc.server.Server(controller, device = "GPU")
server.start()

onlineserver = dcc.server.SocketServer(server)
onlineserver.start()

#%%

import deepcellcontrol as dcc
import numpy as np
import time

dummy_dispatcher = lambda output, meta: print(
    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - index {meta['index']} received: {np.array(output, dtype=np.uint8)}"
    )

server = dcc.server.DistantServer("127.0.0.1")
server.start()

for index in range(100_000):
    inputs = np.random.uniform(size=(28,144,8)).astype(np.float32)
    objectives = np.random.uniform(size=(28,24)).astype(np.float32)
    server.queue.put(((inputs, objectives), dict(index=index), dummy_dispatcher))
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - index {index} sent")
    time.sleep(1)
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:57:55 2023

@author: jeanbaptiste
"""

import matplotlib.pyplot as plt
import numpy as np

import deepcellcontrol as dcc

# Optogenetic input sequence
# (2 hours no light, 10 hours light. 1 hour = 12 timepoints every 5 minutes)
light_sequence = [0]*24 + [1]*120 + [0, 0, 0, 1]*60 + [0] * 120 + [0, 0, 0, 1] * 60

# light_sequence = np.arange(0,1,step=.005)

# Run for the duration of the light sequence:
F_s = []
for c in range(100):
    print(c)
    cell = dcc.simulations.CcaSR_Autoactivation()
    cell.set_light_events(light_sequence)
    series = cell.run(len(light_sequence)*5)
    F_s.append(np.asarray([state["F"] for state in series]))

F = np.array(F_s)
# Plot results:

x = [t/12. for t in range(len(series))]
# F = np.asarray([state["F"] for state in series])
# plt.plot(F/np.max(F), label="GFP")
# plt.plot(x, [state["R"] for state in series], label="LacI")

dcc.utilities.OptoPlotBackground(light_sequence, ymax=800, x=x)
# dcc.utilities.plotq(F, x=x)
for f in F:
    plt.plot(x,f,color="b",alpha=.2,lw=.5)
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.legend()
plt.show()

hist = []
for t in range(F.shape[1]):
    hist.append(np.histogram(F[:,t], bins=np.linspace(0,600,30))[0])
hist = np.array(hist)
hist = np.log(hist)
plt.imshow(hist.transpose())

    
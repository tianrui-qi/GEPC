# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:57:55 2023

@author: jeanbaptiste
"""

import matplotlib.pyplot as plt

import deepcellcontrol as dcc

# Optogenetic input sequence
# (2 hours no light, 10 hours light. 1 hour = 12 timepoints every 5 minutes)
light_sequence = [0]*24 + [1]*120

# Create cell:
cell = dcc.simulations.CcaSR_Inverter()

# Add light sequence events:
cell.set_light_events(light_sequence)

# Run for the duration of the light sequence:
series = cell.run(len(light_sequence)*5)

# Plot results:
x = [t/12. for t in range(len(series))]
plt.plot(x, [state["F"] for state in series], label="GFP")
plt.plot(x, [state["R"] for state in series], label="LacI")
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.legend()
plt.show()
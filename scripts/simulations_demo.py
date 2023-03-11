# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:57:55 2023

@author: jeanbaptiste
"""
import copy

import matplotlib.pyplot as plt
import numpy as np

import deepcellcontrol as dcc

# Optogenetic input sequence
light_sequence = [0]*24 + [1]*120 + [0, 1]*120 + [0] * 120 + [0, 1] * 120

# Cell we will re-use through the script (but of course they can be re-instanciated)
refcell = dcc.simulations.CcaSR_Autoactivation()
refcell.species["E"] = refcell.params['h1'] / refcell.params['h2']
refcell.params['h2'] = 1e-30
refcell.params['h1'] = 1e-30
refcell.set_light_events(light_sequence)

# For plots:
x = [t/12. for t in range(len(light_sequence)+1)]

#%% Deterministic run:

cell = copy.deepcopy(refcell)
series = cell.run(len(light_sequence)*5, solver="ode")

dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
plt.plot(x, [state["F"] for state in series],color="b")
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.title("Deterministic")

#%% Original SSA implementation run:

cell = copy.deepcopy(refcell)
series_list = cell.run(len(light_sequence)*5, solver="original", realizations=100)

F = [[state["F"] for state in series] for series in series_list]
dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
for series in series_list:
    plt.plot(x, [state["F"] for state in series], color="b", alpha=.2, lw=.5)
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.title("Original SSA")

#%% GillesPy2 implementation run:

cell = copy.deepcopy(refcell)
series_list = cell.run(len(light_sequence)*5, solver="gp2", realizations=100)

F = [[state["F"] for state in series] for series in series_list]
dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
for series in series_list:
    plt.plot(x, [state["F"] for state in series], color="b", alpha=.2, lw=.5)
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.title("GillesPy2 SSA")

#%% Bifurcation:    

solve_hours = 72

cell_on = dcc.simulations.CcaSR_Autoactivation()
cell_on.species["E"] = cell_on.params['h1'] / cell_on.params['h2']
cell_on.params['h2'] = 1e-30
cell_on.params['h1'] = 1e-30
cell_off = copy.deepcopy(cell_on)

cell_on.set_light_events([1]*solve_hours*12)
series = cell_on.run(solve_hours*60, solver="ode")
cell_on.time = 0

cell_off.set_light_events([0]*solve_hours*12)
_ = cell_off.run(solve_hours*60, solver="ode")
cell_off.time = 0

u_values = np.linspace(0,1,30)
start_on, start_off = [], []
for u in u_values:
    print(u)
    
    sequence = [u]*solve_hours*12
    
    on = copy.deepcopy(cell_on)
    on.set_light_events(sequence)
    start_on += [on.run(solve_hours*60, solver="ode")[-1]["F"]]
    
    off = copy.deepcopy(cell_off)
    off.set_light_events(sequence)
    start_off += [off.run(solve_hours*60, solver="ode")[-1]["F"]]

plt.plot(u_values, start_on, '-o', label="start on")
plt.plot(u_values, start_off, '-o', label="start off")
plt.xlabel("U")
plt.ylabel("F (#proteins)")


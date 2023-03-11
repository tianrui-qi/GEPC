# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:57:55 2023

@author: jeanbaptiste
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import deepcellcontrol as dcc

# Optogenetic input sequence
light_sequence = [0]*24 + [1]*120 + [0, 1]*120 + [0] * 120 + [0, 1] * 120

# Cell we will re-use through the script (but of course they can be re-instanciated)
refcell = dcc.simulations.CcaSR_Autoactivation()
refcell.species["E"] = refcell.params['h1'] / refcell.params['h2']
refcell.params['h2'] = 1e-30 # if ==0 then propensities throw div by 0 error
refcell.params['h1'] = 1e-30
refcell.set_light_events(light_sequence)

# For plots:
x = [t/12. for t in range(len(light_sequence)+1)]

#%% Deterministic run:

cell = refcell.copy()
series = cell.run(len(light_sequence)*5, solver="ode")

dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
plt.plot(x, [state["F"] for state in series],color="b")
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.ylim(0,100)
plt.title("Deterministic")

#%% Original SSA implementation run:

cell = refcell.copy()
series_list = cell.run(len(light_sequence)*5, solver="original", realizations=100)

F = [[state["F"] for state in series] for series in series_list]
dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
for series in series_list:
    plt.plot(x, [state["F"] for state in series], color="b", alpha=.2, lw=.5)
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.ylim(0,100)
plt.title("Original SSA")

#%% GillesPy2 implementation run:

cell = refcell.copy()
series_list = cell.run(len(light_sequence)*5, solver="gp2", realizations=100)

F = [[state["F"] for state in series] for series in series_list]
dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
for series in series_list:
    plt.plot(x, [state["F"] for state in series], color="b", alpha=.2, lw=.5)
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.ylim(0,100)
plt.title("GillesPy2 SSA")

#%% Bifurcation:    

solve_hours = 72

cell_on = dcc.simulations.CcaSR_Autoactivation()
cell_on.species["E"] = cell_on.params['h1'] / cell_on.params['h2']
cell_on.params['h2'] = 1e-30
cell_on.params['h1'] = 1e-30
cell_off = cell_on.copy()

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
    
    on = cell_on.copy()
    on.set_light_events(sequence)
    start_on += [on.run(solve_hours*60, solver="ode")[-1]["F"]]
    
    off = cell_off.copy()
    off.set_light_events(sequence)
    start_off += [off.run(solve_hours*60, solver="ode")[-1]["F"]]

plt.plot(u_values, start_on, '-o', label="start on")
plt.plot(u_values, start_off, '-o', label="start off")
plt.xlabel("U")
plt.ylabel("F (#proteins)")

#%% Step-by-step computation (for example as in feedback control)

solver = "original"
num_cells = 100

refcell = dcc.simulations.CcaSR_Autoactivation()
refcell.species["E"] = refcell.params['h1'] / refcell.params['h2']
refcell.params['h2'] = 1e-30
refcell.params['h1'] = 1e-30

cells = [refcell.copy() for _ in range(num_cells)]
fluorescence = [[refcell.species["F"]] for _ in range(len(cells))]

for l, light in enumerate(light_sequence):
    for cell, cell_fluo in zip(cells, fluorescence):
        cell.set_light_events([light])
        series = cell.run((l+1)*5, solver = solver)
        cell_fluo.append(series[-1]["F"])

dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
plt.plot(x, np.transpose(fluorescence), color="b", alpha=.2, lw=2/(np.log(num_cells)+1))
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.ylim(0,100)
plt.title("Step-by-step computation")

#%% Serialize and load demo:

stop_at = 20*60 # Where to cut execution (in minutes)

cell = dcc.simulations.CcaSR_Autoactivation()
cell.species["E"] = cell.params['h1'] / cell.params['h2']
cell.params['h2'] = 1e-30
cell.params['h1'] = 1e-30
cell.set_light_events(light_sequence)

# Run for a little bit:
series = cell.run(stop_at)

# Get a dict of the ref_cell:
serialized = cell.serialize()

# Save to disk (could also use pickle, hdf5...)
with open("assets/serialized_cell.json", "w") as f:
    json.dump(serialized, f, indent=2)

# Delete data:
del serialized, cell

# Re-Load from disk:
cell = dcc.simulations.CcaSR_Autoactivation()
with open("assets/serialized_cell.json", "r") as f:
    serialized = json.load(f)
cell.load(serialized)

# Finish running:
series += cell.run(len(light_sequence)*5)[1:] # We already have the start value

dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
plt.plot(x, [state["F"] for state in series],color="b")
plt.plot([stop_at/60, stop_at/60], [0, 100], 'k--')
plt.xlabel("time (hours)")
plt.ylabel("proteins (#)")
plt.ylim(0,100)
plt.title("Reloaded")

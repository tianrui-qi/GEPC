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
# light_sequence = [0]*24 + [1]*180 + [0, 1]*180 + [0] * 180 + [0, 1] * 180
light_sequence = [0]*4*12 + [1]*4*12

# Cell we will re-use through the script (but of course they can be re-instanciated)
# refcell = dcc.simulations.CcaSR_gillespie_full()
# refcell.params['h2'] = 1e-30 # if ==0 then propensities throw div by 0 error
# refcell.params['h1'] = 1e-30
# refcell.set_light_events(light_sequence)

# For plots:
x = [t/12. for t in range(len(light_sequence)+1)]

#%% Deterministic run:


new_params_list = [{'tau': 12},
                    ]

for new_params in new_params_list:

    cell = dcc.simulations.CcaSR_gillespie_simple_noE()
        
    # print(new_params)
    # cell.update_params(new_params)
    
    # Simulate
    cell.set_light_events(light_sequence)
    # cell.species["E"] = cell.params['h1'] / cell.params['h2']
    
    series = cell.run(len(light_sequence)*5, solver="ode")
    
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)
    states = ['F']#,'E','I','J']
    colors = ['b']#,'k','#FFA500','g']
    for s, state_s in enumerate(states):
        plt.plot(x, [state[state_s] for state in series],
                 color=colors[s], 
                 label=state_s)
    plt.legend()
    plt.xlabel("time (hours)")
    plt.ylabel("proteins (#)")
    plt.ylim(0,150)
    # plt.title(f"Deterministic, h1={new_params['h1']:.2e}, h2={new_params['h2']:.2e}")
    plt.show()

#%% Original SSA implementation run:
    
new_params_list = [{'tau': 0},
                    ]
    
for new_params in new_params_list:
    cell = dcc.simulations.CcaSR_gillespie_full()
    # cell.update_params(new_params)
    
    
    cell.set_light_events(light_sequence)     

    series_list = cell.run(len(light_sequence)*5, 
                           solver="original", 
                           realizations=20)
    
    species = ['Sp','Rp','Sp_p','SR','F']
    colors = ['tab:purple','g','k', 'r', 'b']

    dcc.utilities.OptoPlotBackground(light_sequence, ymax=400, x=x)

    for s, series in enumerate(series_list):
    
        
        for i, specie in enumerate(species):
            
            plt.plot(x, [state[specie] for state in series],
                          color=colors[i], label=specie,
                          alpha=.5, lw=.5)
            
    plt.xlabel("time (hours)")
    plt.ylabel("proteins (#)")
    plt.ylim(0,300)
    # plt.title(f"Original SSA, K_I={new_params['K_I']:.2e}")
    plt.show()

# cell = refcell.copy()
# series_list = cell.run(len(light_sequence)*5, 
#                        solver="original", 
#                        realizations=100)

# F = [[state["F"] for state in series] for series in series_list]
# dcc.utilities.OptoPlotBackground(light_sequence, ymax=100, x=x)
# for series in series_list:
#     plt.plot(x, [state["F"] for state in series], color="b", alpha=.2, lw=.5)
# plt.xlabel("time (hours)")
# plt.ylabel("proteins (#)")
# plt.ylim(0,100)
# plt.title("Original SSA")
# plt.show()



#%% Random stimulations

# new_params = new_params_list[3]
n_cells = 13
stims_all = dcc.utilities.random_stimulations(
                        timepoints=36*12,
                        nostim_timepoints=3*12,
                        total_simulations=n_cells)
x = [t/12. for t in range(np.shape(stims_all)[1]+1)]


for i in range(n_cells):
    plt.figure()
    cell = dcc.simulations.CcaSR_gillespie()
    # cell.update_params(new_params)
    
    light_sequence = stims_all[i]

    cell.set_light_events(light_sequence) 

    series = cell.run(len(light_sequence)*5, 
                           solver="ode", 
                           realizations=1)

    
    dcc.utilities.OptoPlotBackground(light_sequence, ymax=150, x=x)

    plt.plot(x, [state['F'] for state in series],
                 color='b')
    plt.xlabel("time (hours)")
    plt.ylabel("proteins (#)")
    plt.ylim(0,150)
    # plt.title(f"Original SSA, K_I={new_params['K_I']:.2e}")
    plt.show()

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
plt.show()

#%% Bifurcation:    

solve_hours = 120

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

u_values = np.linspace(0,1,50)
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
plt.xlabel("Average light input")
plt.ylabel("F (#proteins)")
plt.legend()
plt.savefig("D:/deepcellcontrol/assets/autoactivation/SI_figs/bifurcation.svg", dpi=300)
plt.show()

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
plt.show()

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
plt.show()

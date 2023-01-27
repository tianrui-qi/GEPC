#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions and objects to simulate the behavior of our 
optogenetic systems.

Note: This code was used for preliminary evaluations but has not been
used in a while. We will update it soon as part of a theoretical study.

@author: jeanbaptiste
"""
import copy
import itertools
from multiprocessing import Pool

import numpy as np

from . import utilities as utils
from . import data

SAMPLING = 5

class CcaSR_gillespie():
    
    def __init__(self):
        
        # All parameters based on Chait et al. except for 'h1' and 'h2'.
        # They are lower to reflect the slow changes we see in responsiveness,
        # h2 is further divided by 2 to reflect the higher overal 
        # responsiveness in our data.
        # We also treat the light-activation dynamics (species H) as stochastic
        # instead of the 
        self.params = {
            'h1':0.0710/25,
            'h2':0.0303/50,
            'c2':0.0631,
            'a':0.2827,
            'b':0.0104,
            's':0.9958,
            'nh':3.6655,
            'K':0.4851,
            'tau':12
            }
        self.species = {
            'U':0,
            'H':0.,
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])),
            'F':0
            }
        self.sampling = SAMPLING # Sampling interval
        self.events = [] # External events
        self.past_events = []
        self.time = 0 # time is NOT reset when run() is called! (see run method below)
        
    def run(self, stoptime):
        '''
        Run Gillespie simulation of the system until stopping point.

        Parameters
        ----------
        stoptime : float
            Time-point at which the simulation should be stopped.

        Returns
        -------
        time_series : list of dicts
            List of copies of the species property at sampling time points for 
            the duration of the run.

        '''
        
        # Intialize:
        starttime = self.time
        time_series = []
        newspecies = self.species.copy() # Initialize for 1st while loop run
        self.events.sort(key=lambda elem: elem['time']) # Sort events in ascending time before runnning just in case
        
        while self.time<stoptime:
            
            # Update species values:
            self.species = newspecies
            
            # Run reaction:
            timestep, newspecies = self.run_nextreaction()
                
            # Update time:
            self.time += timestep
            
            # If sampling time, append current species values to time-series:
            while (self.time-starttime) >= len(time_series)*self.sampling:
                time_series.append(self.species.copy())
                
        # Set to stop time:
        self.time = stoptime
        
        # Check if any "past events" need to be added back to the events list:
        for pe in self.past_events[::-1]:
            if pe['time']>=self.time:
                self.events.insert(0,pe)
                self.past_events.remove(pe)

        # Issue sampled timeseries
        time_series = time_series[:int((stoptime-starttime)/self.sampling)+1]
            
        return time_series
        
    def set_light_events(self,light_sequence):
        '''
        Set future light input series. This will append events to the event property
        to be triggered at future timepoints. The input delay (tau) from the 
        Chait et al. model is applied.

        Parameters
        ----------
        light_sequence : 1D array-like
            A sequence of 0 and 1s representing red or green optogenetic inputs
            during sampling intervals. Element 0 will be applied as optogenetic
            input at time+0*sampling+tau.

        Returns
        -------
        None.

        '''
        
        for i, u in enumerate(light_sequence):
            self.events.append({'time': self.time+i*self.sampling+self.params['tau'],
                                'set': (['U',u],)})
        
    def run_nextreaction(self):
        '''
        Simulate next reaction (or apply next event, whichever comes first)

        Returns
        -------
        timestep : float
            Incremental timestep to the next reaction/event.
        newspecies : dict
            Updated version of the species dictionary once reaction/event is 
            applied.

        '''
        
        # Calculate propensities (See SI of Chait et al.):
        propensities = []
        # "Extrinsic" creation:
        propensities.append((self.params['h1'], 
                             (['E',1],))) # (propensity, reaction stoichiometry)
        # "Extrinsic" dilution:
        propensities.append((self.params['h2']*self.species['E'], 
                             (['E',-1],)))
        # CcaSR activation:
        propensities.append((self.species['U'], 
                              (['H',1],)))
        # CcaSR deactivation/dilution:
        propensities.append((self.params['c2']*self.species['H'], 
                              (['H',-1],)))
        # GFP creation:
        propensities.append((self.params['a']*self.species['E']*\
                            ((self.species['H']*self.params['c2'])**self.params['nh'])/\
                            (self.params['K']+((self.species['H']*self.params['c2'])**self.params['nh'])),
                            (['F',1],)))
        # GFP dilution:
        propensities.append((self.params['b']*self.species['F'], 
                             (['F',-1],)))
        
        # Cumulative sum:
        cs = np.cumsum([x[0] for x in propensities])
            
        # Update timestep to next reaction:
        timestep = -np.log(np.random.random()) / cs[-1]
        
        # Copy current species state:
        newspecies = self.species.copy()
        
        # Apply either next event or reaction:
        if len(self.events)>0 and self.time + timestep > self.events[0]['time']: # If we cross the timepoint of the next event
            # Pop out the event:
            e = self.events.pop(0)
            self.past_events.append(e)
            # Apply event:
            if 'set' in e: # event is a 'set' event
                for s in e['set']:
                    newspecies[s[0]] = s[1]
            elif 'react' in e: # event is a 'reaction' event
                for r in e['react']:
                    newspecies[r[0]] = self.species[r[0]] + r[1]
            # Reset timestep to just reach the event:
            timestep = e['time']-self.time
            
        else: # Apply reaction
            # Get random number and select reaction:
            reaction = np.argmax(cs >= np.random.random()*cs[-1])
            # Apply selected reaction stoichiometry:
            for r in propensities[reaction][1]:
                newspecies[r[0]] = self.species[r[0]] + r[1]
        
        # Return:
        return (timestep,newspecies)

def camera_sim(fluo, camera_mult=40,camera_max=4095,camera_offset=100,noise_perc=5):
    '''
    Simulate camera dynamic range and imaging noise.

    Parameters
    ----------
    fluo : array of floats
        "True" fluorescence levels.
    camera_mult : float, optional
        Multiplication factor to get roughly in the same dynamic range as what 
        we get with our microscope. Needs to be estimated for new simulation 
        models.
        The default is 40.
    camera_max : float, optional
        Camera dynamic range. Fluorescence values will be clipped at this level
        The default is 4095.
    camera_offset : float, optional
        Camera offset value to avoid crossing 0.
        The default is 100.
    noise_perc: float, optional
        Gaussian noise sigma, expressed as %.
        The default is 5

    Returns
    -------
    array of floats
        "Measured" fluorescence levels.

    '''
    return np.clip(
                np.multiply(
                    (fluo*camera_mult+camera_offset),
                    np.random.normal(loc=1.0,scale=noise_perc/100,size=fluo.shape)
                    ),
                0,
                camera_max
            )

def training_set(stims, sampling=SAMPLING):
    """
    Generate training set from Gillespie model and pre-determined stimulations

    Parameters
    ----------
    stims : 2D array of bool.
        Stimulations to apply to the cell.
    sampling : int, optional
        Period between measurements, in (simulated) minutes. The default is 5.

    Returns
    -------
    fluo : 2D array of float
        Generated fluorescence.

    """
    
    # Run Gillespie simulations: (not using multiprocessing here because it's relatively fast)
    fluo = []
    for l in range(stims.shape[0]):
        cell = CcaSR_gillespie() # Instantiate new "cell"
        cell.set_light_events(stims[l]) # Set future light events
        ts = cell.run(stims.shape[1]*sampling) # Run until the end
        fluo.append([x['F'] for x in ts[1:]]) # Append fluorescence to list (skip first timepoint before any stim)
        print('%d/%d cells simulated'% (l+1,stims.shape[0]))
        
    fluo = np.array(fluo)
    
    # Simulate camera/measurement noise:
    fluo = camera_sim(fluo)
    
    return fluo

# Define simulation function (Needs to be top level to use in multiprocessing pool):
def _evaluation(stims, cut_off, future_realizations, sampling):
    cell = CcaSR_gillespie() # Instantiate new "cell"
    cell.set_light_events(stims) # Set future light events
    ts = cell.run(cut_off*sampling) # Run until cut off between "Past" and "Future"
    Past = np.array([x['F'] for x in ts[1:]]) # Append fluorescence time-series to the "Past" of the cell
    Past = camera_sim(Past)
    # Run thousands of potential futures for the cell:
    Future = np.empty((future_realizations,stims.shape[0]-cut_off)) # Allocate "empty" future
    for s in range(future_realizations):
        clone = copy.deepcopy(cell) # Copy "Past" cell ("Present" species state and time are also copied)
        ts = clone.run(stims.shape[0]*sampling) # Run this potential "future"
        Future[s,:] = [x['F'] for x in ts[1:]] # Save fluorescence time-series to "future" array (Don't need first time-point as it's always the same)
    Future = camera_sim(np.array(Future))

    return Past, Future

def evaluation_set(
        stims, cut_off, future_realizations=5000, sampling=SAMPLING, num_workers=8
        ):
    
    # Run _evaluation function in parallel:
    with Pool(num_workers) as pool:
        res = pool.starmap(
            _evaluation, 
            zip(
                stims, 
                itertools.repeat(cut_off),
                itertools.repeat(future_realizations), 
                itertools.repeat(sampling)
                )
            )
    
    return res

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

SAMPLING = 5

class Reaction():
    """
    Class to implement reaction propensity evaluation.
    """
    
    def __init__(self, propensity, stoichiometry):
        """
        Init/Instanciate reaction.

        Parameters
        ----------
        propensity : str
            String describing propensity computation. e.g. "a*E*S" for an 
            enzymatic reaction E + S --> C with rate a.
        stoichiometry : Dict
            Description of the stoichiometry of the reaction. 
            e.g. {"E": -1, "S": -1, "C": 1}

        Returns
        -------
        None.

        """
        
        self.propensity = propensity
        self._compiled = compile(propensity, "_", "eval")
        self.stoichiometry = stoichiometry
        
    def update(self, params, species):
        """
        Evaluate and return propensity based on current parameters and soecies
        levels.

        Parameters
        ----------
        params : Dict
            Dictionary of reaction parameters.
        species : Dict
            Dictionary of species numbers.

        Returns
        -------
        float
            Evaluated propensity of the reaction.

        """
        
        return eval(self._compiled, {}, {**params, **species})
        

class CcaSR_gillespie():
    """
    Base class that implements an optogenetics-friendly Gillespie algorithm and
    the simple CcaSR-GFP circuit that we used in our feedback control 
    experiements.
    
                            "Responsiveness" (E)
                                    |
                                    v
    Light Input (U) ---> CcaSR (H) ---> GFP (F)
    
    
    """
    
    def __init__(self):
        
        # All parameters based on Chait et al. except for 'h1' and 'h2'.
        # They are lower to reflect the slow changes we see in responsiveness,
        # h2 is further divided by 2 to reflect the higher overal 
        # responsiveness in our data.
        # We also treat the light-activation dynamics (species H) as stochastic
        # instead of the 
        self.params = {
            'h1':0.0710/25, # "Extrinsic responsiveness" generation rate
            'h2':0.0303/50, # "Extrinsic responsiveness" dilution rate
            'c2':0.0631, # Hill normalization parameter
            'a':0.2827, # PcpcG2 promoter rate
            'b':0.0104, # Proteins dilution rate
            's':0.9958, # Not used in the end
            'nh':3.6655, # Hill coefficient
            'K':0.4851, # Hill threshold (sort of)
            'tau':12 # Response delay
            }
        "Parameters used in the propensity calculations"
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])), # "Extrinsic noise / responsiveness"
            'F':0 # GFP
            }
        self.reactions = (
            Reaction('h1', {'E': 1}), # "Extrinsic" creation
            Reaction('h2*E', {'E': -1}), # "Extrinsic" dilution
            Reaction('U', {'H': 1}), # CcaSR activation
            Reaction('c2*H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('a*E*((c2*H)**nh)/(K+(c2*H)**nh)', {'F': 1}), # GFP creation
            Reaction('b*F', {'F': -1}), # GFP dilution
            )
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
            Time-point at which the simulation should be stopped, in minutes.

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
        
        # Calculate propensities:
        propensities = [r.update(self.params, self.species) for r in self.reactions]
        
        # Cumulative sum:
        cs = np.cumsum(propensities)
            
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
            selected = np.argmax(cs >= np.random.random()*cs[-1])
            # Apply selected reaction stoichiometry:
            for s, r in self.reactions[selected].stoichiometry.items():
                newspecies[s] = self.species[s] + r
        
        # Return:
        return (timestep,newspecies)

class CcaSR_gillespie_simple(CcaSR_gillespie):
    """
    A variant of the base class with simpler parameter choices
    
                            "Responsiveness" (E)
                                    | 
                                    v  
    Light Input (U) ---> CcaSR (H) ---> LGFP (F)
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'eta': 1, # production rate of H per unit of U
            'nu': 0.0303, # dilution of all proteins (i.e.H, F); matches Chait c2 or b
            'rho': 0.0710/25, # production of extrinsic noise; matches Chait h1
            'nu_E': 0.0303/30, # dilution of extrinsic noise; matches Chait h2
            'a': 0.2827, # production rate of F per unit of E; matches Chait a
            'K_H': 13, # concentration of H for 50% production of F; matches Chait K/c2
            'nH': 3.6655, # cooperativity for H activation of F; matches Chait nh
            'tau': 12, # delay between light change and effect on U
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['rho'] / self.params['nu_E'])), # "Extrinsic noise / responsiveness"
            'F':0, # GFP
            }
        self.reactions = (
            Reaction('rho', {'E': 1}), # "Extrinsic" creation
            Reaction('nu_E*E', {'E': -1}), # "Extrinsic" dilution
            Reaction('eta*U', {'H': 1}), # CcaSR activation
            Reaction('nu*H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('a*E * (H**nH)/(K_H**nH+H**nH)', {'F': 1}), # GFP creation
            Reaction('nu*F', {'F': -1}), # GFP dilution
            )        

class CcaSR_Inverter(CcaSR_gillespie):
    """
    A simple inverter circuit where LacI is downstream of PcpcG2 and then
    represses the expression of GFP.
    
                                   "Responsiveness" (E)
                                    |             |
                                    v             V
    Light Input (U) ---> CcaSR (H) ---> LacI (R) ---| GFP (F)
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'h1':0.0710/25, # "Extrinsic responsiveness" generation rate
            'h2':0.0303/50, # "Extrinsic responsiveness" dilution rate
            'c2':0.0631, # Hill normalization parameter
            'a':0.2827, # PcpcG2 promoter rate
            'b':0.0104, # Proteins dilution rate
            's':0.9958, # Not used in the end
            'nh':3.6655, # Hill coefficient
            'K':0.4851, # Hill threshold (sort of)
            'tau':12, # Response delay
            'g':.05, # pLac promoter rate
            'Kr':.25, # pLac-LacI Hill threshold
            'c3':.03, # pLac-LacI Hill normalization parameter
            'nr':2 # pLac-LacI Hill coefficient
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])), # "Extrinsic noise / responsiveness"
            'F':0, # GFP
            'R':0, # LacI
            }
        self.reactions = (
            Reaction('h1', {'E': 1}), # "Extrinsic" creation
            Reaction('h2*E', {'E': -1}), # "Extrinsic" dilution
            Reaction('U', {'H': 1}), # CcaSR activation
            Reaction('c2*H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('a*E*((c2*H)**nh)/(K+(c2*H)**nh)', {'R': 1}), # LacI creation
            Reaction('b*R', {'R': -1}), # LacI dilution
            Reaction('g*E/(Kr+(c3*R)**nr)', {'F': 1}), # GFP creation
            Reaction('b*F', {'F': -1}), # GFP dilution
            )
        
class CcaSR_Cascade(CcaSR_gillespie):
    """
    A simple cascade where CcaSR activates an intermediate that activates F
    
                                       "Responsiveness" (E)
                                    |                     |
                                    v                     V
    Light Input (U) ---> CcaSR (H) ---> Intermediate (I) ---> GFP (F)
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'alpha': 5, # ratio of H production (per unit light) to dilution
            'beta': 5, # ratio of E production (per amount of H) to dilution
            'gamma': 5, # ratio of F production to dilution
            'kappa_i': 5, # ratio of I to H activation strengths, raised to power of I cooperativity
            'nh': 2, # cooperativity of I activation by H
            'ni': 2, # cooperativity of F activation by I
            'tau':12 # Response delay
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])), # "Extrinsic noise / responsiveness"
            'I': 0, # Intermediate
            'F':0, # GFP
            }
        self.reactions = (
            Reaction('beta', {'E': 1}), # "Extrinsic" creation
            Reaction('E', {'E': -1}), # "Extrinsic" dilution
            Reaction('alpha*U', {'H': 1}), # CcaSR activation
            Reaction('H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('gamma*E * (H**nh)/(1+H**nh)', {'I': 1}), # Intermediate creation
            Reaction('I', {'I': -1}), # Intermediate dilution
            Reaction('gamma*E * (I**ni)/(kappa_i+I**ni)', {'F': 1}), # GFP creation
            Reaction('F', {'F': -1}), # GFP dilution
            )

class CcaSR_Autoactivation(CcaSR_gillespie):
    """
    A circuit where F can additionally activate itself
    
                            "Responsiveness" (E)
                                    |         |
                                    |         V                  
                                    |         __
                                    v        V  |
    Light Input (U) ---> CcaSR (H) ---> GFP (F)--
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'eta': 1, # production of H per light
            'nu': 0.03, # dilution of all proteins (midpoint of c2, b, h2 from Chait)
            'K_H': 14, # dissociation constant, H activation of F
            'rho': 0.42, # production of E (used to keep nondim. beta=1)
            'a': 0.038, # production of F per unit E (tunes hysteresis)
            'nh': 3.6, # cooperativity of F activation by H (from Chait)
            'nf': 3.6, # cooperativity of F activation by F (to match nh)
            'K_F': 5.6, # for nondim. kappa = 10
            'tau':12 # Response delay
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['rho'] / self.params['nu'])), # "Extrinsic noise / responsiveness"
            'F':0, # GFP
            }
        self.reactions = (
            Reaction('rho', {'E': 1}), # "Extrinsic" creation
            Reaction('nu*E', {'E': -1}), # "Extrinsic" dilution
            Reaction('eta*U', {'H': 1}), # CcaSR activation
            Reaction('nu*H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('a*E/2 * (H**nh)/(K_H**nh+H**nh)', {'F': 1}), # GFP creation by H
            Reaction('a*E/2 * (F**nf)/(K_F**nf+F**nf)', {'F': 1}), # GFP creation by itself
            Reaction('nu*F', {'F': -1}), # GFP dilution
            )

class CcaSR_FeedforwardPositive(CcaSR_gillespie):
    """
    Positive feedforward control of GFP
    
                                       "Responsiveness" (E) affects all reactions, though not all drawn
                                    |                     |
                                    v                     V
    Light Input (U) ---> CcaSR (H) ---> Intermediate (I) ---> GFP (F)
                                            |                  ^
                                            V                  |
                                             Intermediate (J)
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'alpha': 5, # ratio of H production (per unit light) to dilution
            'beta': 5, # ratio of E production (per amount of H) to dilution
            'gamma': 5, # ratio of F production to dilution
            'kappa_i': 5, # ratio of I to H activation strengths, raised to power of I cooperativity
            'kappa_j': 5, # ratio of J to H activation strengths, raised to power of J cooperativity
            'nh': 2, # cooperativity of I activation by H
            'ni': 2, # cooperativity of F, J activation by I
            'nj': 2, # cooperativity of F activation by J
            'tau':12 # Response delay
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])), # "Extrinsic noise / responsiveness"
            'I': 0, # Intermediate I
            'J': 0, # Intermediate J
            'F':0, # GFP
            }
        self.reactions = (
            Reaction('beta', {'E': 1}), # "Extrinsic" creation
            Reaction('E', {'E': -1}), # "Extrinsic" dilution
            Reaction('alpha*U', {'H': 1}), # CcaSR activation
            Reaction('H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('gamma*E * (H**nh)/(1+H**nh)', {'I': 1}), # Intermediate I creation
            Reaction('I', {'I': -1}), # Intermediate I dilution            
            Reaction('gamma*E * (I**ni)/(kappa_i+I**ni)', {'J': 1}), # Intermediate J creation
            Reaction('J', {'J': -1}), # Intermediate J dilution
            Reaction('gamma*E/2 * (I**ni)/(kappa_i+I**ni)', {'F': 1}), # GFP creation by I
            Reaction('gamma*E/2 * (J**nj)/(kappa_j+J**nj)', {'F': 1}), # GFP creation by J
            Reaction('F', {'F': -1}), # GFP dilution
            )

class CcaSR_FeedforwardNegative(CcaSR_gillespie):
    """
    Negative feedforward control of GFP
    
                                       "Responsiveness" (E)
                                    |                     |
                                    v                     V
    Light Input (U) ---> CcaSR (H) ---> Intermediate (I) ---> GFP (F)
                                            |                  _
                                            V                  |
                                             Intermediate (J)
    
    This class inherits from the `CcaSR_FeedforwardPositive` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.reactions = (
            Reaction('beta', {'E': 1}), # "Extrinsic" creation
            Reaction('E', {'E': -1}), # "Extrinsic" dilution
            Reaction('alpha*U', {'H': 1}), # CcaSR activation
            Reaction('H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('gamma*E * (H**nh)/(1+H**nh)', {'I': 1}), # Intermediate I creation
            Reaction('I', {'I': -1}), # Intermediate I dilution            
            Reaction('gamma*E * (I**ni)/(kappa_i+I**ni)', {'J': 1}), # Intermediate J creation
            Reaction('J', {'J': -1}), # Intermediate J dilution
            Reaction('gamma*E/2 * (I**ni)/(kappa_i+I**ni)', {'F': 1}), # GFP creation by I
            Reaction('gamma*E/2 * (kappa_j)/(kappa_j+J**nj)', {'F': 1}), # GFP creation by J
            Reaction('F', {'F': -1}), # GFP dilution
            )

class CcaSR_FeedbackPositive(CcaSR_gillespie):
    """
    Positive feedback control of GFP
    
                                       "Responsiveness" (E)
                                    |                     |
                                    v                     V
    Light Input (U) ---> CcaSR (H) ---> Intermediate (I) ---> GFP (F)
                                            ^                  |
                                            |                  V
                                             Intermediate (J)
    
    This class inherits from the `CcaSR_gillespie` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.params = {
            'alpha': 5, # ratio of H production (per unit light) to dilution
            'beta': 5, # ratio of E production (per amount of H) to dilution
            'gamma': 5, # ratio of F production to dilution
            'kappa_i': 5, # ratio of I to H activation strengths, raised to power of I cooperativity
            'kappa_j': 5, # ratio of J to H activation strengths, raised to power of J cooperativity
            'kappa_f': 5, # ratio of F to H activation strengths, raised to power of F cooperativity
            'nh': 2, # cooperativity of I activation by H
            'ni': 2, # cooperativity of F activation by I
            'nj': 2, # cooperativity of I activation by J
            'nf': 2, # cooperativity of J activation by F
            'tau':12 # Response delay
            }
        self.species = {
            'U':0, # Optogenetic input
            'H':0., # CcaS-CcaR
            'E':round(np.random.poisson(self.params['h1']/self.params['h2'])), # "Extrinsic noise / responsiveness"
            'I': 0, # Intermediate I
            'J': 0, # Intermediate J
            'F':0, # GFP
            }
        self.reactions = (
            Reaction('beta', {'E': 1}), # "Extrinsic" creation
            Reaction('E', {'E': -1}), # "Extrinsic" dilution
            Reaction('alpha*U', {'H': 1}), # CcaSR activation
            Reaction('H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('gamma*E/2 * (H**nh)/(1+H**nh)', {'I': 1}), # Intermediate I creation
            Reaction('gamma*E/2 * (J**nj)/(1+J**nj)', {'I': 1}), # Intermediate I creation
            Reaction('I', {'I': -1}), # Intermediate I dilution            
            Reaction('gamma*E * (F**nf)/(kappa_f+F**nf)', {'J': 1}), # Intermediate J creation
            Reaction('J', {'J': -1}), # Intermediate J dilution
            Reaction('gamma*E * (I**ni)/(kappa_i+I**ni)', {'F': 1}), # GFP creation by I
            Reaction('F', {'F': -1}), # GFP dilution
            )

class CcaSR_FeedbackNegative(CcaSR_gillespie):
    """
    Negative feedback control of GFP
    
                                       "Responsiveness" (E)
                                    |                     |
                                    v                     V
    Light Input (U) ---> CcaSR (H) ---> Intermediate (I) ---> GFP (F)
                                            ^                  |
                                            |                  _
                                             Intermediate (J)
    
    This class inherits from the `CcaSR_FeedbackPositive` class.
    """
    
    def __init__(self):
        # Run parent class init:
        super().__init__()
        
        # Alter the reactions network:
        self.reactions = (
            Reaction('beta', {'E': 1}), # "Extrinsic" creation
            Reaction('E', {'E': -1}), # "Extrinsic" dilution
            Reaction('alpha*U', {'H': 1}), # CcaSR activation
            Reaction('H', {'H': -1}), # CcaSR deactivation/dilution
            Reaction('gamma*E/2 * (H**nh)/(1+H**nh)', {'I': 1}), # Intermediate I creation
            Reaction('gamma*E/2 * (J**nj)/(1+J**nj)', {'I': 1}), # Intermediate I creation
            Reaction('I', {'I': -1}), # Intermediate I dilution            
            Reaction('gamma*E * (kappa_f)/(kappa_f+F**nf)', {'J': 1}), # Intermediate J creation
            Reaction('J', {'J': -1}), # Intermediate J dilution
            Reaction('gamma*E * (I**ni)/(kappa_i+I**ni)', {'F': 1}), # GFP creation by I
            Reaction('F', {'F': -1}), # GFP dilution
            )

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

def training_set(stims, sampling=SAMPLING, cell_class = CcaSR_gillespie):
    """
    Generate training set from Gillespie model and pre-determined stimulations

    Parameters
    ----------
    stims : 2D array of bool.
        Stimulations to apply to the cells. Dimensions are (cells, time).
    sampling : int, optional
        Period between measurements, in (simulated) minutes. The default is 5.
    cell_class : CcaSR_gillespie or descendant class, optional.
        The type of cell to simulate. The default is CcaSR_gillespie

    Returns
    -------
    fluo : 2D array of float
        Generated fluorescence. Dimensions are (cells, time).

    """
    
    # Run Gillespie simulations: (not using multiprocessing here because it's relatively fast)
    fluo = []
    for l in range(stims.shape[0]):
        cell = cell_class() # Instantiate new "cell"
        cell.set_light_events(stims[l]) # Set future light events
        ts = cell.run(stims.shape[1]*sampling) # Run until the end
        fluo.append([x['F'] for x in ts[1:]]) # Append fluorescence to list (skip first timepoint before any stim)
        print('%d/%d cells simulated'% (l+1,stims.shape[0]))
        
    fluo = np.array(fluo)
    
    # Simulate camera/measurement noise:
    fluo = camera_sim(fluo)
    
    return fluo


def evaluation_set(
        stims, 
        cut_off,
        cell_class = CcaSR_gillespie,
        future_realizations=5000, 
        sampling=SAMPLING, 
        num_workers=8,
        ):
    """
    Generate evaluation set with multiple future realization of the cell 
    response after an arbitrary cut-off timepoint.

    Parameters
    ----------
    stims : 2D array of bool.
        Stimulations to apply to the cells. Dimensions are (cells, time).
    cut_off : int
        cut-off time point after which multiple realizations of the cell 
        response are computed.
    cell_class : CcaSR_gillespie or descendant class, optional.
        The type of cell to simulate. The default is CcaSR_gillespie
    future_realizations : int, optional
        Number of future realization to compute per cell. The default is 5000.
    sampling : int, optional
        Period between measurements, in (simulated) minutes. The default is 5.
    num_workers : int, optional
        Number of parallel workers to use to compute each cell trajectory. If
        None, no parallel execution is used. The default is 8. 

    Returns
    -------
    res : List[(1D array, 2D array)]
        List of all cells and their simulations before and after cutoff. Each
        list element contains the single trajectory of the cell before cut-off,
        and its multiple realisations after cut-off. Dimensions of the two
        arrays are (time,) and (realizations, time).

    """
    
    if num_workers is None:
        res = []
        for s, stim in enumerate(stims):
            print(f"{s}/{stims.shape[0]} cells")
            res.append(
                _evaluation(stim, cut_off, future_realizations, sampling, cell_class)
                )
        return res
    
    # Run _evaluation function in parallel:
    with Pool(num_workers) as pool:
        res = pool.starmap(
            _evaluation,
            zip(
                stims,
                itertools.repeat(cut_off),
                itertools.repeat(future_realizations),
                itertools.repeat(sampling),
                itertools.repeat(cell_class)
                )
            )
    
    return res


def _evaluation(stims, cut_off, future_realizations, sampling, cell_class):
    """
    This function implements the actual evaluation routine for a specific cell.
    It has to be a top-level function otherwise the parallel processing 
    library is sad :(

    Parameters
    ----------
    stims : 1D array of bool.
        Stimulations to apply to the 1 cell. Dimensions are (time,).
    cut_off : int
        cut-off time point after which multiple realizations of the cell 
        response are computed.
    future_realizations : int, optional
        Number of future realization to compute per cell. The default is 5000.
    sampling : int, optional
        Period between measurements, in (simulated) minutes. The default is 5.
    cell_class : CcaSR_gillespie or descendant class, optional.
        The type of cell to simulate. The default is CcaSR_gillespie

    Returns
    -------
    Past : 1D numpy array
        The single trajectory of the cell until cut-off. Dimensions are (time,)
    Future : 2D numpy array.
        The multiple trajectories of the cell's response "realizations" after
        cut-off. Dimensions are (realizations, time).

    """
    
    cell = cell_class() # Instantiate new "cell"
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

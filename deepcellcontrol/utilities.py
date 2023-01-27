#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:14:23 2020

@author: jeanbaptiste
"""
import os

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import cv2

# Movies etc colormaps:
gfpmap = np.zeros(shape=(256,4))
gfpmap[:,3]= 1
gfpmap[:,1] = np.arange(0,1,step=1/256)
gfpmap[:,2] = np.arange(0,.5,step=.5/256)
gfpmap = ListedColormap(gfpmap)

graymap = np.zeros(shape=(256,4))
graymap[:,3]= 1
graymap[:,:3] = np.repeat(np.arange(0,1,step=1/256)[:,np.newaxis],3,axis=-1)
graymap = ListedColormap(graymap)


def getRandomStimulations(totalchambers, timepoints, upper_limit=1, lower_limit=-2):
    '''
    Generate an array of bounded cumulative sums of +1/-1 Bernouilli 
    sequences that is then thresholded around 0. The final binary sequence 
    qualitatively approximates the type of optogenetic input control sequences 
    presented in Chait et al.

    Parameters
    ----------
    totalchambers : int
        Total number of cells/chambers/sequences to generate.
    timepoints : int
        Length of sequences.
    upper_limit : int, optional
        Upper bound of the cumulative sum of the sequence. Higher upper limits
        will generate more and longer sequences of 1s, or 'green' inputs.
        The default is 1.
    lower_limit : int, optional
        Lower bound of the cumulative sum of the sequence. Lower lower limits
        will generate more and longer sequences of 0s, or 'red' inputs.
        The default is -2.

    Returns
    -------
    2D array of booleans
        Sequence of random binary/optogenetic inputs.

    '''
    
    # Generate binary coin flips:
    flips = np.round(np.random.rand(totalchambers,timepoints))
    flips[flips==0]=-1
    
    # Initialize sum array:
    sum_arr = np.zeros((totalchambers,timepoints))
    sum_arr[:,0] = np.floor((upper_limit+lower_limit)/2)
    
    # Run over timepoints:
    for t in range(timepoints-1):
        # print((sum_arr[:,t], flips[:,t]))
        sum_arr[:,t+1] = sum_arr[:,t] + flips[:,t] # Add timepoints sequentially
        sum_arr = sum_arr.clip(lower_limit,upper_limit) # Apply lower and upper limits
        
    return sum_arr>=0


def random_stimulations(
        timepoints = 36*12,nostim_timepoints = 3*12, total_simulations = 12000
        ):
    '''
    Produces a generic sample of random stimulations sequences

    Parameters
    ----------
    timepoints : int, optional
        Total number of timepoints. The default is 36*12.
    nostim_timepoints : int, optional
        Number of initial timepoints set to 0. The default is 3*12.
    total_simulations : int, optional
        Total number of stimulation sequences to generate. The default is 12000.

    Returns
    -------
    stims : 2D array of bool
        Random stimulation sequences.

    '''
    
    # Init stims array
    stims = np.empty((0,timepoints),dtype=bool)
    
    # Lambda to generate new stimulations and append to stims:
    get_stims = lambda upper_limit, lower_limit: np.append(
            stims,
            getRandomStimulations(
                min(total_simulations-stims.shape[0],int(total_simulations/6)),
                timepoints,
                upper_limit=upper_limit,
                lower_limit=lower_limit
                ),
            axis=0
            )
    
    # Generate random stimulations:
    stims = get_stims(upper_limit=3,lower_limit=-2)
    stims = get_stims(upper_limit=1,lower_limit=-2)
    stims = get_stims(upper_limit=0,lower_limit=-2)
    stims = get_stims(upper_limit=3,lower_limit=-3)
    stims = get_stims(upper_limit=2,lower_limit=-3)
    stims = get_stims(upper_limit=4,lower_limit=-3)
    stims = get_stims(upper_limit=1,lower_limit=-3)
    
    # Shuffle them:
    np.random.shuffle(stims)
    
    # Set first few hours to all red inputs, like in the experiments
    stims[:,:nostim_timepoints] = 0
    
    return stims


def OptoPlotBackground(stims,x=None,ymin=0,ymax=1,alpha=1):
    '''
    Plot a background of red and green stripes for optogenetic inputs

    Parameters
    ----------
    stims : 1D array of floats
        0s and 1s representing a time sequence of red and green optogenetic 
        inputs.
    x : 1D array of ints or floats, optional
        An array of the same size as stims for X-axis coordinates of the 
        inputs. If None, stims sequence indexes will be used.
        The default is None.
    ymin : float or int, optional
        Lower bound of the stripes. 
        The default is 0.
    ymax : float or int, optional
        Upper bound of the stripes. 
        The default is 1.

    Returns
    -------
    None.

    '''
    
    lightx, lighty = [], []
    for i, e in enumerate(stims):
        if x is None:
            illumination_start = i-.5
            illumination_stop = i+.5
        else:
            if i>0:
                illumination_start = x[i]-(x[i]-x[i-1])/2
            else:
                illumination_start = x[i]
            if i<len(x)-1:
                illumination_stop = x[i]+(x[i+1]-x[i])/2
            else:
                illumination_stop = x[i]
                
        lightx.append(illumination_start)
        lightx.append(illumination_start)
        lightx.append(illumination_stop)
        lightx.append(illumination_stop)
        lighty.append(ymax if not e else ymin)
        lighty.append(ymax if e else ymin)
        lighty.append(ymax if e else ymin)
        lighty.append(ymax if not e else ymin)
        
    lightx, lighty = np.array(lightx), np.array(lighty)
    
    plt.fill_between(lightx, lighty,ymin,facecolor='xkcd:light mint',alpha=alpha)
    plt.fill_between(lightx, lighty,ymax,facecolor='xkcd:pale rose',alpha=alpha)

def evaluationPlot(stims,past,future,prediction,savefig=None,show=True,dyn_range=4095):
    '''
    Plot an evaluation sample, the mean and standard deviation estimates for
    future responses based on Gillespie simulations, and the neural network 
    forecast.

    Parameters
    ----------
    stims : 1D array of floats
        Optogenetic stimulations for this sample (past and future).
    past : 1D array of floats
        Past simulated fluorescence trajectory for this sample.
    future : 1D or 2D array of floats
        Future simulated fluorescence trajectories for this sample. Dimensions
        are simulations x time.
    prediction : 1D array of floats
        Neural network prediction for this sample.
    savefig : NoneType or str or path-like or file-like, optional
        Filename to save plot as. See documentation for fname parameter of
        matplotlib.pyplot.savefig. If None, no file is saved.
        The default is None.
    show : bool, optional.
        Flag to show the figure.
        The default is True.
    dyn_range : int or float, optional
        Dynamic range to plot the data.

    Returns
    -------
    None.

    '''
    
    # Compute mean and std dev of future predictions:
    future_mean = np.mean(future,axis=0)
    future_std = np.std(future,axis=0)
    
    # Get past and future timepoints (t=0 is present, ie latest past timepoint)
    pt = np.arange(-len(past)+1,1) # 'Past' X-axis timepoints 
    ft = np.arange(1,future.shape[-1]+1) # 'Future' X-axis timepoints
    
    # Plot stimulations background:
    OptoPlotBackground(stims,
                       x=np.concatenate((pt,ft),axis=0),
                       ymin=0,
                       ymax=dyn_range)
    
    if future.ndim == 2:
        # Plot 'past' fluorescence trajectory:
        plt.plot(pt,past,'xkcd:black')
        
        # Plot standard deviation envelop of future trajectories:
        plt.fill_between(ft,
                         (future_mean+future_std),
                         (future_mean-future_std),
                         facecolor='xkcd:grey', alpha=.5)
        
        # Plot 3 random future trajectories:
        for _ in range(3):
            plt.plot(ft,future[np.random.randint(future.shape[0]),:],'xkcd:dark grey',lw=.5)
            
        # Plot mean future trajectory:
        plt.plot(ft,future_mean,color='xkcd:black')
        
    else:
        # Plot full fluorescence trajectory:
        plt.plot((*pt,*ft), (*past,*future), color='xkcd:black')
    
    # Plot prediction:
    plt.plot((0, *ft),(past[-1], *prediction),color='xkcd:electric blue')
    
    # Wrap up:
    plt.plot([.5, .5],[0,dyn_range],'--k')
    plt.ylim(0,dyn_range)
    plt.xlim(-len(past)+1,future.shape[-1])
    plt.xlabel('time points')
    plt.ylabel('Fluorescence (a.u.)')
    plt.title('Evaluation sample')
    if savefig is not None:
        plt.savefig(savefig+'.png',dpi=300)
        plt.savefig(savefig+'.svg',dpi=300)
        plt.savefig(savefig+'.pdf',dpi=300)
    if show:
        plt.show()
    plt.clf()


def plot_autoencoding(Y, Yhat, features_list = None, savefig = None, show = False):
    """
    Plot autoencoding results

    Parameters
    ----------
    Y : 2D numpy array
        Groundtruth for a single sample. Dimensions are (time, features)
    Yhat : 2D numpy array
        Prediction for a single sample. Dimensions are (time, features)
    features_list : List[str], optional
        List of features names. The default is None.
    savefig : str, optional
        Path to save the plot to. The default is None.

    Returns
    -------
    None.

    """
    
    features_list = features_list or [f"feature #{x}" for x in range(Y.shape[1])]
    
    plt.figure(figsize=(6, 2*len(features_list)), dpi=300)
    for f, feature in enumerate(features_list):
        
        ax = plt.subplot(len(features_list),1,f+1)
        ax.plot(Y[:,f])
        ax.plot(Yhat[:,f])
        ax.set_ylabel(feature)
        
    if  savefig is not None:
        plt.savefig(savefig + ".png",dpi=300)
        plt.savefig(savefig + ".svg",dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close()

def sine_objective(period=8*60,
                    offset=1000,
                    amplitude=750,
                    delay=0,
                    duration=24*60,
                    sampling=5):
    '''
    Generate sine objective vector

    Parameters
    ----------
    period : int or float, optional
        Period of the sine, in minutes. 
        The default is 8*60.
    offset : int or float, optional
        The offset value added to the sine function. 
        The default is 750.
    amplitude : int or float, optional
        The amplitude of the sine function. 
        The default is 500.
    delay : int or float, optional
        The amount to delay the shift the sine by, in minutes.
        The default is 0.
    duration : int or float, optional
        The total duration of the sine objective, in minutes. Must be a
        multiple of sampling.
        The default is 24*60.
    sampling : int, optional
        The sampling rate of control objective, in minutes.
        The default is 5.

    Returns
    -------
    1D numpy array
        Sine objective.

    '''
    
    return np.array(
        [offset + amplitude*np.sin(2*np.pi*(t*sampling-delay)/period - np.pi/2) for t in range(0,int(duration/sampling))]
        )

def concentric_sines_objectives(gridsize,
                          period=8*60,
                          offset=1000,
                          amplitude=750,
                          prop_speed=2,
                          duration=24*60,
                          sampling=5):
    '''
    Generate control objective array of concentric sinewaves

    Parameters
    ----------
    gridsize : int
        Size of the array to draw the sines in.
    period : int or float, optional
        Period of the sine, in minutes. 
        The default is 8*60.
    offset : int or float, optional
        The offset value added to the sine function. 
        The default is 750.
    amplitude : int or float, optional
        The amplitude of the sine function. 
        The default is 750.
    prop_speed : int or float, optional
        'Propagation speed' of the sine waves. 
        The default is 2.
    duration : int or float, optional
        The total duration of the sine objective, in minutes. Must be a
        multiple of sampling.
        The default is 24*60.
    sampling : int, optional
        The sampling rate of control objective, in minutes.
        The default is 5.

    Returns
    -------
    objectives: 2D numpy array
        The objectives array to feed into the control pipeline. The dimensions
        are (gridsize**2) -by- duration/sampling

    '''
    
    # Define objective propagation coefficient:
    obj_prop = (prop_speed*period)/(gridsize)
    
    # Concentric circles in a grid:
    objectives = np.full((gridsize**2,int(duration/sampling)),offset-amplitude)
    c=0
    for i in range(gridsize):
        for j in range(gridsize):
            delay = np.sqrt((i-float(gridsize)/2)**2+(j-float(gridsize)/2)**2)*obj_prop
            objectives[c,int(delay/sampling):] = sine_objective(
                period=period,
                offset=offset,
                amplitude=amplitude,
                delay=delay,
                duration=duration,
                sampling=sampling
                )[int(delay/sampling):]
            c += 1
    return objectives

def movie_objective(movie_folder,
                    shape=(3840,2160),
                    omin=250,
                    omax=1750,
                    clip_ratio=.02,
                    color=False):
    '''
    Turn movie frames into control objectives for cells

    Parameters
    ----------
    movie_folder : str
        Path to folder containing movie frames.
    shape : tuple of 2 ints, optional
        Target shape of the processed movie. Each pixel is a cell.  
        The default is (100,100).
    omin : int or float, optional
        Min value to clip the movie pixels to. 
        The default is 250.
    omax : int or float, optional
        Max value to clip the movie pixels to.
        The default is 1750.
    clip_ratio : float, optional
        Ratio of extrema pixels to clip to omin or omax. 
        The default is .02.
    color : bool, optional
        Whether the movie frame files are RGB.
        The default is False.

    Returns
    -------
    objective : 2D numpy array of floats
        Single-cell control objectives. The dimensions are shape[0]*shape[1] 
        -by- number of movie frames

    '''

    # Identify images:
    imgfiles = [x for x in os.listdir(movie_folder) if os.path.splitext(x)[1].lower() in ('.tif','.tiff','.png')]
    imgfiles.sort()
    objective = np.empty(shape+(len(imgfiles),))
    
    # Compile into objectives array:
    for f, filename in enumerate(imgfiles):
        if color:
            I = cv2.imread(os.path.join(movie_folder,filename),cv2.IMREAD_COLOR)
            I = np.mean(I,axis=2).astype(np.uint16)
        else:
            I = cv2.imread(os.path.join(movie_folder,filename),cv2.IMREAD_ANYDEPTH)
        I = cv2.resize(I,shape[::-1])
        objective[:,:,f] = np.array(I,dtype=float)
    
    # min & max:
    mmin = np.quantile(objective,clip_ratio)
    mmax = np.quantile(objective,1-clip_ratio)
    
    objective = ((objective-mmin)/(mmax-mmin))*(omax-omin)+omin
    objective = np.clip(objective,a_min=omin,a_max=omax)
    
    objective = np.reshape(objective, (shape[0]*shape[1],len(imgfiles)))
    
    return objective

def random_objectives(
        total_cells,
        total_timepoints,
        omin=250,
        omax=1750,
        std_slope=1500/(6*12),
        sampling = 5
        ):
    '''
    Generate random cell control objectives.

    Parameters
    ----------
    total_cells : int
        Number of single-cell objectives to generate.
    total_timepoints : TYPE
        Number of timepoints per objective time-series.
    omin : int or float, optional
        Min value that objectives can reach. 
        The default is 250.
    omax : int or float, optional
        Max value that objectives can reach.
        The default is 1750.
    std_slope : float, optional
        Std deviation of the slope of the objectives. Each random objective is
        a succession of linear segments of normally-randomized slope.
        The default is 1500/(6*12).
    sampling : int, optional
        The sampling rate of control objective, in minutes.
        The default is 5.

    Returns
    -------
    objective : TYPE
        DESCRIPTION.

    '''
    
    # Init array:
    objective = np.empty((total_cells,total_timepoints))
    objective[:,0] = np.random.uniform(omin, omax, size=total_cells)
    
    # Run through cells, generate independent random objectives:
    for c in range(total_cells):
        t0 = 0
        while t0 < total_timepoints-2:
            
            # Get random segment slope:
            slope = np.random.normal(0,std_slope, size=1)
                
            # Get random duration for next segment (up to 4 hours):
            t1 = np.min((t0 + np.random.randint(1,int(4*60/sampling)),
                        total_timepoints-1))
            
            # Add to objective:
            for t0 in range(t0,t1):
                
                objective[c,t0+1] = objective[c,t0] + slope[0]
                
                # If objective goes above or below min/max, break to next segment:
                if objective[c,t0+1]<omin or objective[c,t0+1]>omax:
                    break
    
    return objective


def plotq(fluo, x = None, q = .5, color="b"):
    """
    Quantile plot

    Parameters
    ----------
    fluo : 2D numpy array
        Fluorescence levels. Dimensions are (Cells, time)
    q : float, optional
        The quantile to plot around the median. e.g. if q=0.5, the region
        between 0.25 and 0.75 is shown in shaded fill
        The default is .5.
    color : str, optional
        Color to plot the results as. The default is "b".

    Returns
    -------
    None.

    """
    
    if x is None:
        x = np.arange(0,len(fluo[0]),1)/12
    plt.plot(x,np.nanmedian(fluo,axis=0), color=color)
    plt.fill_between(
        x,
        np.nanquantile(fluo,.5-q/2,axis=0),
        np.nanquantile(fluo,.5+q/2,axis=0),
        color=color,
        alpha=.2,
        )

def color_img(I, vmin=0.1, vmax=0.7, cmap=gfpmap):
    """
    Color image based on color map

    Parameters
    ----------
    I : 2D array of floats
        Grayscale image to color. Dynamic range from 0 to 4095
    vmin : float, optional
        Minimum of image dynamic range, once it is rescaled to [0, 1]. 
        The default is 0.1.
    vmax : float, optional
        Maximum of image dynamic range, once it is rescaled to [0, 1]. 
        The default is 0.7.
    cmap : colormap object, optional
        Matplotlib colormap to apply to I. The default is gfpmap.

    Returns
    -------
    I : 3D array of floats
        Color image, dynamic range [0, 1] (unless cmap scale is different).

    """
    
    I = ((I.astype(np.float64)/4095)-vmin)/(vmax-vmin)
    I = cmap(I)[:,:,:3]
    
    return I

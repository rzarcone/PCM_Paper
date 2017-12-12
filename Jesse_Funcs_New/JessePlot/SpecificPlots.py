# Python3 divison Int/Int = Float
from __future__ import division
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

#==============================================================================
# Data Plots 
#==============================================================================

    
def plotDC(filename):
    '''Needs loadDCData filename'''
    data = globals()[filename]    
    plt.plot(data['V'], data['I'])
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

def semilogyDC(filename):
    '''Needs loadDCData filename'''
    data = globals()[filename]    
    plt.semilogy(data['V'], abs(data['I']))
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')

def plotPulses( V, R, pulse_starts=np.zeros(1), pulse_stops=np.zeros(1), 
               RVplot=False):
    """Plots a series of gradual set/reset pulses
    """
    if not pulse_stops.any():
        pulse_stops = np.array([R.size])
        
    lengths = pulse_stops - pulse_starts
    print (lengths)
    

    if RVplot:
        for l, pstart, pstop in zip(lengths, pulse_starts, pulse_stops):
            plt.semilogy(V[pstart:pstop], R[pstart:pstop], 'o-', linewidth=1)
            plt.xlabel('Voltage WL (V)')
            plt.ylabel('Resistance (Ohms)')

    else:
        for l, pstart, pstop in zip(lengths, pulse_starts, pulse_stops):
            plt.semilogy(np.arange(l)+1, R[pstart:pstop])
            plt.xlabel('Pulse Number (n)')
            plt.ylabel('Resistance (Ohms)')


def createRArray(R, pulse_starts=np.zeros(1), pulse_stops=np.zeros(1), array_length=None):
    '''
    Creates an array with a row for each pulse sequence ((pulse_starts.size, array_length))
    Sequences defined by pulse_starts, and pulse_stops (arrays of indices).
    '''

    if not pulse_stops.any():
        pulse_stops = np.array([R.size])
        
    lengths = pulse_stops - pulse_starts
    print (lengths)

    if not array_length:        
        array_length = max(lengths)

    R_array = np.zeros( (lengths.size, array_length))

    for i, l, pstart, pstop in zip(np.arange(lengths.size), lengths, pulse_starts, pulse_stops):
       R_array[i, 0:l] = R[pstart:pstop]

    R_array[R_array == 0] = np.nan        
    return R_array
        





#==============================================================================
# Specific Plots
#==============================================================================

    
def plotDensity( Px, Pyx, limits=[-5,15], no_ticks=False, fill=True,
                color_list=['r', 'y', 'g', 'c'], plot_Pyx=False, cmap=False):
    '''
    Plots distributions given Px[array], and Pyx[CapacityTools.FunctionList]
    '''
    Py = Pyx.dot(Px)
    PxPyx = Pyx * Px    
    
    fig, ax = plt.subplots()
    
    x = np.linspace(limits[0], limits[1], 1000)
    plt.plot(x, Py(x), 'b')
    
    if plot_Pyx:
        Pplot = Pyx
    else:
        Pplot = PxPyx
        
    Pplot = [p for i, p in enumerate(Pplot) if Px[i] != 0]        
        
    for i, p in enumerate(Pplot):
        if cmap:
            color = cmap(i/len(Pplot))
        else:
            color = color_list[i%len(color_list)]
            
        plt.plot(x, p(x), '--', color=color)

        if fill:
            plt.fill_between(x, p(x), color=color, alpha=0.1)

    if no_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
    plt.show()
    
    
    
def plotDividers(dividers):
    '''
    Add dividers to a plot
    '''
    ax = plt.gca()
    ylim0 = ax.get_ylim()
    
    for d in dividers:
        plt.plot([d,d], [0,1e5], 'k', linewidth=1.5)

    ax.set_ylim(ylim0)                
    

def plotDI(dI, fill=True, color_list=['r', 'y', 'g', 'c']):
    '''
    Add dI overlay to a plot
    '''
    ax = plt.gca()
    xlim0 = ax.get_xlim()    
    x = np.linspace(xlim0[0], xlim0[1], 1000)

    print (dI)
    for i, p in enumerate(dI):
        plt.plot(x, p(x), '-', color=color_list[i%len(color_list)], linewidth=1.5)
        if fill:
            plt.fill_between(x, p(x), hatch='/', color=color_list[i%len(color_list)], alpha=0.4)
        
    plt.show()



    
def plotDensityDiscrete( Px_discrete, Pyx_discrete ):
    '''
    Plot Discrete Pyx Matrices
    '''
    plt.matshow(np.log10(Pyx_discrete), cmap=plt.cm.Blues)
    plt.xlabel('Write')
    plt.ylabel('Read')
    plt.colorbar(label='$log_{10}$ P(Read|Write)', fraction=0.15, shrink=0.8)
    plt.show()



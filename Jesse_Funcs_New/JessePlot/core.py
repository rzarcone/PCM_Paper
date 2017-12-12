# Python3 divison Int/Int = Float
from __future__ import division
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os


#==============================================================================
# Plotting functions
#==============================================================================



def rc(style='ppt_1', close=True):
    '''A function to set the default parameters for plots, just how I like 'em
    
    Keywords
    ------
    style : String, Options include 'ppt_1', 'ppt_2'
    
    close : Bool, Close all plots
    '''        
    #This all could also be done with plt.rc(), ex:
    #plt.rc('font', **{'family':'sans-serif', 'sans-serif':'Helvetica'})
    
    
    if close:
        plt.close('all')
    else:
        plt.figure()

    #Common starting point for all styles
    plt.rcdefaults()
    
    p = plt.rcParams    
    
    p['lines.linewidth'] = 2
    p['axes.linewidth'] = 2
    p['xtick.minor.width'] = 1
    p['ytick.minor.width'] = 1
    p['xtick.major.width'] = 2
    p['ytick.major.width'] = 2
    p['xtick.minor.size'] = 5
    p['ytick.minor.size'] = 5
    p['xtick.major.size'] = 10
    p['ytick.major.size'] = 10
    p['figure.dpi'] = 80 
    p['savefig.dpi'] = 300 


    if style == 'ppt_1':
        p['font.size'] = 20
        p['axes.labelsize'] = 24
        p['figure.figsize'] = (8, 6) #w,h

    elif style == 'ppt_2':
        p['font.size'] = 16
        p['axes.labelsize'] = 20
        p['figure.figsize'] = (5, 6) #w,h  

    elif style == 'ppt_4':
        p['font.size'] = 16
        p['axes.labelsize'] = 20
        p['figure.figsize'] = (5, 3) #w,h  

    elif style == 'ppt_6':
        p['font.size'] = 16
        p['axes.labelsize'] = 20
        p['figure.figsize'] = (3, 3) #w,h 
        
    elif style == 'word_3':
        p['lines.linewidth'] = 1.5
        p['axes.linewidth'] = 1.5
        p['xtick.minor.width'] = 1
        p['ytick.minor.width'] = 1
        p['xtick.major.width'] = 1.5
        p['ytick.major.width'] = 1.5
        p['xtick.minor.size'] = 2
        p['ytick.minor.size'] = 2
        p['xtick.major.size'] = 5
        p['ytick.major.size'] = 5
        p['figure.dpi'] = 300 
        p['savefig.dpi'] = 900 
        p['font.family'] = 'sans-serif'
        p['font.sans-serif'] = ['Arial']
        p['font.weight'] = 'bold'
        p['axes.labelweight'] = 'bold'
        p['font.size'] = 10
        p['legend.fontsize']= 10
        p['axes.labelsize'] = 12
        p['figure.figsize'] = (29/12, 2) #w,h 
        
    elif style == 'word_2/3':
        p['lines.linewidth'] = 1.5
        p['axes.linewidth'] = 1.5
        p['xtick.minor.width'] = 1
        p['ytick.minor.width'] = 1
        p['xtick.major.width'] = 1.5
        p['ytick.major.width'] = 1.5
        p['xtick.minor.size'] = 2
        p['ytick.minor.size'] = 2
        p['xtick.major.size'] = 5
        p['ytick.major.size'] = 5
        p['figure.dpi'] = 300 
        p['savefig.dpi'] = 900 
        p['font.family'] = 'sans-serif'
        p['font.sans-serif'] = ['Arial']
        p['font.weight'] = 'bold'
        p['axes.labelweight'] = 'bold'
        p['font.size'] = 10
        p['legend.fontsize']= 10
        p['axes.labelsize'] = 12
        p['figure.figsize'] = (29/6, 2) #w,h 

    elif style == 'word_3x3':
        p['lines.linewidth'] = 1.5
        p['axes.linewidth'] = 1.5
        p['xtick.minor.width'] = 1
        p['ytick.minor.width'] = 1
        p['xtick.major.width'] = 1.5
        p['ytick.major.width'] = 1.5
        p['xtick.minor.size'] = 2
        p['ytick.minor.size'] = 2
        p['xtick.major.size'] = 5
        p['ytick.major.size'] = 5
        p['figure.dpi'] = 300 
        p['savefig.dpi'] = 900 
        p['font.family'] = 'sans-serif'
        p['font.sans-serif'] = ['Arial']
        p['font.weight'] = 'bold'
        p['axes.labelweight'] = 'bold'
        p['font.size'] = 10
        p['legend.fontsize']= 10
        p['axes.labelsize'] = 12
        p['figure.figsize'] = (29/3, 29/3) #w,h 

    elif style == 'ipynb':
        plt.rcdefaults()
        p['lines.linewidth'] = 1.5
        p['axes.linewidth'] = 1.5
        p['xtick.minor.width'] = 1
        p['ytick.minor.width'] = 1
        p['xtick.major.width'] = 1.5
        p['ytick.major.width'] = 1.5
        p['xtick.minor.size'] = 2
        p['ytick.minor.size'] = 2
        p['xtick.major.size'] = 5
        p['ytick.major.size'] = 5
        # p['figure.dpi'] = 300 
        # p['savefig.dpi'] = 900 
        p['font.family'] = 'sans-serif'
        p['font.sans-serif'] = ['Arial']
        p['font.weight'] = 'bold'
        p['axes.labelweight'] = 'bold'
        p['font.size'] = 14
        p['legend.fontsize']= 14
        p['axes.labelsize'] = 16
        p['figure.figsize'] = (58/12, 4) #w,h 

    else:
        print 'Wrong style: [ppt_1, ppt_2, ppt_4, ppt_6, word_3, word_2/3, ipynb]'


def saveFig(filename, transparent=True, format='png', path='./../figs/'):
    '''Convenience function to make a transparent plot
    Creates a folder at 'path', and puts a figure in it.
    '''
    if path:
        if not os.path.exists(path):
            os.mkdir(path)
        filename = path + filename
    
    plt.savefig(filename, transparent=transparent, format=format)




def legend(title=''):
    plt.legend(shadow = False, loc ='upper left', bbox_to_anchor = (1.0, 1.0), 
    title=title, fontsize = 14)

# Plotting Helper Funcs
def expandAxes(ratio=1.1):
    ax = plt.gca()
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    dx = ((xlimits[1]-xlimits[0]) * (ratio-1) ) / 2.
    dy = ((ylimits[1]-ylimits[0]) * (ratio-1) ) / 2.
    ax.set_xlim(xlimits[0]-dx, xlimits[1]+dx)
    ax.set_ylim(ylimits[0]-dy, ylimits[1]+dy)






def shiftedColorMap(cmap, data=False, start=0, midpoint=0.5, stop=1.0, 
                    set=True, register=False, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    if type(data) == np.ndarray:
        vmax = data.max()
        vmin = data.min()
        midpoint = 1 - vmax/(vmax + abs(vmin))


    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    if set:
        plt.set_cmap(newcmap)

    if register:
        plt.register_cmap(cmap=newcmap)

    return newcmap

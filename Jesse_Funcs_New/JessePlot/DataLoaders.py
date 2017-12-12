# Python3 divison Int/Int = Float
from __future__ import division
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os


#==============================================================================
# Data Loading / Manipulation
#==============================================================================
# Anything that uses globals() can only be run from the module of interest


def loadNPZ(filename, suffix = ''):
    '''
    Load arrays from a .npz pickled file to the global namespace
    '''
    print "Importing...\n"  
    with np.load(filename) as data:
        for key in data.keys():
            name = key + suffix
            print name
            globals()[name] = data[key]


def loadPulseData(filename, suffix = ''):
    """Given a pulse text file, puts the data into the global namespace
    """
    data = np.genfromtxt(filename+'.txt', skip_header=3, names=True,
                   dtype='i8,f8,S5,f8,f8,f8,f8,f8,f8')
    print "Importing...\n"
    for key in data.dtype.fields.keys():
        name = key + suffix
        print name
        globals()[name] = data[key]


def loadDCData(filename):
    """Given a DC sweep text file, puts the data into the global namespace
    """
    data = np.genfromtxt(filename+'.txt', skip_header=3, names=True)
    globals()[filename] = data


def getTxtFileList(folder_path='.'):
    """Returns a list of '.txt' filenames in the folder 'folder_path'
    """
    files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(f) and os.path.splitext(f)[1] == '.txt']
    return files




#==============================================================================
# Data  Manipulation
#==============================================================================

        
def scaleArray(array, limits=(0, 1)):
    return ((array - array.min()) / (array.max()-array.min())) * limits[1] + limits[0]    



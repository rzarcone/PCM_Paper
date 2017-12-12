from __future__ import division

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import Lasso, LassoCV

#==============================================================================
# Compressed Sensing
#==============================================================================
def cs(k, n, m, n_examples, 
       noise_func = lambda y:add_noise(y, sigma=0.0),
       algorithm='omp', alpha = 1e-2):
    '''
    Full compressed sensing on n_examples
    '''
    
    x = k_sparse_gaussian(k=k, n=n, n_examples=n_examples, sigma=1)
    xhat = np.zeros(x.shape)
    
    for i in xrange(n_examples):
        R = R_gaussian(n, m)
        xhat[:,i] = reconstruct(x[:,i], 
                     R,  
                     noise_func=noise_func,
                     algorithm=algorithm,
                     alpha= 1e-1)
        print '%d/%d' % (i+1, n_examples)
    
    err = NMSE(x,xhat)
    print 'NMSE: %0.2f dB\n' % (err)
    return x, xhat, err




def reconstruct(x, R=None, compression=4, 
                noise_func = lambda y: add_noise(y, sigma=0.1),
                algorithm='omp', alpha = 1e-2):
    '''
    Perform compressed sensing reconsturction on a sparse matrix x
    
    Params
    ------
    x
        array(n, n_examples)
    R
        array(m, n)
    '''
    n = x.shape[0]

    if type(R) != np.ndarray:
        m = np.ceil(n / compression) 
        R = R_gaussian(n, m)
    else:
        m = R.shape[0]
        

    y = R.dot(x)
    y = noise_func(y) 

        
        
    ## SKLEARN L1 Lasso (Coordinate Descent)
#    rgr_lasso = linear_model.LassoCV()
    if algorithm == 'lasso':
        # rgr_lasso = linear_model.Lasso(alpha=alpha, max_iter=10000, tol=1e-9)
        rgr_lasso = linear_model.LassoCV()
        rgr_lasso.fit(R, y)
        xhat = rgr_lasso.coef_.T.squeeze()
    
    if algorithm == 'omp':
        omp_cv = OrthogonalMatchingPursuitCV()
        omp_cv.fit(R, y)
        xhat = omp_cv.coef_
    
    return xhat





def learn_proj(x, Phi, eta=0.01, max_iter=1000, epsilon=1e-4, 
               R0=None, compression=4, noise_sigma=0.0):
    '''
    Learn the compressed sensing projection matrix. 
    
    Params:
    -------
    I
        array(npixels, nexamples)    
    Phi
        array(npixels, nbases)
    R0
        array(ncompressed, npixels) 
        
    Outputs:
    --------
    R
        array(ncompressed, npixels)
    '''              

    batch_size=x.shape[1]
    n = x.shape[0]
    m = np.ceil(n / compression) 

    # Initialize R
    if type(R0) != np.ndarray:
        R = 1/n * np.random.randn(m, n)
    else:
        R = R0

# For other minimizers
#    def cost(R, *args):
#        R = R.reshape(m, n)
#        a = cs(x, Phi, R, compression, noise_sigma) #(nbases, nex)
#        y = np.dot(R, x).squeeze()  #(ncomp, nex)
#        yhat = R.dot(Phi).dot(a) #(ncomp, nex)
#
#        cost = 1/2 * np.sum((y-yhat)**2) + np.sum(np.abs(a))       
#        return cost
#    
#
#    def grad_cost(R, *args):
#        R = R.reshape(m, n)
#        dR = np.dot( y-yhat,  Phi.dot(a).T ) #(ncomp, nex), (nex, npixels)        
#        return dR.ravel()

    
    for i in xrange(max_iter):

        a = cs(x, Phi, R, compression, noise_sigma) #(nbases, nex)

        y = np.dot(R, x).squeeze()  #(ncomp, nex)
        yhat = R.dot(Phi).dot(a) #(ncomp, nex)
        err = y-yhat
        
        C = 1/2 * np.sum(err**2) + np.sum(np.abs(a))       
        dCdR = -np.dot( err,  Phi.dot(a).T ) #(ncomp, nex), (nex, npixels)
        
        R = R - eta * dCdR
#        R = R / R.sum(axis=1)[:,np.newaxis] # normalize each row (relative weights)

        eps = abs(eta * dCdR).mean()        
#        print a.nonzero()[0].shape[0]
        print 'Cost: %0.2e, dR: %0.2e' % (C, eps)
#        plt.matshow(dR)

        
        if eps < epsilon:
            break
    
    return R






#==============================================================================
# Data Generators
#==============================================================================

def k_sparse_gaussian(k=3, n=100, n_examples=1, sigma=1.0):
    '''Generate a k-sparse vector of length n, sampled from 
    gaussian with mean 0 and stddev sigma.
    '''
    x = np.zeros( (n, n_examples) )
    inds = range(n)
    for i in range(n_examples):
        k_inds = np.random.choice(inds, k, replace=False)
        x[k_inds, i] = sigma * np.random.randn(k)

    return x

def R_gaussian(n, m):
    '''Create a dense gaussian random matrix, column normalized.
    '''
    R = 1/m * np.random.randn(m, n)
    R = R / np.sum(R, axis=0)
    return R


def NMSE(x, xhat, dB=True):
    '''Normalized Mean-Squared Error of Reconstruction
    '''
    NMSE = np.mean( (x-xhat)**2 ) / np.mean(x**2)
    if dB:
        NMSE = 10 * np.log10(NMSE)
    return NMSE
    
    
def add_noise(x, sigma=0.1):
    '''Add Gaussian noise
    '''
    noise = sigma * np.random.randn(*x.shape) 
    y = x + noise
    return y
    
    
#==============================================================================
# Analyzing Data
#==============================================================================

def transition_point(x, y, threshold):
    '''Find first point (x) where abs(dy) > threshold'''
    dy = np.abs(y[1:]-y[:-1]) / 2
    i = np.where(dy >= threshold)[0]

    if i.size > 0:
        i = i[0]
        xpos = (x[i] + x[i + 1]) / 2

    else:
        xpos = np.nan

    return xpos, i
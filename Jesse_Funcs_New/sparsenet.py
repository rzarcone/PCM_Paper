from __future__ import division

from scipy import io, optimize
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

from sklearn import linear_model

IMAGE_FILE = './../data/IMAGES.mat'

def learn_dict(N=64, M=256, lambdav=0.1, eta=3.0, num_trials=1000, batch_size=100, BUFF=4, save=False):
    """
    N: # Inputs
    M: # Outputs
    lambdav: Sparsity Constraint
    eta: Learning Rate
    num_trials: Learning Iterations
    batch_size: Batch size per iteration
    BUFF: Border when extracting image patches
    """
    IMAGES = io.loadmat(IMAGE_FILE)
    IMAGES = IMAGES['IMAGES']

    (imsize, imsize, num_images) = np.shape(IMAGES)

    sz = np.sqrt(N)
    eta = eta / batch_size

    # Initialize basis functions
    Phi = np.random.randn(N,M)
    Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))) )

    # Minibatch of patches
    I = np.zeros((N,batch_size))

    for t in xrange(num_trials):
        
        # Choose a random image
        imi = np.ceil(num_images * random.uniform(0,1))

        # Fill the mini-batch with patches
        for i in xrange(batch_size):
            r = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
            c = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))

            I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1],N,1)

        # Coefficient Inference
        ahat = sparsify_LCA(I,Phi,lambdav)
    
        # Calculate Residual Error
        R = I-np.dot(Phi,ahat)
    
        # Update Basis Functions
        dPhi = np.dot(R, ahat.T)
        Phi = Phi + eta * dPhi
        # Normalize
        Phi = np.dot(Phi, np.diag( 1/np.sqrt( np.sum(Phi * Phi, axis = 0) ) ) )

        # Plot every 100 iterations
        if np.mod(t,100) == 0:
          print "Iteration " + str(t)
          if save: 
              np.savez('Phi.npz', Phi=Phi)
          plot_dict(Phi)
    
    return Phi



def sparsify_LCA(I, Phi, lambdav=0.1, num_iterations=75, eta=0.1, cost='L0', display=False):
    """
    Inference step. Learns coefficients.
    I: Image batch to learn coefficients
    Phi: Dictionary
    lambdav: Sparsity coefficient
    num_iterations:
    eta:
    """
    if len(I.shape) > 1:
        batch_size = I.shape[1]
    else:
        batch_size = 1
        I = np.c_[I]

    (N, M) = Phi.shape

    b = np.dot(Phi.T, I)
    G = np.dot(Phi.T, Phi) - np.eye(M)

    u = np.zeros((M,batch_size))

    l = 0.5 * np.max(np.abs(b), axis = 0)
    a = g(u, l)

    for t in range(num_iterations):
        u = eta * (b - np.dot(G, a)) + (1 - eta) * u
        a = g(u, l, cost)

        l = 0.95 * l
        l[l < lambdav] = lambdav

        if display:
            print sum((I - np.dot(Phi, a)**2))

    return a


def g(u, theta, cost='L0'):
    """
    LCA threshold function
    u: coefficients
    theta: threshold value
    """
    if cost == 'L0':
        #hard threshold        
        a = u
    elif cost == 'L1':
        #soft threshold
        a = u - theta
    a[np.abs(u) < theta] = 0
    return a



def sparsify_grad(I, Phi, lambdav=0.1, sigma=.2, num_iterations=75, cost='log'):
    batch_size = I.shape[1]    
    (N, M) = Phi.shape #Pixels, Basis
    a0 = np.random.randn(M, batch_size)

    if cost == 'L0':
        C = lambda a: np.sum(a!=0)
        C_prime = lambda a: (a==0)
    elif cost == 'L1':
        C = lambda a: np.sum(np.abs(a))
        C_prime = lambda a: (a!=0)
    elif cost == 'log':
        C = lambda a: np.sum( np.log(1.0 + a**2) )
        C_prime = lambda a: 2 * a / (1.0 + a**2) 
        
        

    def cost(a, *args):
        I, Phi, sigma = args
        R = I-np.dot(Phi,a)
        MSE = np.sum(R**2)
        
        a_scale = a / sigma
        sparse = lambdav * C(a_scale) #Sparse Penalty
        
        print MSE, sparse         
        return MSE + sparse


    def grad_cost(a, *args):
        I, Phi, sigma = args
        b = np.dot(Phi.T, I)
        C = np.dot(Phi.T,Phi)
        
        a_scale = a / sigma        
        sparse_prime = (lambdav / sigma) * C_prime(a_scale) #Sparse Penalty
        
        dCost_da = - np.c_[b] + np.c_[np.dot(C, a)] + np.c_[sparse_prime]
        return dCost_da
        

#   # Conjugate Gradient
#    a = optimize.fmin_cg(cost, a0, 
#                     fprime=grad_cost, args=(I, Phi, sigma), gtol=1e-40, norm=np.inf,
#                     epsilon=1, maxiter=None, full_output=0, 
#                     disp=1, retall=0, callback=None)

#   # BFGS
#    a = optimize.fmin_bfgs(cost, a0, fprime=grad_cost, args=(I, Phi, sigma), gtol=1e-40, norm=np.inf,
#                           epsilon=1.4901161193847656e-08, maxiter=None, 
#                           full_output=0, disp=1, retall=0, callback=None)

    # Steepest Descent
    a = a0
    for i in xrange(num_iterations):
        a -= 0.01 * grad_cost(a, I, Phi, sigma)
        cost(a, I, Phi, sigma)


    return a



def L0_cost(a):
    return a.nonzero()[0].shape[0]
    
def L1_cost(a):
    return sum(a)






#==============================================================================
# Compressed Sensing
#==============================================================================
def cs(I, Phi, R=None, compression=4, noise_sigma=0.0):
    n = I.shape[0]
    m = np.ceil(n / compression) 

    if type(R) != np.ndarray:
        R = 1/n * np.random.randn(m, n)
    
    y = np.dot(R, I)
    Omega = np.dot(R, Phi)

    #add noise
    if noise_sigma > 0.0:
        y = y + noise_sigma * np.std(y) * np.random.randn(*y.shape)


    ## SKLEARN L1 Lasso
#    rgr_lasso = linear_model.LassoCV()
    rgr_lasso = linear_model.Lasso(alpha=1e-6, max_iter=1000, tol=1e-5)
    rgr_lasso.fit(Omega, y)
    a = rgr_lasso.coef_.T.squeeze()
    
    return a




def learn_proj(I, Phi, eta=0.01, max_iter=1000, epsilon=1e-4, 
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

    batch_size=I.shape[1]
    n = I.shape[0]
    m = np.ceil(n / compression) 

    # Initialize R
    if type(R0) != np.ndarray:
        R = 1/n * np.random.randn(m, n)
    else:
        R = R0

# For other minimizers
#    def cost(R, *args):
#        R = R.reshape(m, n)
#        a = cs(I, Phi, R, compression, noise_sigma) #(nbases, nex)
#        y = np.dot(R, I).squeeze()  #(ncomp, nex)
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

        a = cs(I, Phi, R, compression, noise_sigma) #(nbases, nex)

        y = np.dot(R, I).squeeze()  #(ncomp, nex)
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
# Plotting and Image Functions
#==============================================================================


def plot_dict(Phi, n_plot=None, spacing=1, cmap=cm.gray, background=0):
    '''
    Given Phi (N inputs (pixels), M outputs (neurons)) plot the dictionaries

    Params
    ----
    n_plot
        number of dictionaries to plot (int)
    spacing
        number of pixels between plots (int)
    '''
    sz_N = np.ceil( np.sqrt(Phi.shape[0]) ).astype(int)
    sz_M = np.ceil( np.sqrt(Phi.shape[1]) ).astype(int)
    if not n_plot:
        n_plot = Phi.shape[1]

    pixels = sz_N*sz_M + (sz_M+1)*spacing
    image = np.zeros( (pixels, pixels) ) + background

    for i in range(sz_M):
        for j in range(sz_M):
            yb = i*( sz_N + spacing ) + spacing
            xb = j*( sz_N + spacing ) + spacing
            i_dict = i*sz_M + j            
            
            if i_dict < n_plot:
                # Only print if there is an image there
                image[ yb:yb+sz_N, xb:xb+sz_N ] = np.reshape( Phi[:,i_dict], (sz_N, sz_N) )

    plt.imshow(image, cmap=cmap, interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(left=False, right=False, top=False, bottom=False)
    plt.show()
    plt.draw()



def make_patches(image, patch_size, n_patches=False, random=False, BUFF=0):
    '''
    Break Image into patches. Assumes square patches.
    '''
    l_p = np.sqrt(patch_size).astype(int)
    l_im = np.sqrt(image.size).astype(int)
    border = np.ceil(BUFF/l_p)    
    

    if not n_patches:
        n_patches = image.size / patch_size
    
    n_patches=int(n_patches)
        
    patches = np.zeros([patch_size, n_patches])
    patch_list = list( itertools.product( np.arange( (l_im / l_p) - border ), repeat=2 ) )


    if random:
        np.random.shuffle(patch_list)
    
    if n_patches < (l_im / l_p)**2:
        patch_list = patch_list[:n_patches]
    
    for k, (i, j) in enumerate( patch_list ):
        r = int(i*l_p + BUFF)
        c = int(j*l_p + BUFF)
        patches[:, k] = image[r:r+l_p, c:c+l_p].ravel()  #Image Patch
    
    
    return patches
    
    
    
def make_image(patches):
    '''
    Reconstruct Image. Assumes square patches. 
    Image.size = [sqrt(n_patches) * len_patch, sqrt(n_patches) * len_patch] 
    '''
    patch_size, n_patches = patches.shape
    sz = np.sqrt(patch_size)
    imsize = sz * np.sqrt(n_patches)

    image = np.zeros((imsize, imsize))
    
    for k, (i, j) in enumerate( itertools.product( np.arange(imsize/sz), repeat=2 ) ):
        r = i*sz
        c = j*sz
        image[r:r+sz, c:c+sz]  = patches[:, k].reshape(sz,sz)  #Image Patch

    return image

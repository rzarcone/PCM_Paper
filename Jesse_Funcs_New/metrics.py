# 1) BER( Bits, SNR(peakpower, variance) )
# 2) C( BER(Bits, SNR) ) -> Channel Rate[devices/bit of info] with optimal code
# **3) JPEG R(D) -> Implement JPEG, compare to real deal for different 'quality' levels
# 	**-2D DCT 8x8
# 	**-Quantization (ISO Standard Matrices)
# 	**-Entropy Coding (Huffman + RLE)
# 4) R(D) for CS encoding of coefficients instead of Huffman/RLE
# 5) Use both NMSE(i.e. SNR) and NSSIM for distortion metric
# 
# Also,
# Refactor Capacity Code -> iPython Notebook for IEDM, Techcon

from __future__ import division
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter, uniform_filter

def nmse(img1, img2, dB=True):
    '''    
    Calculate the normalized mean square error between two images
        img1 : original image
        img2 : altered image
    '''
    NMSE = np.mean( (img1-img2)**2 ) / np.mean(img1**2)
    if dB:
        NMSE = 10 * np.log10(NMSE)
    return NMSE



def fspecial_gaussian(shape=(3,3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    
    eps = np.finfo(h.dtype).eps
    h[ h < eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
 
    return h



def mssim_matlab(img1, img2, K=np.r_[0.01, 0.03], window=fspecial_gaussian(shape=(11,11), sigma=1.5), L=255):
    '''
    Adapted from matlab version below by Jesse Engel 09/15/14:
    ========================================================================
    SSIM Index with automatic downsampling, Version 1.0
    Copyright(c) 2009 Zhou Wang
    All Rights Reserved.
    
    ----------------------------------------------------------------------
    Permission to use, copy, or modify this software and its documentation
    for educational and research purposes only and without fee is hereby
    granted, provided that this copyright notice and the original authors'
    names appear on all copies and supporting documentation. This program
    shall not be used, rewritten, or adapted as the basis of a commercial
    software or hardware product without first obtaining permission of the
    authors. The authors make no representations about the suitability of
    this software for any purpose. It is provided "as is" without express
    or implied warranty.
    ----------------------------------------------------------------------
    
    This is an implementation of the algorithm for calculating the
    Structural SIMilarity (SSIM) index between two images
    
    Please refer to the following paper and the website with suggested usage
    
    Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
    quality assessment: From error visibility to structural similarity,"
    IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,
    Apr. 2004.
    
    http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
    
    Note: This program is different from ssim_index.m, where no automatic
    downsampling is performed. (downsampling was done in the above paper
    and was described as suggested usage in the above website.)
    
    Kindly report any suggestions or corrections to zhouwang@ieee.org
    
    ----------------------------------------------------------------------
    
    Input : (1) img1: the first image being compared
           (2) img2: the second image being compared
           (3) K: constants in the SSIM index formula (see the above
               reference). defualt value: K = [0.01 0.03]
           (4) window: local window for statistics (see the above
               reference). default widnow is Gaussian given by
               window = fspecial_gaussian(11, 1.5)
           (5) L: dynamic range of the images. default: L = 255
    
    Output: (1) mssim: the mean SSIM index value between 2 images.
               If one of the images being compared is regarded as 
               perfect quality, then mssim can be considered as the
               quality measure of the other image.
               If img1 = img2, then mssim = 1.
           (2) ssim_map: the SSIM index map of the test image. The map
               has a smaller size than the input images. The actual size
               depends on the window size and the downsampling factor.
    
    ========================================================================
    '''
    if img1.shape != img2.shape:
        print 'Images must be the same shape'
        return

    M, N = img1.shape
    H, W = window.shape

    # # automatic downsampling
    f = max(1,round(min(M,N)/256))

    # #downsampling by f
    # #use a simple low-pass filter 
    if f>1:
        lpf = np.ones((f,f))
        lpf /= np.sum(lpf)
       
        img1 = convolve2d(img1,lpf,mode='same',boundary='symm')
        img2 = convolve2d(img2,lpf,mode='same',boundary='symm')
        
        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]


    C1, C2 = (K*L)**2
    window /= np.sum(window)
    
    # print window.shape, img1.shape, img2.shape

    mu1   = convolve2d(img1, window,  'valid')
    mu2   = convolve2d(img2, window, 'valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = convolve2d(img1*img2, window, 'valid') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = (denominator1*denominator2 > 0)
        ssim_map[index] = (numerator1[index]*numerator2[index])/(denominator1[index]*denominator2[index])
        index = (denominator1 != 0.) and (denominator2 == 0.)
        ssim_map[index] = numerator1(index)/denominator1(index)

    mssim = np.mean(ssim_map)

    return mssim, ssim_map





def mssim(img1, img2, dynamic_range=1, fast=True):
    """ 
    Computes mean structural similarity (MSSIM) index between two images.

    Implemented by K. Koepsell, 2013

    img1:
        original image
    img2:
        distorted image

    see Wang, Bovik, Sheikh, Simoncelli (2004): Image Quality Assessment: From
    Error Visibility to Structural Similarity
    """
    patch_sz = 11
    cp = patch_sz // 2

    assert img1.shape == img2.shape, "Images have to have the same shape"
    assert img1.shape[0] >= patch_sz, "Images need at least %d rows" % patch_sz
    assert img1.shape[1] >= patch_sz, "Images need at least %d cols" % patch_sz
    assert img1.max() <= dynamic_range and img2.max() <= dynamic_range

    # luminance similarity
    std = 1.5
    m1 = gaussian_filter(img1, std)[cp: -cp, cp: -cp]
    m2 = gaussian_filter(img2, std)[cp: -cp, cp: -cp]
    C1 = (0.01 * dynamic_range) ** 2
    l = (2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1)

    if fast:
        img10 = img1 - gaussian_filter(img1, 4)
        img20 = img2 - gaussian_filter(img2, 4)
        s1 = np.sqrt(gaussian_filter(img10 ** 2, std)[cp: -cp, cp: -cp])
        s2 = np.sqrt(gaussian_filter(img20 ** 2, std)[cp: -cp, cp: -cp])
        s12 = gaussian_filter(img10 * img20, std)[cp: -cp, cp: -cp]
    else:
        mean1 = uniform_filter(img1, patch_sz)
        mean2 = uniform_filter(img2, patch_sz)
        s1 = np.zeros_like(img1[cp: -cp, cp: -cp])
        s2 = np.zeros_like(img1[cp: -cp, cp: -cp])
        s12 = np.zeros_like(img1[cp: -cp, cp: -cp])
        for ri in xrange(patch_sz):
            rmax = ((img1.shape[0] - ri) // patch_sz) * patch_sz
            for ci in xrange(patch_sz):
                cmax = ((img1.shape[1] - ci) // patch_sz) * patch_sz
                idx = (slice(ri, ri + rmax), slice(ci, ci + cmax))
                img10 = img1[idx] - np.repeat(np.repeat(
                    mean1[cp + ri:-cp:patch_sz, cp + ci:-cp:patch_sz],
                    patch_sz, 0), patch_sz, 1)
                img20 = img2[idx] - np.repeat(np.repeat(
                    mean2[cp + ri:-cp:patch_sz, cp + ci:-cp:patch_sz],
                    patch_sz, 0), patch_sz, 1)
                std1 = np.sqrt(gaussian_filter(img10 ** 2, std))
                std2 = np.sqrt(gaussian_filter(img20 ** 2, std))
                var12 = gaussian_filter(img10 * img20, std)
                s1[ri::patch_sz, ci::patch_sz] = \
                    std1[cp:-cp:patch_sz, cp:-cp:patch_sz]
                s2[ri::patch_sz, ci::patch_sz] = \
                    std2[cp:-cp:patch_sz, cp:-cp:patch_sz]
                s12[ri::patch_sz, ci::patch_sz] = \
                    var12[cp:-cp:patch_sz, cp:-cp:patch_sz]

    # contrast similarity
    C2 = (0.03 * dynamic_range) ** 2
    c = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)

    # structural similarity
    C3 = C2 / 2
    s = (s12 + C3) / (s1 * s2 + C3)

    return (l * c * s).mean()

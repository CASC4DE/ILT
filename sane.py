#!/usr/bin/env python 
# encoding: utf-8

"""
sane.py
#########
Algorithm for denoising time series, named sane (standing for "Support Selection for Noise Elimination")

main function is 
sane(data, rank)
data : the series to be denoised
rank : the rank of the analysis

Copyright (c) 2015 IGBMC. All rights reserved.
Marc-Andr\'e Delsuc <madelsuc@unistra.fr>
Lionel Chiron <lionel.chiron@gmail.com>

This software is a computer program whose purpose is to compute sane denoising.

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

Created by Lionel Chiron and Marc-Andr\'e on september 2015.

version 1.0 
15/nov/2015
"""
from __future__ import division, print_function
import numpy as np
import numpy.linalg as linalg
from numpy.fft import fft, ifft


debug = 0 # put to 1 for debuging message
###################
#The following code allows to speed-up fft
# borrowed from scipy.signals
def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target
    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target
    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)
            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
#####################
def sane(data, k, orda = None, iterations = 1, trick = True, optk = False, ktrick = False):
    """ 
    sane algorithm. Name stands for Support Selection for Noise Elimination.
    From a data series return a denoised series denoised
    data : the series to be denoised - a (normally complex) numpy buffer
    k : the rank of the analysis
    orda : is the order of the analysis
        internally, a Hankel matrix (M,N) is constructed, with M = orda and N = len(data)-orda+1
        if None (default) orda = (len(data)+1)/2
    iterations : the number of time the operation should be repeated
    optk : if set to True will calculate the rank giving the best recovery for an automatic estimated noise level. 
    trick : permits to enhanced the denoising by using a cleaned signal as the projective space. "Support Selection"
    ktrick : if a value is given, it permits to change the rank on the second pass.
             The idea is that for the first pass a rank large enough as to be used to compensate for the noise while
             for the second pass a lower rank can be used. 
    
    ########
    values are such that
    orda <= (len(data)+1)//2
    k < orda
    N = len(data)-orda+1
    Omega is (N x k)
    
    Sane is based on the same idea than urQRd, however, there is a much clever selection of the basis on which the random projection is performed.
    This allows a much better recovery of small signals buried into the noise, compared to urQRd.
    
    the flags trick, optk, ktrick control the program
    when all are false, sane folds back to urQRd algorithm.
    Optimal is certainly trick = True, optk = True, ktrick = True  but not fully tested yet.

    ##########
    sane uses a new trick for performing a better denoising.
    A rank a little above the number of peaks as to be given. 
    this permit to make a filtering matrix containing the signal quasi only after a first pass.
    On a second pass the full signal is projected on a new filtering subspace done from preceding denoising.
    A higher number of iterations will decrease even more the smallest amplitudes. 
    ##########
    """
    if not orda:
        orda = (data.size)//2                                            # defining orda if not given.
    if optk:                                                            #  optimal rank, ensures that all the signal part is retrieved.
        optrank = OPTK(data, orda)
        k = optrank.find_best_rank()  
    if np.allclose(data, 0.0):                                           # dont do anything if data is empty
        return data
    #### optimize sizes to next regular
    L = len(data)
    if L>340:    # last big step is _next_regular(325) == 360; then max extension is 6%
        Lr = _next_regular(L)
        orda_r = 2*Lr - _next_regular(2*Lr-orda)
        if debug>0:
            if L != Lr or orda != orda_r:
                print("SANE regularisation %d %d => %d %d"%(L, orda, Lr, orda_r))
    else:
        Lr = L
        orda_r = orda
    if L != Lr:
        data_r = np.concatenate((data,np.zeros(Lr-L)))   # create a copy and add zero
    else:
        data_r = data       # just a link
    #####
    if (2*orda_r > data_r.size):                                            # checks if orda not too large.
        raise(Exception('order is too large'))
    #####
    if (k >= orda_r):                                                     # checks if rank not too large
#        print('type(k) ', type(k))
        raise(Exception('rank is too large, or orda is too small'))
    N = len(data_r)-orda_r + 1
    dd = data_r.copy()
    for i in range(iterations+1):
        if i == 1 and ktrick:
            Omega = np.random.normal(size = (N, ktrick))                            # Omega random real gaussian matrix Nxk
        else:
            Omega = np.random.normal(size = (N, k))                            # Omega random real gaussian matrix Nxk
        if i == 1 and trick:
            dataproj = data_r.copy()          # will project orignal dataset "data.copy()" on denoised basis "dd"
        else:    
            dataproj = dd.copy()            # Makes normal urQRd iterations, for sane_trick, it is the first passage.
        if trick:
            Q, QstarH = saneCore(dd, dataproj, Omega)                                # Projection :  H = QQ*H   
            dd = Fast_Hankel2dt(Q, QstarH)
        elif i != 1 and not trick:  # eliminate from classical urQrd the case i == 1.
            Q, QstarH = saneCore(dd, dataproj, Omega)                                # Projection :  H = QQ*H   
            dd = Fast_Hankel2dt(Q, QstarH)
    denoised = dd
    if data.dtype == "float":                                           # this is a kludge, as a complex data-set is to be passed - use the analytic signal if your data are real
        denoised = np.real(denoised)
    return denoised[:L]

def saneCore(dd, data, Omega):
    '''
    Core of sane algorithm
    '''
    Y =  FastHankel_prod_mat_mat(dd, Omega)
    Q, r = linalg.qr(Y)                                                  # QR decompsition of Y
    del(r)                                                              # we don't need it any more
    QstarH = FastHankel_prod_mat_mat(data.conj(), Q).conj().T# 
    return Q, QstarH                                                    # H approximation given by QQ*H    

def vec_mean(M, L):
    '''
    Vector for calculating the mean from the sum on the antidiagonal.
    data = vec_sum*vec_mean
    '''
    vec_prod_diag = [1/float((i+1)) for i in range(M)]
    vec_prod_middle = [1/float(M) for i in range(L-2*M)]
    vec_mean_prod_tot = vec_prod_diag + vec_prod_middle + vec_prod_diag[::-1]
    return np.array(vec_mean_prod_tot)

def FastHankel_prod_mat_mat(gene_vect, matrix):
    '''
    Fast Hankel structured matrix matrix product based on FastHankel_prod_mat_vec
    '''
    N,K = matrix.shape 
    L = len(gene_vect)
    M = L-N+1
    data = np.zeros(shape = (M, K), dtype = complex)
    for k in range(K):
        prod_vect = matrix[:, k] 
        data[:,k] = FastHankel_prod_mat_vec(gene_vect, prod_vect) 
    return data

def FastHankel_prod_mat_vec(gene_vect, prod_vect):
    """
    Compute product of Hankel matrix (gene_vect)  by vector prod_vect.
    H is not computed
    M is the length of the result
    """
    L = len(gene_vect)                                                  # length of generator vector
    N = len(prod_vect)                                                  # length of the vector that is multiplied by the matrix.
    M = L-N+1
    prod_vect_zero = np.concatenate((np.zeros(M-1), prod_vect[::-1]))   # prod_vect is completed with zero to length L
    fft0, fft1 = fft(gene_vect), fft(prod_vect_zero)                    # FFT transforms of generator vector and 
    prod = fft0*fft1                                                    # FFT product performing the convolution product. 
    c = ifft(prod)                                                      # IFFT for going back 
    return np.roll(c, +1)[:M]

def Fast_Hankel2dt(Q,QH):
    '''
    returning to data from Q and QstarH
    Based on FastHankel_prod_mat_vec.
    '''
    M,K = Q.shape 
    K,N = QH.shape 
    L = M+N-1
    vec_sum = np.zeros((L,), dtype = complex)
    for k in range(K):
        prod_vect = QH[k,:]
        gene_vect = np.concatenate((np.zeros(N-1), Q[:, k], np.zeros(N-1))) # generator vector for Toeplitz matrix
        vec_k = FastHankel_prod_mat_vec(gene_vect, prod_vect[::-1])         # used as fast Toeplitz
        vec_sum += vec_k 
    datadenoised = vec_sum*vec_mean(M, L)                                    # from the sum on the antidiagonal to the mean
    return datadenoised


 





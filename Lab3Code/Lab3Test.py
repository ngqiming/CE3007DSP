# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:45:55 2020

@author: NG QI MING
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack

                         
def myIDTFS(xdtfs, N):
    
    for k in np.arange(0:N):
        x
    
def myDTFS(ipX):
    N = len(ipX)
    C = np.zeros(shape=(N), dtype=complex)
    for k in np.arange(0, N):
        for n in np.arange(0, N):      
            C[k] = C[k] + ipX[n] * np.exp(-1j*(2*np.pi/N)*k*n)
    
    return C
    
        
    
def q2b():
    #define the x values
    ipX = [1,1,0,0,0,0,0,0,0,0,0,0]
    #Define Length of the fourier transform
    Xdtfs = myDTFS(ipX)
    x = fftpack.fft(ipX)
    #checking DTFS with build in function
    #print (Xdtfs)
    #print(x)
    #for IDTFS 
    xCheck = fftpack.ifft(Xdtfs)
    print(xCheck)



    
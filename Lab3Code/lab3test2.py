# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:24:39 2020

@author: NG QI MING
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import cmath
from scipy import signal

def myDTFS(ipX):
    N = len(ipX)
    c = np.zeros(shape=(N), dtype=complex)
    for k in np.arange(0, N):
        for n in np.arange(0, N):      
            c[k] = c[k] + ipX[n]*np.exp(-1j*(2*np.pi/N)*k*n)
    return c

def q2b():
    #define the x values
    ipX = [1,1,0,0,0,0,0,0,0,0,0,0]
    #Define Length of the fourier transform
    Xdtfs = myDTFS(ipX)
    #x = fftpack.fft(ipX)
    
    #for IDTFS 
    xn = myIDTFS(Xdtfs)
    #xCheck = fftpack.ifft(Xdtfs) 
    
  
    
    
def myIDTFS(Xdtfs):
    len_of_Xdtfs = len(Xdtfs)
    xn = [0] * len_of_Xdtfs
    for n in range(len(xn)):
        for k in range(len_of_Xdtfs):
            xn[n] += Xdtfs[k] * np.exp(1j*(2*np.pi/len_of_Xdtfs)*k*n)
    
    return xn

def myDFT(ipx):
    N = len(ipx)
    x = [0] *  N
    N = len(ipx)
    for k in np.arange(0, N):
        result = 0
        for n in np.arange(0, N):
            intermediate_result = 2 * np.pi / N * k * n
            result += ipx[n] * (np.cos(intermediate_result) - 1j * np.sin(intermediate_result))
        x[k] = result
    return x

def myIDFT(Xdtf):
    len_of_Xdtf = len(Xdtf)
    x = [0] * len_of_Xdtf
    for n in range(len(x)):
        result = 0
        for k in range(len_of_Xdtf):
            intermediate_result = 2 * np.pi / len_of_Xdtf * k * n
            result += Xdtf[k] * (np.cos(intermediate_result) + 1j * np.sin(intermediate_result))
        x[n] = result / len_of_Xdtf
    return x
    
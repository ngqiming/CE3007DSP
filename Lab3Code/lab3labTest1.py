# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:36:35 2020

@author: NG QI MING
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as signal

def myDTFS(ipX):
    N = len(ipX)
    Xdtfs = np.zeros(shape=(N), dtype=complex)
    for k in np.arange(0, N):
        for n in np.arange(0, N):      
            Xdtfs[k] = Xdtfs[k] + ipX[n]*np.exp(-1j*(2*np.pi/N)*k*n)
    return 1/N * Xdtfs
def myDFTConvolve (ipX,impulseH):
    
    lenipX = len(ipX)
    lenimpH = len(impulseH)
    for i in range(lenimpH-1):
        ipX.append(0)
    for i in range(lenipX-1):
        impulseH.append(0)
    
    return myIDFT(myDFT(ipX)*myDFT(impulseH))
def myDFT(ipX):
    N = len(ipX)
    Xdft = np.zeros(shape=(N), dtype=complex)
    for k in np.arange(0, N):
        for n in np.arange(0, N):      
            Xdft[k] = Xdft[k] + ipX[n]*np.exp(-1j*(2*np.pi/N)*k*n)
    return Xdft

def myIDFT(Xdft):
    N = len(Xdft)
    X_Idft = np.zeros(shape=(N), dtype=complex)
    for n in np.arange(0, N):
        for k in np.arange(0, N):      
            X_Idft[n] = X_Idft[n] + Xdft[k]*np.exp(1j*(2*np.pi/N)*k*n)
    return 1/N * X_Idft


def q1():
    ipX = [0,1,2,3,0,0,0,0]
    output = myDTFS(ipX)
    output1 = fftpack.fft(ipX)
    #k value 
    k = np.arange(0, len(output1))
    magnitude = np.absolute(output1)
    phase = np.angle(output)
    plt.figure()
    plt.stem(k,magnitude)
    plt.show()
    
    plt.figure()
    plt.stem(k,phase)
    plt.show()

def q1b():
    ipX = [0,1,2,3,0,0,0,0]
    output = fftpack.fft(ipX)
    xo = np.arange(0,360,45)
    magnitude = np.absolute(output)
    phase = np.angle(output)
    plt.stem(xo,magnitude)
    plt.show()
    
    plt.figure()
    plt.stem(xo,phase)
    plt.show()
    
def q3():
    x = [1,2,3,4,5,6,7]
    h = [1,1]
    
    y = np.convolve(x,h)
    print(y)
    
    myoutput = myDFTConvolve(x,h)
    print(myoutput)
    
    scipyoutput = signal.fftconvolve(x,h)
    print(scipyoutput)
    
    #linear convolution
   
    X = fftpack.fft(x)
    H = fftpack.fft(h)
    Y = np.multiply(X,H)
    tst_y = fftpack.ifft(Y)
    tst_y_abs = np.absolute(tst_y)
    print(tst_y_abs)
    
q3()
    
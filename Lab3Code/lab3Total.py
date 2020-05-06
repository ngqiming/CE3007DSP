# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:55:34 2020

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

def myIDTFS(Xdtfs):
    N = len(Xdtfs)
    X_Idtfs = np.zeros(shape=(N), dtype=complex)
    for n in np.arange(0, N):
        for k in np.arange(0, N):      
            X_Idtfs[n] = X_Idtfs[n] + Xdtfs[k]*np.exp(1j*(2*np.pi/N)*k*n)
    return X_Idtfs

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

def q2_b():
    ipX = [1,1,0,0,0,0,0,0,0,0,0,0]
    output = myDTFS(ipX)
    phase = np.angle(output)
    magnitude = np.absolute(output)
    n = np.arange(0, len(output))
    print(n)
    plt.figure()
    plt.stem(n,magnitude)
    plt.show()
    
    plt.figure()
    plt.stem(n,phase)
    plt.show()
   
def q2_c():
    ipX = [1,1,0,0,0,0,0,0,0,0,0,0]
    X = fftpack.fft(ipX)
    print(X)
    print(myIDTFS(myDTFS(ipX)))
    print(myDFT(ipX))
    print(myIDFT(myDFT(ipX)))
    
def q2_d():
    #get the 2b values
    ipX = [1,1,0,0,0,0,0,0,0,0,0,0]
    output = myDTFS(ipX)
    phase = np.angle(output)
    magnitude = np.absolute(output)
    n = np.arange(0, len(output))
    print(n)
    fig, axs = plt.subplots(2,1)
    plt.figure()
    axs[0].stem(n,phase*180/np.pi)
    axs[1].stem(n,magnitude)
    
    
    ipX2 = [0,1,1,0,0,0,0,0,0,0,0,0]
    output = myDTFS(ipX2)
    phase = np.angle(output)
    magnitude = np.absolute(output)
    n = np.arange(0, len(output))
    fig, axs = plt.subplots(2,1)
    plt.figure()
    axs[0].stem(n,phase*180/np.pi)
    axs[1].stem(n,magnitude)
    
    
    ipX3 = [10,10,0,0,0,0,0,0,0,0,0,0]
    output = myDTFS(ipX3)
    phase = np.angle(output)
    magnitude = np.absolute(output)
    n = np.arange(0, len(output))
    fig, axs = plt.subplots(2,1)
    plt.figure()
    #showing the phasor in degree (not radian)
    axs[0].stem(n,phase*180/np.pi)
    axs[1].stem(n,magnitude)
    plt.show()
def q_3():
    # Lab 3 Question 3, example code
    # showing how to use quiver to plot the arrow for
    # the Fourier basis at index k
    N = 32
    k=1
    W = np.zeros(shape=(N),dtype=complex)
    for n in np.arange(0,N):
        W[n] = np.exp(-1j*(2*np.pi/N)*k*n)
    
    W_angle = np.angle(W)
    # the lenbth is 1, we are only interested in the angle of each phasor
    
    plt.figure()
    plt.title('Each row shows the k-th harmonic, from n=0..N-1 index')
    Q = plt.quiver( np.cos(W_angle),np.sin(W_angle),  units='width')
    titleStr = 'Fourier complex vectors N='+str(N)
    plt.title(titleStr)
    plt.ylabel('k-values')
    plt.xlabel('n-values')
    plt.grid()
    plt.show()
    #showing the phasor in degree (not radian)
    print(np.angle(W)*180/np.pi)
    
def q_4():
    
    length=[12,24,47,96]
    for i in range(len(length)):
        N = length[i]
        ipX = np.zeros(shape=(N), dtype = complex)
        for i in range(0,7):
            ipX[i] = 1
        output = myDTFS(ipX)
        n = np.arange(0, len(output))
        omega = np.angle(output)
        magnitude = np.absolute(output)
        fig, axs = plt.subplots(2,1)
        plt.figure()
        axs[0].stem(n,magnitude, use_line_collection= True)
        axs[1].stem(n,omega, use_line_collection= True)
        plt.show()

def myDFTConvolve (ipX,impulseH):
    
    lenipX = len(ipX)
    lenimpH = len(impulseH)
    for i in range(lenimpH-1):
        ipX.append(0)
    for i in range(lenipX-1):
        impulseH.append(0)
    
    return myIDFT(myDFT(ipX)*myDFT(impulseH))

def q_5():
    
    # Lab 3 Question 5, example code
    # showing how to use fft to convolve
    # notice I DID not take care of the zero padding
    # you will need to to make it work!
    x = [1,2,3,4,5,6,7]
    h = [1,1,0,0,0,0,0]
    y = np.convolve(x,h)
    print(y)
    myoutput = myDFTConvolve(x,h)
    print(myoutput)
    
    scipyoutput = signal.fftconvolve(x,h)
    print(scipyoutput)
    
    # you should write your own fft and ifft routine!
    X = fftpack.fft(x)
    H = fftpack.fft(h)
    Y = np.multiply(X,H)
    tst_y = fftpack.ifft(Y)
    tst_y_abs = np.absolute(tst_y)
    print(tst_y_abs)
    
#q2_b()
#q2_c()   
#q2_d()
q_3()
#q_4()
#q_5()

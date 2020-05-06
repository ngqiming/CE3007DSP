# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:43:32 2020

@author: NG QI MING
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import winsound
# plotting 3D complex plane
from mpl_toolkits.mplot3d import Axes3D

#3.1a 
def GenSampledWave (A,F,Phi,Fs,sTime,eTime ):
    n = np.arange(sTime,eTime,1.0/Fs) #sampling period 
    y = A*np.cos(2 * np.pi * F * n + Phi)
    return [n,y]

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))
    

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))
    
 #3.1a   
def PlaySound():
     A=0.1; Phi = 0; Fs=16000; sTime=0; eTime = 0.4
     for F in range(1000, 19000, 1000):
         print("The value of f is {}".format(F))
         [n,yfloat] = GenSampledWave(A, F, Phi, Fs, sTime, eTime)
         
         # Although we created the signal in the date type float and dynamic range -1.0:1.0
         # when we save it and when we wish to listen to it using winsound it should be in 16 bit int.
         y_16bit = fnNormalizeFloatTo16Bit(yfloat)
         # Lets save the file, fname, sequence, and samplingrate needed
         wavfile.write('t1_16bit.wav', Fs, y_16bit)
         wavfile.write('t1_float.wav', Fs, yfloat)
         # Lets play the wavefile using winsound given the wavefile saved above
         winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
         print("The F value is: {}".format(F) )
         

def q31b():
    A = 0.5; Phi = np.pi/2; Fs = 16000; sTime=-0.01; eTime = 0.01
    #Generate 
    F = 1000 #change here for 17000
    plt.figure(1)
    graph = GenSampledWave(A, F, Phi, Fs, sTime, eTime)
    plt.plot(graph[0], graph[1] , 'r--o');
    plt.xlabel('sample index: nTs'); plt.ylabel('y[n]')
    plt.grid()
   
    
    plt.figure(2)
    no_of_samples = np.arange(0, 96, 1)
    plt.stem(no_of_samples, graph[1], 'g')
    plt.grid()
    plt.show()


def q3_2(seq = "0123456789*#", Fs = 10000, durTone= 0.5):
     
    freq = {'0': [1336, 941],
            '1': [1209, 697],
            '2': [1336, 697],
            '3': [1477, 697],
            '4': [1209, 770],
            '5': [1336, 770],
            '6': [1477, 770],
            '7': [1209, 852],
            '8': [1336, 852],
            '9': [1477, 852],
            '*': [1209, 941],
            '#': [1477, 941]
             }
     
    sTime = 0
    eTime = sTime + durTone
    yTotal = []
    
    for s in seq:
         
         n = np.arange(sTime,eTime,1.0/Fs)
         y = 0.5*np.sin(2 * np.pi * freq[s][0] * n)
         y1 = 0.5*np.sin(2 * np.pi * freq[s][1] * n)
         y3 = y + y1

         yTotal = np.concatenate((yTotal, y3))
         

    y_16bit = fnNormalizeFloatTo16Bit(yTotal)
    wavfile.write('t1_16bit.wav', Fs, y_16bit)
    winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    
def q3_4():
    numSamples = 18
    A=0.95; w=2*np.pi/18;
    n = np.arange(0, numSamples, 1)
    y1 = np.multiply(np.power(A, n), np.exp(1j * w * n))
    # plotting in 2-D, the real and imag in the same figure
    plt.plot(n, y1[0:numSamples].real,'r--o')
    plt.plot(n, y1[0:numSamples].imag,'g--o')
    plt.xlabel('n'); plt.ylabel('y[n]')
    plt.title('Complex exponential (red=real) (green=imag)')
    plt.grid()
    plt.show()
    
    
    # plotting in polar, understand what the spokes are
    plt.figure(2)
    for x in y1:
        plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
    
        plt.title('Polar plot showing phasors at n=0..N')
        plt.show()
    
    # plotting 3D complex plane
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    reVal = y1[0:numSamples].real
    imgVal = y1[0:numSamples].imag
    ax.plot(n,reVal, imgVal,  label='complex exponential phasor')
    ax.scatter(n,reVal,imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imag')
    ax.legend()
    plt.show()
    
def q3_5():
    numSamples = 16
    A = 1
    for k in range(4):
        n = np.arange(0, numSamples, 1)
        omega = (2 * np.pi) / numSamples
        
        y1= np.multiply(A, np.exp(1j * omega * k * n))
       
        # plotting in polar, understand what the spokes are 
        plt.figure(2)
        for x in y1:
            plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
        
        plt.title('Polar plot showing phasors at n=0..N')
        plt.show()
      
        # plotting 3D complex plane
        plt.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        reVal = y1[0:numSamples].real
        imgVal = y1[0:numSamples].imag
        ax.plot(n,reVal, imgVal,  label='complex exponential phasor')
        ax.scatter(n,reVal,imgVal, c='r', marker='o')
        ax.set_xlabel('sample n')
        ax.set_ylabel('real')
        ax.set_zlabel('imag')
        ax.legend()
        plt.show()
    
#PlaySound
#q31b()
#q3_2("8", 10000, 0.5)
q3_4()
#q3_5()
        

     
    
                   

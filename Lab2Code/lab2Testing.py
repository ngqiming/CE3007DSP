# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:56:42 2020

@author: NG QI MING
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import scipy
import winsound
from scipy import signal




# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnGenSampledSinusoid(A,Freq,Phi, Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    #return evenly spaced n value 
    n = np.arange(sTime,eTime,1.0/Fs) 
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]

def delta(n):
    if n== 0:
        return 1
    else:
        return 0
    
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))
    
def q1b():
      
    n = np.arange(0,50)
    xn = np.cos(0.2* np.pi * n)
    h = np.array([0.1, 0.2, 0.3])
    y = np.convolve(xn,h)
 
    fig, ax = plt.subplots(2, 1)
    ax[0].stem(xn,linefmt='g-',markerfmt='go',basefmt='r')
    ax[0].grid()
    ax[1].stem(y,linefmt ='b', markerfmt= 'bo',basefmt ='r')
    ax[1].grid()
    plt.ylabel('y[n]')
    plt.xlabel('sample n')
    plt.show()

def q3():
     impluseH = np.zeros(18000)
     impluseH[1] = 1
     impluseH[14000] = 0.5
     impluseH[17900] = 0.3
     
     ipcleanfilename = 'testIp_16bit.wav'
     winsound.PlaySound(ipcleanfilename, winsound.SND_FILENAME)
     [Fs, sampleX_16bit] = wavfile.read(ipcleanfilename)
     # we wish to do everything in floating point, hence lets convert to floating point first.
     sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)
     print("Convolving...")
     y = np.convolve(sampleX_float, impluseH)
     #change backing to 16bit
     y_16bit = fnNormalizeFloatTo16Bit(y)
     wavfile.write('t1_16bit.wav', Fs, y_16bit)
     print("Playing convolved wav file...")
     winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
     
     fig, ax = plt.subplots(2, 1) 
     ax[0].plot(sampleX_float)
     ax[0].grid()
     ax[1].plot(y)
     ax[1].grid()
     plt.show()
     

    
    
def qn4():
    
    #qn 4a
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    h3 = np.convolve(h1,h2)
    fig, ht = plt.subplots(2, 1) 
    ht[0].stem(h1,linefmt='b-',markerfmt='bo',basefmt='r')
    ht[0].grid()
    ht[1].stem(h2,linefmt='b-',markerfmt='bo',basefmt='r')
    ht[1].grid()
    plt.xlabel("n")
    plt.ylabel("h[n]")
    plt.show()
    
    #qn4b
    #get the range 
    n = np.arange(0,10)
    #get the xn full of 100 zeros
    xn=np.zeros(len(n))
    
    for i in range(len(n)):
        xn[i] = delta(n[i]) - 2 * delta(n[i] - 15)
    
    y1 = np.convolve(xn,h1)
    y2 = np.convolve(xn,h2)
    
    y3 = scipy.signal.lfilter(h1,[1],xn)
    y4 = scipy.signal.lfilter(h2,[1],xn)
    
    fig, ax = plt.subplots(4,1)
    ax[0].stem(y1,linefmt='b-',markerfmt='bo',basefmt='r')
    ax[1].stem(y3,linefmt='b-',markerfmt='bo',basefmt='r')
    ax[2].stem(y2,linefmt='b-',markerfmt='bo',basefmt='r')
    ax[3].stem(y4,linefmt='b-',markerfmt='bo',basefmt='r')
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.show()
    
    #qn4c
    x1n, x1 = fnGenSampledSinusoid(0.1, 700, 0, 16000, 0, 0.03)
    x2n, x2 = fnGenSampledSinusoid(0.1, 3333, 0, 16000, 0,0.03)
    #add 2 inputs tgt
    xTotal = x1 + x2
    y5 = np.convolve(xTotal,h1)
    y6 = np.convolve(xTotal,h2)
    ynew = np.convolve(xTotal,h3)
    
    x3float = fnNormalize16BitToFloat(xTotal)
    y5float = fnNormalize16BitToFloat(y5)
    y6float = fnNormalize16BitToFloat(y6)
    #qn4ci
    [f, t, Sxx_clean] = signal.spectrogram(x3float, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx_clean))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
    plt.show()
    #qn4ciii
    fig, ax = plt.subplots(4,1)
    ax[0].plot(x1n, x1)
    ax[0].grid()
    ax[1].plot(x2n, x2)
    ax[1].grid()
    ax[2].plot(ynew)
    ax[2].grid()
    ax[3].plot(y6)
    ax[3].grid()
    plt.show()
    
    
def qn4a():
    n = np.arange(0, 100)
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    h3 = np.convolve(h1, h2)

    x1 = np.zeros(len(n))
    for i in range(len(n)):
        x1[i] = delta(n[i]) - 2 * delta(n[i] - 15)
    y1 = np.convolve(x1, h1)
    y2 = np.convolve(x1, h2)

    x2n, x2 = fnGenSampledSinusoid(0.1, 700, 0, 16000, 0, 0.05)
    x3 = np.zeros(len(n))
    x3n, x3 = fnGenSampledSinusoid(0.1, 3333, 0, 16000, 0, 0.05)
    x4 = x2 + x3
    
    y3 = np.convolve(x4, h1)
    y4 = np.convolve(x4, h2)
    y5 = np.convolve(x4, h3)
    print("x4: ", x4)
    print("y5: ", y5)
    x4float = fnNormalize16BitToFloat(x4)
    output = fnNormalize16BitToFloat(y5)
    [f, t, Sxx] = signal.spectrogram(output, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
    plt.show()

    fig, axes = plt.subplots(6, 1)
    axes[0].plot(x2n, x2)
    axes[0].grid()
    axes[1].plot(x3n, x3)
    axes[1].grid()
    axes[2].plot(x3n, x4)
    axes[2].grid()
    axes[3].plot(y3)
    axes[3].grid()
    axes[4].plot(y4)
    axes[4].grid()
    axes[5].plot(y5)
    axes[5].grid()
    plt.show()

    # x2_16bit = helper.fnNormalizeFloatTo16Bit(x2)
    # wavfile.write('t1_16bit.wav', 16000, x2_16bit)
    # print("Playing x2_16bit wav file...")
    # winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    # x3_16bit = helper.fnNormalizeFloatTo16Bit(x3)
    # wavfile.write('t1_16bit.wav', 16000, x3_16bit)
    # print("Playing x3_16bit wav file...")
    # winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    x4_16bit = fnNormalizeFloatTo16Bit(x4)
    wavfile.write('t1_16bit.wav', 16000, x4_16bit)
    print("Playing x4_16bit wav file...")
    winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    # y3_16bit = helper.fnNormalizeFloatTo16Bit(y3)
    # wavfile.write('t1_16bit.wav', 16000, y3_16bit)
    # print("Playing y3_16bit wav file...")
    # winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    y4_16bit = fnNormalizeFloatTo16Bit(y4)
    wavfile.write('t1_16bit.wav', 16000, y4_16bit)
    print("Playing y4_16bit wav file...")
    winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    y5_16bit = fnNormalizeFloatTo16Bit(y5)
    wavfile.write('t1_16bit.wav', 16000, y5_16bit)
    print("Playing y5_16bit wav file...")
    winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
def qn5():
    
    #qn5a
    ipnoisyfilename = 'helloworld_noisy_16bit.wav'
    print("Playing original noisy wav file...")
    winsound.PlaySound(ipnoisyfilename, winsound.SND_FILENAME)
    Fs, sampleX_16bit = wavfile.read(ipnoisyfilename)
    x = fnNormalize16BitToFloat(sampleX_16bit)
    [f, t, Sxx_clean] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx_clean))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
    plt.show()
    
    #qn5c
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    y = np.zeros(len(x), dtype=float)
    y_ifil = signal.lfilter(B, A, x)

    for n in range(len(x)):
         if n == 0:
            y[n] = 1 * x[n]
         
         elif n == 1:
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] - (-0.722744) * y[n - 1]
    
         elif n == 2:
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] + 0.99999 * x[n - 2] - (-0.722744) * y[n - 1] - 0.888622 * y[
                n - 2]
    
   
    #for i in range(len(y)):
      #  if y[i] == y_ifil[i]:
       #     continue
       # else:
        #    print("=================")
         #   print(y[i])
          #  print(y_ifil[i])
           # print("=================")
    #5d
    #remove the noise
    y_ifil_16bit =  fnNormalizeFloatTo16Bit(y_ifil)
    wavfile.write('t1_16bit.wav', 16000, y_ifil_16bit)
    print("Playing modified wav file...")
    winsound.PlaySound("t1_16bit.wav", winsound.SND_FILENAME)
    
    [f, t, Sxx] = signal.spectrogram(y_ifil, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of modified signal')
    plt.show()
    
    
    
def quiz():
    x1n, x1 = fnGenSampledSinusoid(0.1, 1000, 0, 8000, 0, 1)
    x2n, x2 = fnGenSampledSinusoid(0.1, 1500, 0, 8000, 0, 1)
    x3 = x1 + x2
    x3_16bit =fnNormalizeFloatTo16Bit(x3)
    wavfile.write('t1_16bit.wav', 8000, x3_16bit)
    print("Playing original wav file...")
    winsound.PlaySound("t1_16bit.wav", winsound.SND_FILENAME)

    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    x_ifil = signal.lfilter(B, A, x3)
    x_ifil_16bit = fnNormalizeFloatTo16Bit(x_ifil)
    wavfile.write('t1_16bit.wav', 8000, x_ifil_16bit)
    print("Playing processed wav file...")
    winsound.PlaySound("t1_16bit.wav", winsound.SND_FILENAME)

    [f, t, Sxx] = signal.spectrogram(x_ifil, 8000, window=('blackmanharris'), nperseg=512, noverlap=int(0.9 * 512))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
    plt.show()
    
    
    
    
            

            

     
#q1b()
#q3()     
#qn4()
qn5()
quiz()


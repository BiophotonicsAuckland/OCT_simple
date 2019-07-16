# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:20:25 2019

@author: mzly903
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# File directory
filepath = r'C:\Users\mzly903\Desktop\Work\OCT\data4'

filepathDispersion = filepath

# Linearization and dispersion compensation files (Uncomment if different)
filepathDispersion = r'C:\Users\mzly903\Desktop\1704\magda'

fileName = input('Enter file name: ')

# Change names if linearization was changed
fileName_lin_phase = 'lin_st54_len1888'
fileName_nonlin_phase = 'nonlin_st54_len1888'
fileName_disp = 'disp_st0_len1888'

#Zero-padding

Zero = int(input('If you want to use zero-padding enter the value: ') or '1') #if you want to try use 2048 first  

#File uploading 
fileSpectrum = np.loadtxt(filepath + '\\' + fileName + '.txt') # OCT data
linPhase = np.loadtxt(filepathDispersion + '\\' + fileName_lin_phase + '.txt') # linearization data
nonlinPhase = np.loadtxt(filepathDispersion + '\\' + fileName_nonlin_phase + '.txt') # linearization data
dispersionCompensation = np.loadtxt(filepathDispersion + '\\' + fileName_disp + '.txt') # dispersion data

#Enter parameters
rawNumber = int(input('Enter which Ascan you want to build from the file:' ) or "1")
startSpectrum = int(input('Enter start point from %s ' % fileName_lin_phase))
lengthSpectrum = int(input('Enter lenth you use from %s ' % fileName_lin_phase))

#Set spectrum
spectrum = fileSpectrum [rawNumber, startSpectrum:startSpectrum+lengthSpectrum]

#interpolation
interpolation = interp1d(nonlinPhase, spectrum, fill_value='extrapolate')
spectrumLinearized  = interpolation(linPhase)

# compensate dispersion
spectrumLinearizedCompensated =spectrumLinearized*np.exp(-1j*dispersionCompensation)

# Build Ascan
Ascan = np.abs(np.fft.fft(spectrumLinearizedCompensated, n=Zero))

#Plot Ascan
plt.figure()
plt.xlabel('Intensity')
plt.ylabel('Optical distance')
plt.plot(Ascan, 'r')
#plt.plot(f, 'b', linewidth=1)
plt.axis('tight')

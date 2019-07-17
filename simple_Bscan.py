# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:51:18 2019

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


#File uploading 
spectra = np.loadtxt(filepath + '\\' + fileName + '.txt') # OCT data
linPhase = np.loadtxt(filepathDispersion + '\\' + fileName_lin_phase + '.txt') # linearization data
nonlinPhase = np.loadtxt(filepathDispersion + '\\' + fileName_nonlin_phase + '.txt') # linearization data
dispersionCompensation = np.loadtxt(filepathDispersion + '\\' + fileName_disp + '.txt') # dispersion data

#Zero-padding
Zero = int(input('If you want to use zero-padding enter the value: ') or '1') #if you want to try use 2048 first  

#cut spectra
countAscans = len(spectra)
firstAscan = int(input('If you want to cut spectra what will be first Ascan: ') or '1')
lastAscan = int(input('If you want to cut spectra what will be last Ascan: ') or countAscans)

#Build Bscan
arrayBscan = []
for i in range(firstAscan, lastAscan):
    spectrum = spectra [i, 54 : 54+1888]
    interpolation = interp1d(nonlinPhase, spectrum, fill_value='extrapolate')
    spectrumLinearized = interpolation(linPhase)

# compensate dispersion
    spectrumLinearizedCompensated=spectrumLinearized*np.exp(-1j*dispersionCompensation)
    Ascan = np.abs(np.fft.fft(spectrumLinearizedCompensated, n=Zero))
    arrayBscan.append(Ascan)

#plot image
plt.figure()
plt.imshow(np.rot90(arrayBscan,1), cmap='binary')
plt.clim([0, 8000])


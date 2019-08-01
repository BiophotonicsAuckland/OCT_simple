
"""
Created on Tue Jul 30 08:06:57 2019

@author: mzly903
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import pyplot, transforms


# File directory
filepath_res = r'C:\Users\mzly903\Desktop\Work\OCT\data5'
name = 'Saphire0deg_v2.txt'

# Linearization and dispersion compensation files (Uncomment if different)

filepath2 = r'C:\Users\mzly903\Desktop\1704\magda'
filename_lin_phase = 'lin_st54_len1888'
filename_nonlin_phase = 'nonlin_st54_len1888'
filename_disp = 'disp_st0_len1888'

#File uploading 
phase_lin = np.loadtxt(filepath2 + '\\' + filename_lin_phase + '.txt')
phase_nonlin = np.loadtxt(filepath2 + '\\' + filename_nonlin_phase + '.txt')
dispCompVect = np.loadtxt(filepath2 + '\\' + filename_disp + '.txt')
spectrum = np.loadtxt(filepath_res + '\\' + name)

#Gaussian
Low_wavelength_cut = 780
High_wavelength_cut = 920
#central_wavelength1 = int(input('Please enter first central wavelength peak from 780 to 920: '))
central_wavelength1 = 800
#central_wavelength2 = int(input('Please enter first central wavelength peak from 780 to 920: '))
central_wavelength2 = 900
vector_length = len(dispCompVect)

x = np.arange(-5, 5, 10/vector_length) #gaussian range
mu1 = -5+(central_wavelength1-Low_wavelength_cut)/((High_wavelength_cut-Low_wavelength_cut)/10)  # 
mu2 = -5+(central_wavelength2-Low_wavelength_cut)/((High_wavelength_cut-Low_wavelength_cut)/10)  
sigma = 0.8
gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )

# Just to try

data_raw = 21  # for files with big data set. For single Ascan files get rid of data_row everywhere


#Walk-off
Zero = 65536

spectrum_f = spectrum [data_raw, 54 : 54+1888]
spectrum1 = gaussian1*spectrum [data_raw, 54 : 54+1888]
spectrum2 = gaussian2*spectrum [data_raw, 54 : 54+1888]

f = interp1d(phase_nonlin, spectrum_f, fill_value='extrapolate')
f1 = interp1d(phase_nonlin, spectrum1, fill_value='extrapolate')
f2 = interp1d(phase_nonlin, spectrum2, fill_value='extrapolate')

spectrum_lin = f(phase_lin)
spectrum_lin1 = f1(phase_lin)
spectrum_lin2 = f2(phase_lin)

# compensate dispersion
spectrum_lin_disp =spectrum_lin*np.exp(-1j*dispCompVect)
spectrum_lin_disp1 =spectrum_lin1*np.exp(-1j*dispCompVect)
spectrum_lin_disp2 =spectrum_lin2*np.exp(-1j*dispCompVect)

absFFT = np.abs(np.fft.fft(spectrum_lin_disp, n=Zero))
absFFT1 = np.abs(np.fft.fft(spectrum_lin_disp1, n=Zero))
absFFT2 = np.abs(np.fft.fft(spectrum_lin_disp2, n=Zero))

#plot
plt.figure(37)
#plt.imshow(np.rot90(my_list,1), cmap='binary')
#plt.plot(my_list5, my_list2, 'r', linewidth=1.5)
plt.xlabel('Optical distance')
plt.ylabel('Intensity')
plt.title('Dispersion Walk-off')
plt.plot(absFFT/10, 'k', label="Dispersion Walk-off", linewidth=1)
plt.plot(absFFT1, 'b', label="Dispersion Walk-off", linewidth=1)
plt.plot(absFFT2, 'r', label="Dispersion Walk-off", linewidth=1)
y = np.arange(Low_wavelength_cut, High_wavelength_cut, (High_wavelength_cut-Low_wavelength_cut)/vector_length)
plt.figure(38)
plt.plot(y, spectrum[data_raw, 54 : 54+1888])
plt.plot(y, 1000*gaussian1, 'b')
plt.plot(y, 1000*gaussian2, 'r')
plt.xlabel('Wavelength nm')
plt.axis('tight')

plt.grid()

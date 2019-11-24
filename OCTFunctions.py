# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:38:10 2019

@author: mzly903
"""

import threading
import scipy.constants
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import peakutils
import scipy
from scipy.ndimage import gaussian_filter
import mahotas as mh
from IPython.html.widgets import interact, fixed
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
filepath2 = r'C:\Users\mzly903\Desktop\Work\OCT\3. 2016-11-09 Experiment'
filepath3 = r'C:\Users\mzly903\Desktop\Work\OCT\3. 2016-11-09 Experiment'
filename_lin_phase = 'lin_st100_len1509'
filename_nonlin_phase = 'nonlin_st100_len1509'
filename_disp = 'disp_st0_len1509'

#File uploading 
phase_lin = np.loadtxt(filepath2 + '\\' + filename_lin_phase + '.txt')
phase_nonlin = np.loadtxt(filepath2 + '\\' + filename_nonlin_phase + '.txt')
dispCompVect = np.loadtxt(filepath2 + '\\' + filename_disp + '.txt')


pretty_data = []

p=18
number_ascans = 190
Zero = 2**p # 2**11=2048 no zero padding
cut_DC =  (2**(p-10))*21
def click():
    root = Tk()
    
    root.withdraw() 
    print ('choose file:')
    name = askopenfilename() 
    print(name + ' is selected')
    spectrum = np.loadtxt(name)
    return spectrum, name

def calibrate(origin, disp = True):
    if disp == False:
        spectrum_data = origin [105:105+len(phase_lin)]

        f = interp1d(phase_nonlin, spectrum_data, fill_value='extrapolate')
        
        spectrum_lin_disp = f(phase_lin)
        
    else:
        spectrum_data = origin [105:105+len(phase_lin)]
        f = interp1d(phase_nonlin, spectrum_data, fill_value='extrapolate')
        spectrum_lin = f(phase_lin)
        spectrum_lin_disp =spectrum_lin *np.exp(-1j*dispCompVect)
    return spectrum_lin_disp


def ascan(enter_data):
    absFFT = np.abs(np.fft.fft(enter_data, n = Zero))
    return absFFT


def bscan(y, first_ascan = 0, last_ascan = number_ascans, go = 0):

    data_bscan = []
    
    
    for g in range (first_ascan, last_ascan):
        spectrum = y [g]
        absFFT = np.abs(np.fft.fft(spectrum, n=Zero))
        if go == 'gauss':
            absFFT = gaussian_filter(np.abs(np.fft.fft(spectrum, n=Zero)),sigma = 18)
        #data_bscan.append(absFFT[int(len(absFFT)/1.25):len(absFFT)-cut_DC])
        data_bscan.append(absFFT)
        procent_old = round((g-1)*100/len(y))
        procent_new = round((g)*100/len(y))
    
        if procent_new > procent_old :
            print ('Please wait ' + str(100 - procent_new) + '% left')
            if procent_new == 94:
                print ('Almost there')
    return np.rot90(data_bscan,1)

Low_wavelength_cut = 780
High_wavelength_cut = 920

central_wavelength1 = 810
central_wavelength2 = 890

resolution =(2.5/(2**(p-11)))*10**(-6) # resolution in mm where 2.5 is a pix 

def dispersion(spectrum, ref_ind=1.508):
    x = np.arange(-5, 5, 10/len(spectrum)) #gaussian range

    mu1 = -5+(central_wavelength1-Low_wavelength_cut)*10/(High_wavelength_cut-Low_wavelength_cut)  # 
    mu2 = -5+(central_wavelength2-Low_wavelength_cut)*10/((High_wavelength_cut-Low_wavelength_cut))  
    
    sigma = 0.8
    
    gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
    gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )
    ###### Algorithm starts
    
    
    spectrum1 = gaussian1*spectrum
    spectrum2 = gaussian2*spectrum
    
    absFFT1 = np.flip(np.abs(np.fft.fft(spectrum1, n=Zero)))
    absFFT2 = np.flip(np.abs(np.fft.fft(spectrum2, n=Zero)))
    

   
    
    cut = int(len(absFFT1)/2)

    index1 = peakutils.indexes(absFFT1[cut_DC:cut], thres=0.2, min_dist=2*(2**(p-10))) 
    index2 = peakutils.indexes(absFFT2[cut_DC:cut], thres=0.2, min_dist=2*(2**(p-10))) 
    
    #print ('z1 is ' + str(index1) + '\n z2 is ' + str(index2))
    c = scipy.constants.c # in m/s 

    omega1 = 2*np.pi*c/(central_wavelength1)  # wavelength in nm  
    omega2 = 2*np.pi*c/(central_wavelength2)  # wavelength in nm
    l_s = np.abs(index1[0]-index1[-1])* resolution*ref_ind  # in mm
    z_1 = (index1[-1])*resolution #in m
    z_2 = (index2[-1])*resolution
    walk_off = np.abs(z_2-z_1)*ref_ind
    
    beta_2 =(walk_off/(c*l_s*(omega1-omega2)))*10**(24)

    return walk_off 


def dispersionplus(spectrum, ref_ind=1.508):
    x = np.arange(-5, 5, 10/len(spectrum)) #gaussian range

    mu1 = -5+(central_wavelength1-Low_wavelength_cut)*10/(High_wavelength_cut-Low_wavelength_cut)  # 
    mu2 = -5+(central_wavelength2-Low_wavelength_cut)*10/((High_wavelength_cut-Low_wavelength_cut))  
    
    sigma = 0.8
    
    gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
    gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )
    ###### Algorithm starts
    
    
    spectrum1 = gaussian1*spectrum
    spectrum2 = gaussian2*spectrum
    

    f1 = interp1d(phase_nonlin, spectrum1, fill_value='extrapolate')
    f2 = interp1d(phase_nonlin, spectrum2, fill_value='extrapolate')

    spectrum_lin1 = f1(phase_lin)
    spectrum_lin2 = f2(phase_lin)

# compensate dispersion

    #spectrum_lin_disp1 =spectrum_lin1*np.exp(-1j*dispCompVect)
    #spectrum_lin_disp2 =spectrum_lin2*np.exp(-1j*dispCompVect)    
    
    absFFT1 = np.flip(np.abs(np.fft.fft(spectrum_lin1, n=Zero)))
    absFFT2 = np.flip(np.abs(np.fft.fft(spectrum_lin2, n=Zero)))
    plt.figure()

    plt.plot(spectrum, c = 'r', linewidth=0.5)    
   
    cut_DC =  (2**(p-10))*21
    cut = int(len(absFFT1)/2)

    index1 = peakutils.indexes(absFFT1[cut_DC:cut], thres=0.9, min_dist=2*(2**(p-10)))  
    index2 = peakutils.indexes(absFFT2[cut_DC:cut], thres=0.9, min_dist=2*(2**(p-10))) 
    
    index_first = peakutils.indexes(absFFT1[cut_DC:int(cut-cut/2)], thres=0.9, min_dist=2*(2**(p-10))) 
    index_last = peakutils.indexes(absFFT1[int(cut-cut/2):cut], thres=0.9, min_dist=2*(2**(p-10))) 

    #print ('z1 is ' + str(index1) + '\n z2 is ' + str(index2))
    c = scipy.constants.c # in m/s 

    omega1 = 2*np.pi*c/(central_wavelength1)  # wavelength in nm  
    omega2 = 2*np.pi*c/(central_wavelength2)  # wavelength in nm
    l_s = (np.abs(index_first-index_last)*resolution)*10**6
    print ('distance' + str(l_s))
    #l_s = 975  # in mm
    z_1 = (index1[-1])*resolution #in m
    z_2 = (index2[-1])*resolution
    walk_off = np.abs(z_2-z_1)*ref_ind
    
    beta_2 =(walk_off/(c*l_s*(omega1-omega2)))*10**(24)

    return beta_2

def showmegraph(ascan_data, title='title is missing', full = True):

    plt.figure()
    plt.title(title)
    cut_DC =  (2**(p-10))*21
    cut = int(len(ascan_data)/2)
    if full == False:
        plt.plot(ascan_data[cut_DC:cut], c = 'r', linewidth=0.5)
    else:
        plt.plot(ascan_data, c = 'r', linewidth=0.5) 
    plt.axis('tight')


def showmegraphplus(ascan_data, title='title is missing', full = True):

    plt.figure()
    plt.title(title)
    cut_DC =  (2**(p-10))*21
    cut = int(len(ascan_data)/2)
    if full == False:
        fig, ax = plt.subplots()
        ax.plot(ascan_data[cut_DC:cut], c = 'r', linewidth=0.5)
        axins = zoomed_inset_axes(ax, 1.5, loc=2) # zoom-factor: 2.5, location: upper-left
        axins.plot(ascan_data[cut_DC:cut])
        x1, x2, y1, y2 = 120, 160, 4, 6 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        
        
    else:
        plt.plot(ascan_data, c = 'r', linewidth=0.5) 
    plt.axis('tight')
        


def showmeimage(bscan_data,  title='title for B-scan is missing', full = True):

    plt.figure()
    if full == True:
        plt.imshow(bscan_data, cmap='binary')
    else:
        plt.imshow(bscan_data[:int(len(bscan_data)/3)], cmap='binary')
    plt.clim([0, 14000])
    plt.title(title)
    plt.axis('tight')


def showmehist (data_hist, title_hist = 'Title is missing'):
    plt.hist(data_hist, alpha=0.5, histtype='bar', ec='black')
    st_dev = np.std(data_hist)
    ave = np.mean(data_hist)
    plt.title(str(title_hist) + '\n Standart deviation: %.2f \n Average: %.2f [$fs^2/mm$] ' %(st_dev, ave))
    
    plt.show()
    
def showme3D (cscan, title_hist = 'Title is missing'):
    plt.hist(cscan, alpha=0.5, histtype='bar', ec='black')
    st_dev = np.std(cscan)
    ave = np.mean(cscan)
    plt.title(str(title_hist) + '\n Standart deviation: %.2f \n Average: %.2f [$fs^2/mm$] ' %(st_dev, ave))
    
    plt.show()
    
    
def segmentation (image):

    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    

    sure_bg = cv2.dilate(opening,kernel,iterations=500)
    

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
    

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    

    ret, markers = cv2.connectedComponents(sure_fg)
    

    markers = markers+1
    
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    
    plt.imshow(img)
    
def first_peak (ascan):
    initial = ascan[0]*6
    for i in range (0, len(ascan)):
        value = ascan[i]        
        if initial < value:
            break
    return i
            

def bscan_fast(y, first_ascan = 0, last_ascan = number_ascans, go = 0):

    data_bscan = []
    cut_DC =  (2**(p-10))*21
    
    for g in range (first_ascan, last_ascan):
        spectrum = y [g]
        absFFT = np.abs(np.fft.fft(spectrum, n=Zero))
        if go == 'gauss':
            absFFT = gaussian_filter(np.abs(np.fft.fft(spectrum, n=Zero)),sigma = 18)
        #data_bscan.append(absFFT[int(len(absFFT)/1.25):len(absFFT)-cut_DC])
        data_bscan.append(absFFT)
        procent_old = round((g-1)*100/len(y))
        procent_new = round((g)*100/len(y))
    
        if procent_new > procent_old :
            print ('Please wait ' + str(100 - procent_new) + '% left')
            if procent_new == 94:
                print ('Almost there')
                
    
    return np.rot90(data_bscan,1) 
   

    
    
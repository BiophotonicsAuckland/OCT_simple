# -*- coding: utf-8 -*-
"""
Created on Thu May 7 18:59:23 2020

@author: mzly903
"""

# Short cuts for many things in OCT data processing
import OCTFunctions as octf

import numpy as np
# stable version 1.17.4 

from matplotlib import pyplot as plt
# stable version 3.1.1 

import scipy.stats as stats # gaussian

import csv # work with csv files to save data
# stable version 1.0


import glob

import datetime

time = str(datetime.datetime.now())
date_time = time[5:19].replace(":", "-")

class DataProcessing:

    """ All data preprocessing features """


    def __init__(self, data, data_name):

        self.data = data
        self.data_name = data_name


    def fourrier(self, Zero = 2048, first_ascan = 0, last_ascan = None, absolute = True):
        """" 
        if absolute == True: Returns a depth profile; 
        otherwise just Fourrier transform 
        """
        if last_ascan == None:
            last_ascan = len(self.data)

        if isinstance(self.data[0], int) or isinstance(self.data[0], float):
            depth_profile = np.fft.fft(self.data, n = Zero)
            if absolute:
                depth_profile = abs(depth_profile)

        else:
            depth_profile = []

            for i in range(first_ascan, last_ascan):
                fourrier_raw = np.fft.fft(self.data[i], n=Zero)
                if absolute:
                    fourrier_raw = abs(fourrier_raw)
                depth_profile.append(fourrier_raw)

        return depth_profile


    def gauss_window(self, cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920):
        
        """" 
        Returns a list with Gaussian distribution
        
        """
        
        mu = cw
        x = np.linspace(wavelength_from, wavelength__to, pixels)
        gaussian_distribution = stats.norm.pdf(x, mu, sigma)
        max_ = max(gaussian_distribution)
        for l in range(len(gaussian_distribution)):
            gaussian_distribution[l] = gaussian_distribution[l]/max_

        return gaussian_distribution



    def apply_gauss_window(self, cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920 ):
        """" 
        Returns a spectrum multiplied by a Gaussian window
        
        """
        if isinstance(self.data[0], int) or isinstance(self.data[0], float): 
            after_gaussian_spectrum = self.data*self.gauss_window(cw, sigma, pixels, wavelength_from, wavelength__to)

        else:
            after_gaussian_spectrum = []
            for i in range(len(self.data)):
                new_ascan = self.data[i]*self.gauss_window(cw, sigma, pixels, wavelength_from)
                after_gaussian_spectrum.append(new_ascan)

        return after_gaussian_spectrum
    
    
""" Helps the programme to understand where is the object    
Includes methods with semi-automatic visualisation        
"""    
class ComputerVision:
   
    def __init__(self, fourrier_data, data_name):
        
        self.fourrier_data = fourrier_data
        self.name = data_name
        
    def geometrical_line(self):
        
        bscan = self.fourrier_data[:]
        
        l_values = []
        
        for ii in range(0,4):
            crop_raw_left = bscan[ii][int(len(bscan[0])/20):int(len(bscan[0])/3)]
            value = np.argmax(crop_raw_left) + int(len(bscan[0])/20)
            l_values.append(value)
        r_values = []
        for jj in range(len(bscan)-4, len(bscan)):
            crop_raw_right = bscan[jj][int(len(bscan[0])/20):int(len(bscan[0])/3)]
            value = np.argmax(crop_raw_right)  + int(len(bscan[0])/20)
            r_values.append(value)
        
        l_values.remove(max(l_values))
        l_values.remove(min(l_values))
        
        r_values.remove(max(r_values))
        r_values.remove(min(r_values))
        
        left_point = int(sum(l_values)/len(l_values))
        right_point = int(sum(r_values)/len(r_values))

        line = [left_point] 

        line_sp = np.linspace(left_point, right_point, len(bscan)-2)
        line.extend(line_sp)
        line.append(right_point)
        
        return line


    def threshlold_segmentation(self, thr_value = 0.67, smoothing = True):
        """"
        Algoritm cuts first 5% of the depth profile(to get rid of noise peak)
        Algoritm cuts second half of the depth profile(to get rid of mirror image)
            UPDATES: CUTS last 2/3, spheroid is located on the first 1/3 of the image 
    
        
        Algoritm sets a threshold value (thr_value*100)% of the highest peak in an Ascan
        
        Algoritm goes from the beginning of croped image (or Ascan) to 
        the point threshold value
        
        Algoritm skips noise points before the surface:
            It counts if "above threshold" point has been met skip_pixels times in succession.
            where skip_pixels = 0.045% of the length of initial data raw (e.g for 4096 pixels depth-profile = 2 pixels, for 65536 = 30)
        
        WHEN "above threshhold" value has been met skip_pixels times it returns coordinate of the first "correct" surface
        
        In the same way bottom surface is detected. The cropped raw is turned around
        
        
        """

        abscan = self.fourrier_data[:]


        if isinstance(abscan[0], int) or isinstance(abscan[0], float):
            skip_pixels = int(len(abscan)*0.0005 + 1) # (for 4096 pix: skips 2 pixels, for 65536 skips 33 pixels )
            
            for_segmentation = abscan[:int(len(abscan)/3)] # crop mirror effect
            for_segmentation = for_segmentation[int(len(for_segmentation)/30):] # crop peak at the beggining
            threshold = thr_value*max(for_segmentation)
            i = 0
            check_value = 0
            confidence_interval = 0
            while confidence_interval < skip_pixels and i < len(for_segmentation):
                check_value = for_segmentation[i]
                
                if check_value > threshold:
                    confidence_interval += 1
                else:
                    confidence_interval = 0 # if the next point is below threshold
                i += 1
            top_surface = int(len(for_segmentation)/30) + i - skip_pixels

            ############## Bottom surface #################

            for_b_segmentation = for_segmentation[:(i-skip_pixels):-1] # reverse data raw and slice untill the top surface peak
            
            threshold = thr_value*max(for_b_segmentation)
            i = 0
            check_value = 0
            confidence_interval = 0
            while confidence_interval < skip_pixels and i < len(for_b_segmentation):
                check_value = for_b_segmentation[i]
    
                if check_value > threshold:
                    confidence_interval += 1
                else:
                    confidence_interval = 0 # if the next point is below threshold
                i += 1
                
            bottom_surface = int(len(abscan)/3) - i + skip_pixels

        else:
            
            """
            
            Define the line of petry dish 
            
            Top surface is always above this line
            
            """

            line = self.geometrical_line()
            
            """ image surface detection"""
            
            top_surface = []
            bottom_surface = []
            skip_pixels = int(len(abscan[0])*0.0007 + 1) # (for 4096 pix: skips 2 pixels, for 65536 skips 33 pixels )

            for y in range(len(abscan)):

                for_segmentation = abscan[y][int(len(abscan[0])/30):int(line[y])] # consider points only untill the petri dish
                threshold = thr_value*max(for_segmentation)
                i = 0
                check_value = 0
                confidence_interval = 0
                
                
                while confidence_interval < skip_pixels  and i < len(for_segmentation):
                    check_value = for_segmentation[i]
                    
                    if check_value > threshold:
                        confidence_interval += 1 # to ignore noise points before the surface 
                    # else:
                    #    confidence_interval = 0 # if the next point is below threshold
                    i += 1
                detected_point_one_raw = int(len(abscan[0])/30) + i - skip_pixels
                top_surface.append(detected_point_one_raw)
                
                ################## Bottom surface ########### 
                
                for_b_segmentation = abscan[y][int(line[y]):int(len(abscan[y])/3)] # reverse data raw and slice untill the top surface peak
                for_b_segmentation = for_b_segmentation[::-1]
                
                thr_value = 0.66
                threshold = thr_value*max(for_b_segmentation)
                i = 0
                check_value = 0
                confidence_interval = 0
                while confidence_interval < (skip_pixels-10) and i < len(for_b_segmentation):
                    check_value = for_b_segmentation[i]
        
                    if check_value > threshold:
                        confidence_interval += 1
                    else:
                        confidence_interval = 0 # if the next point is below threshold
                    i += 1
                    
                detected_point_b_surface = int(len(abscan[0])/3) - i + skip_pixels
                bottom_surface.append(detected_point_b_surface)
                
                
        ############### Smoothing ############
        if smoothing:
            
            confidence =  top_surface[0]*0.02      
            smooth_top = octf.are_neighbors_okay(top_surface, confidence)
            for l in range(len(smooth_top)):
                if smooth_top[l] == False:
                    top_surface[l] = (top_surface[l-1]+top_surface[l+1])/2
            smooth_bottom =  octf.are_neighbors_okay(bottom_surface, confidence)
            for e in range(len(bottom_surface)):
                if smooth_bottom[e] == False:
                    bottom_surface[e] = (bottom_surface[e-1]+bottom_surface[e+1])/2      
                
        return (top_surface, bottom_surface, line) 



    def show(self, first_ascan = 0, last_ascan = None):
        """" 
        if absolute == True: Returns depth profile; 
        otherwise just Fourrier transform 
        """
        if last_ascan == None:
            last_ascan = len(self.fourrier_data)

        if isinstance(self.fourrier_data[0], int) or isinstance(self.fourrier_data[0], float):
            plt.figure()
            plt.plt(self.fourrier_data)
            plt.title(self.name)
        else:
            plt.figure()
            plt.imshow(np.rot90(self.fourrier_data), cmap="gray")
            plt.clim([0, 10000])
            plt.axis("tight")
            plt.title(self.name)
    
    def where_to_crop(self):
        
        """ 
        Returns 2 coordinates 
        
        Left-upper and right-lower ((x1, x2, y1, y2))
        
        of the cut-window
        
        """
        

        self.show()
        crop_window = plt.ginput(2)
        x1 = int(crop_window[0][0])
        x2 = int(crop_window[1][0])
        y1 = int(crop_window[0][1])
        y2 = int(crop_window[1][1])

        return (x1, x2, y1, y2)

    def crop_image(self):
        
        """ 
        Returns cropped image data raws (pixels)
        
        """
        l = self.where_to_crop()
        x_axis_cut = np.array(self.fourrier_data[l[0]:l[1]])

        # return x_axis_cut[:, l[2]:l[3]]
        return x_axis_cut[:]


class DataFeautures(ComputerVision):
    
    
    def __init__(self, data_raw, name):
        
        self.data_raw = data_raw
        self.name = name
    
    def build_line(point1, point2):
        line = np.linspace(int(point1), int(point2), int(point2[0]) - int(point1[0] - 1))
        return line
    
    def peak_detection(self, from_ = 0, to = None):
        if to == None:
            to = len(self.data_raw)
        peak = np.argmax(self.data_raw[from_:to])
        return peak
    

    def refractive_index(peak1, peak2, geometrical_point):
        return (peak2-peak1)/(geometrical_point-peak1)

def peak_detection(row, from_ = 0, to = None):
    if to == None:
        to = len(row)
    peak = from_ + np.argmax(row[from_:to])
    return peak        

def geom_line(bscan):

    l_values = []
    
    for ii in range(0,4):
        crop_row_left = bscan[ii][int(len(bscan[0])/20):int(len(bscan[0])/3)]
        value = np.argmax(crop_row_left) + int(len(bscan[0])/20)
        l_values.append(value)
    r_values = []
    for jj in range(len(bscan)-4, len(bscan)):
        crop_row_right = bscan[jj][int(len(bscan[0])/20):int(len(bscan[0])/3)]
        value = np.argmax(crop_row_right)  + int(len(bscan[0])/20)
        r_values.append(value)
    
    l_values.remove(max(l_values))
    l_values.remove(min(l_values))
    
    r_values.remove(max(r_values))
    r_values.remove(min(r_values))
    
    left_point = int(sum(l_values)/len(l_values))
    right_point = int(sum(r_values)/len(r_values))

    line = [left_point] 

    line_sp = np.linspace(left_point, right_point, len(bscan)-2)
    line.extend(line_sp)
    line.append(right_point)
    
    return line

def vertical_line(x, y):
    print(x)
    print(y)
    return [[x]*int(y), [y]*int(y)]

def refractive_ind(peak1, peak2, geometrical_point):
    return (peak2-peak1)/(geometrical_point-peak1)

def norm_inten(data_row):
    print("start")
    return [float(i)*40/max(data_row) for i in data_row]

def norm_gvd(data1, data2, peak1, peak2):
    dif = int((int(peak1) - int(peak2)))
    l = -abs(dif)
    if dif >= 0:
        mod_data1 = data1[:l]
        mod_data2 = data2[dif:]
    else:
        mod_data1 = data1[abs(dif):]
        mod_data2 = data2[:]
    return [mod_data1, mod_data2, dif]

plt.close("all")

### Load all d (day 7) spheroids ###

files = glob.glob('C:/Users/mzly903/Desktop/PhD/2. Data/1. Spheroids/Experiment 4 21 Jan 2017/21JanMeasurements/d*.txt')
with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
    newFileWriter = csv.writer(newFile)
    newFileWriter.writerow(['name', 'Ascan', "cw1", "cw2" ,"sigma" ,'ri',
                            "Oprtical_distance1","Oprtical_distance2",'walks normalized',
                            'Geometrical Distance', 'Optical Distnace', ''])
for yu in files[44:45]:
# upload file and get its name
    plt.close("all")
    file = np.loadtxt(yu)
    name = yu[-10:]

    zero_pad_image = 15  #### power degree of the zero padding values eg 2**15
    proccessed_data = DataProcessing(file, name)  #### create a DataProcessing object
    bscan = proccessed_data.fourrier(Zero = 2**zero_pad_image) #### apply method
    recognition = ComputerVision(bscan, name) #### create a ComputerVision object
   
    b = np.array(recognition.where_to_crop()) ## recognition part
    line_geo = geom_line(bscan[b[0]: b[1]]) ## geometrical line for RI
#make sure the image is ok   
    plt.figure()

    plt.imshow(np.rot90(bscan[b[0]:b[1]]), cmap = "gray")
    plt.clim([0, 10000])
    plt.axis("tight")
# draw a line for geometrical thickness
    plt.plot(line_geo)
    plt.title("Make sure line is properly detected and click anywhere")
# this ginput just to stop a program and see the image
    ttt = plt.ginput()

    
    print(ttt)
    if ttt[0][0] < 3:
        print("Bscan repeat", yu)
        filename =  'dataf1.txt'   

        with open(filename, "w+") as json1:
            text = "Repeat " + str(yu)
            json1.write(text)

        continue
        
# declare new object of CombuterVision class
    croped_signal = ComputerVision(bscan[b[0]:b[1]], name)
# get surface interpolation points to show on the Ascan    
    geometrical_points = croped_signal.geometrical_line()
    
    print(name)
    #### Work with every Ascan to collect as much data as possible ####
    sigma1 = 30
    sigma2 = 15
    cw_l1 = 810
    cw_l2 = 830
    cw_h1 = 880
    cw_h2 = 870
    
    def g_window(cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920):
        
        """" 
        Returns a list with Gaussian distribution
        
        """
        
        mu = cw
        x = np.linspace(wavelength_from, wavelength__to, pixels)
        gaussian_distribution = stats.norm.pdf(x, mu, sigma)
        max_ = max(gaussian_distribution)
        for l in range(len(gaussian_distribution)):
            gaussian_distribution[l] = gaussian_distribution[l]/max_

        return gaussian_distribution
    show_gauss1 = g_window(cw_l1, sigma1)
    show_gauss2 = g_window(cw_l2, sigma1)
    show_gauss3 = g_window(cw_h1, sigma1)
    show_gauss4 = g_window(cw_h2, sigma1)
    show_gauss5 = g_window(cw_l1, sigma2)
    show_gauss6 = g_window(cw_l2, sigma2)
    show_gauss7 = g_window(cw_h1, sigma2)
    show_gauss8 = g_window(cw_h2, sigma2)
    
    for_check = 77
    check = file[for_check]
    x = np.linspace(766.8, 920, 2048)
    for yt in range(len(check)):
        check[yt] = check[yt]/max(check)
    plt.figure()
    plt.plot(x,check)
    plt.plot(x,show_gauss1, "b")
    plt.plot(x,show_gauss2, "b")
    plt.plot(x,show_gauss3, "r")
    plt.plot(x,show_gauss4, "r")
    plt.plot(x,show_gauss5, "b")
    plt.plot(x,show_gauss6, "b")
    plt.plot(x,show_gauss7, "r")
    plt.plot(x,show_gauss8, "r")
    plt.axvline(x=cw_l1, c = "b")
    plt.axvline(x=cw_l2, c = "b")
    plt.axvline(x=cw_h1, c = "r")
    plt.axvline(x=cw_h2, c = "r")
    plt.title(f"Spectrum {for_check} and all Gaussian Windows")
    
    for ascan in range(b[0], b[1]):
        
        
        current_ascan = bscan[ascan] # work on each Ascan
        
        octf.showme(current_ascan, name + str(ascan)) # show Ascan
        cut_noise = int(len(bscan[ascan])/100) # get rid of artificial first peak 
        print("Argmax", cut_noise +  np.argmax(current_ascan[cut_noise:int(cut_noise*50)]),"Line", geometrical_points[ascan - b[0]])

        specrum = DataProcessing(file[ascan], name) # new object of DataProcessing class

        wave1 = specrum.apply_gauss_window(cw = cw_l1, sigma=sigma1) #gaussian 1
        wave2 = specrum.apply_gauss_window(cw = cw_h1, sigma=sigma1) #gaussian 2
        wave3 = specrum.apply_gauss_window(cw = cw_l2, sigma=sigma1) #gaussian 1
        wave4 = specrum.apply_gauss_window(cw = cw_h2, sigma=sigma1) #gaussian 2
        
        wave12 = specrum.apply_gauss_window(cw = cw_l1, sigma=sigma2) #gaussian 1
        wave22 = specrum.apply_gauss_window(cw = cw_h1, sigma=sigma2) #gaussian 2
        wave32 = specrum.apply_gauss_window(cw = cw_l2, sigma=sigma2) #gaussian 1
        wave42 = specrum.apply_gauss_window(cw = cw_h2, sigma=sigma2) #gaussian 2
        
        profile = DataProcessing(file[ascan], name) # depth profile full spectrum
        
        profile1 = DataProcessing(wave1, name) # depth profile gaussain 1
        profile2 = DataProcessing(wave2, name)# depth profile gaussain 2
        
        profile3 = DataProcessing(wave3, name) # depth profile gaussain 1
        profile4 = DataProcessing(wave4, name)# depth profile gaussain 2
        
        profile12 = DataProcessing(wave12, name) # depth profile gaussain 1
        profile22 = DataProcessing(wave22, name)# depth profile gaussain 2
        
        profile32 = DataProcessing(wave32, name) # depth profile gaussain 1
        profile42 = DataProcessing(wave42, name)# depth profile gaussain 2
        
        # Increase zero padding 
        zero_pad_graph = 19
        
        graph = profile.fourrier(Zero = 2**zero_pad_graph) 
       
        graph1 = profile1.fourrier(Zero = 2**zero_pad_graph)
        graph2 = profile2.fourrier(Zero = 2**zero_pad_graph)
        
        graph3 = profile3.fourrier(Zero = 2**zero_pad_graph)
        graph4 = profile4.fourrier(Zero = 2**zero_pad_graph)
        
        graph12 = profile12.fourrier(Zero = 2**zero_pad_graph)
        graph22 = profile22.fourrier(Zero = 2**zero_pad_graph)
        
        graph32 = profile32.fourrier(Zero = 2**zero_pad_graph)
        graph42 = profile42.fourrier(Zero = 2**zero_pad_graph)
        
        ##### Cut first (noise) peak and mirror half of the Ascan
        noise_level = int(len(graph)/100)
        graph = graph[noise_level:noise_level*50]
        graph1 = graph1[noise_level:noise_level*50]
        graph2 = graph2[noise_level:noise_level*50]
        
        graph3 = graph3[noise_level:noise_level*50]
        graph4 = graph4[noise_level:noise_level*50]

        graph12 = graph12[noise_level:noise_level*50]
        graph22 = graph22[noise_level:noise_level*50]
        
        graph32 = graph32[noise_level:noise_level*50]
        graph42 = graph42[noise_level:noise_level*50]        
        
        # New zero padding requires new miltiplication factor for actual (geometrical) position of the petri dish surface     

        vl2 = geometrical_points[ascan - b[0]]*(2**(zero_pad_graph-zero_pad_image))
        vl2 = vl2 - noise_level # to take into account "cut first (noise) peak"
        
        # show Ascan + line of actual surface (line on the graph)
        octf.showme(graph, name + "\n Ascan " + str(ascan) + "\n To discard click on the value above 1/2 max")
        plt.plot(graph2)
        plt.plot(graph1)
        
        
        # draw line of the surface
        plt.axvline(x=vl2, c = "r")
        # select 2 regions (4 points - one "from" another "to" for each region)what you want to zoom
        c_f_p_big = plt.ginput(4)
        if c_f_p_big[0][1] > max(graph1)*0.75: # click on the top of the graph if you don't want to save
            continue
        
        f_peak_ri = peak_detection(graph,from_ =  int(c_f_p_big[0][0]), to = int(c_f_p_big[1][0]))
        l_peak_ri = peak_detection(graph,from_ =  int(c_f_p_big[2][0]), to = int(c_f_p_big[3][0]))
        
        ri = refractive_ind(f_peak_ri, l_peak_ri, vl2)
        print("refractive index", ri)
        if ri > 1.44 or ri < 1.34:
            continue        
        
        # show first region (usually first peak)

        plt.figure()
        plt.plot(graph1[int(c_f_p_big[0][0]):int(c_f_p_big[1][0])])
        plt.plot(graph2[int(c_f_p_big[0][0]):int(c_f_p_big[1][0])])
        plt.title(f"Ascan {ascan} from file {name} \n to discard click on x > 2/3 len and y > 1/2 max")
        # select first peak region or skip ascan
        c_f_p = plt.ginput(2)
        length = len(graph1[int(c_f_p_big[0][0]):int(c_f_p_big[1][0])])
        max_ = max(graph1[int(c_f_p_big[0][0]):int(c_f_p_big[1][0])])
        if c_f_p[0][0] > length*0.7 and c_f_p[0][1] > max_*0.75:
            print(f"Ascan {ascan} skipped")
            continue
        # detect first peak for Full spectrum, wave1 and wave2
        f_peak = int(c_f_p_big[0][0]) + peak_detection(graph,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak1 = int(c_f_p_big[0][0]) + peak_detection(graph1,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak2 = int(c_f_p_big[0][0]) + peak_detection(graph2, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak3 = int(c_f_p_big[0][0]) + peak_detection(graph3,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak4 = int(c_f_p_big[0][0]) + peak_detection(graph4, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
           
        f_peak12 = int(c_f_p_big[0][0]) + peak_detection(graph12,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak22 = int(c_f_p_big[0][0]) + peak_detection(graph22, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak32 = int(c_f_p_big[0][0]) + peak_detection(graph32,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        f_peak42 = int(c_f_p_big[0][0]) + peak_detection(graph42, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
           
            
        
        # show second region (usually second peak)
        # n_graph = norm_inten(graph[int(c_f_p_big[2][0]):int(c_f_p_big[3][0])])

        plt.figure()
        # plt.plot(n_graph)
        plt.plot(graph1[int(c_f_p_big[2][0]):int(c_f_p_big[3][0])])
        plt.plot(graph2[int(c_f_p_big[2][0]):int(c_f_p_big[3][0])])
        plt.title(f"Ascan {ascan} from file {name} \n to discard click on x > 2/3 len and y > 1/2 max")
         # select bottom peak region or skip ascan
        c_f_p = plt.ginput(2)
        length = len(graph1[int(c_f_p_big[2][0]):int(c_f_p_big[3][0])])
        max_ = max(graph1[int(c_f_p_big[2][0]):int(c_f_p_big[3][0])])
        if c_f_p[0][0] > length*2/3 and c_f_p[0][1] > max_*0.7:
            print(f"Ascan {ascan} skipped")
            continue
    # detect bottom peak for Full spectrum, wave1 and wave2
        l_peak = int(c_f_p_big[2][0])  + peak_detection(graph,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak1 = int(c_f_p_big[2][0])  + peak_detection(graph1,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak2 = int(c_f_p_big[2][0])  + peak_detection(graph2, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak3 = int(c_f_p_big[2][0])  + peak_detection(graph3,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak4 = int(c_f_p_big[2][0])  + peak_detection(graph4, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))

        l_peak12 = int(c_f_p_big[2][0])  + peak_detection(graph12,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak22 = int(c_f_p_big[2][0])  + peak_detection(graph22, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak32 = int(c_f_p_big[2][0])  + peak_detection(graph32,from_ =  int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        l_peak42 = int(c_f_p_big[2][0])  + peak_detection(graph42, from_ = int(c_f_p[0][0]), to = int(c_f_p[1][0]))
        
    # normalization gvd
        gvd_graph = norm_gvd(graph1, graph2, f_peak1, f_peak2)
  # make sure everything is okay graph       

        plt.figure()
        plt.plot(gvd_graph[0])
        plt.plot(gvd_graph[1])
     # refr index   

         # walk-off 
        
        optical_dist_full = (l_peak - f_peak)
        
        optical_dist_l1 = (l_peak1 - f_peak1)
        optical_dist_h1 = (l_peak2 - f_peak2)
        optical_dist_l2 = (l_peak3 - f_peak3)
        optical_dist_h2 = (l_peak4 - f_peak4)

        optical_dist_l3 = (l_peak12 - f_peak12)
        optical_dist_h3 = (l_peak22 - f_peak22)
        optical_dist_l4 = (l_peak32 - f_peak32)
        optical_dist_h4 = (l_peak42 - f_peak42)
        
        walk_off1 = optical_dist_h1 - optical_dist_l1
        
        walk_off2 = optical_dist_h2 - optical_dist_l2
        
        walk_off3 = optical_dist_h3 - optical_dist_l3
        
        walk_off4 = optical_dist_h4 - optical_dist_l4


        print('Ascan:', ascan)
        print('RI:', ri)
        # save = input("to save press Y: ")
        save = 'n'
        if save.lower() == 'y':
            with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([name, ascan, cw_l1, cw_h1, sigma1,
                                        ri,optical_dist_l1, optical_dist_h1,
                                        walk_off1, vl2 - f_peak,
                                        l_peak - f_peak])
                
            with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([name, ascan,cw_l2, cw_h2, sigma1, ri,
                                        optical_dist_l2, optical_dist_h2,
                                        walk_off2, vl2 - f_peak,
                                        l_peak - f_peak])
            with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([name, ascan, cw_l1, cw_h1, sigma2, ri,
                                        optical_dist_l3, optical_dist_h3,
                                        walk_off3, vl2 - f_peak,
                                        l_peak - f_peak])
            with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([name, ascan, cw_l2, cw_h2, sigma2,  ri,
                                        optical_dist_l4, optical_dist_h4,
                                        walk_off4, vl2 - f_peak,
                                        l_peak - f_peak])
 

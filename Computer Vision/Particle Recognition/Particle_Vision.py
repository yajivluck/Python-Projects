#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:51:10 2023


This program accesses images from a folder denoted by "source_path" where 
png images of microphones after being exposed to particles are found. It uses
computer vision to contour particles, determines their area, and approximate a 
radius for each of them by using their area and doing a particle point approximation.


This program takes images, grey-scales them, thresholds them in a binary fashion (black/white) and then finds contours
of the clusters that were above the thresholded in color. Finally, it applies said contours to images to see what is actually 
being measured, and outputs information such as the area of each clusters for each images as well as the diameter of the circle
approximation of each clusters for each images. In this context, a cluster is recognized as a particle on the microphone.

@author: yajivluck
@email: rohaan.luckheenarain@mail.mcgill.ca
"""

import cv2
import imageio
import os
import glob
import math
import csv
import pandas as pd
import numpy as np

#To open images for N seconds and close them all after
N = 5 #Seconds to keep images opened
N = N * 1000
#Threshold parameter for small contours to be discarded
epsilon = 0.1
#Path to folder where images to be analyzed are
source_path = '/Users/yajivluck/Desktop/Alena_Particle_Vision/Images/*'
#Path where diameter files will be saved to 
output_path = '/Users/yajivluck/Desktop/Alena_Particle_Vision/Diameter_Image/'
PIXEL_RESOLUTION = 1; #TO MODIFY, THIS VALUE TRANSLATES PIXEL VALUES TO REAL WORLD MEASUREMENTS BASED ON THE MICROSCOPE

#Retrieve an array where each element is an image from the path given
def getImgData():
    paths = [os.path.splitext(os.path.basename(path))[0] for path in glob.glob(source_path)]
    imgData = [cv2.imread(path) for path in glob.glob(source_path)]     
    return imgData,paths
 
#Retrieve array of filenames for files used where program is being run
def getFileName():
     paths = [os.path.basename(path) for path in glob.glob(source_path)] #Array of filenames
     return paths
 
#Display all image in Images for N seconds
def displayImages(images_array, N):
    #Display all images 
    for i,image in enumerate(images_array):
        cv2.imshow(str(i), image)   
    #Wait N seconds before removing image window
    cv2.waitKey(N*1000)
    cv2.destroyAllWindows()
    
#Indefinite show of image
def showimage(images_array):
    #Display all images 
    for i,image in enumerate(images_array):
        cv2.imshow(str(i), image)
        

#Get information on contours of current image
def get_contours(pair):
    original_image, thresholded_image = pair
    # Find contours in the input image
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
    return contours, hierarchy


#Draw contours on original image from contour taken in thresholded. Takes pair of original/thresh as argument
def draw_contours(pair):
    information = []
    original_image, thresholded_image = pair
    contours, hierarchy = get_contours(pair)
    #Build information array of contours for each images
    information.append((contours,hierarchy))
    # Make a copy of the input image
    image_copy = original_image.copy()
    # Draw the contours on the copy image in green color
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return image_copy, information
    

#Function that will return the area of each particle from image (contour approx)
def get_contour_area(contours):
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        #Convert area from getContour to actual area based on pixel_resolution
        area = area * (PIXEL_RESOLUTION**2)
        areas.append(area)
    return areas

#Function given the area of a "particle", approximate the radius by assuming the area
#Corresponds to a circle's area
def approximate_diameter(areas, epsilon):
    diameters = []
    for area in areas:
        #radius = math.sqrt( ( area / (4 * math.pi) ) )  #Sphere approximation in 3D
        radius = math.sqrt( ( area / math.pi ) )  #Circle approximation in 2D
        if radius < epsilon: 
            continue
        diameter = 2 * radius
        diameters.append(round(diameter,1)) #Build diameter array with one decimal precision (Alena Request)
    #A = pi * r^2 (2D)
    #A = 4 * pi * r^2 (3D)
    return diameters #Diameter


def save_lists_as_csv(lists, names):
    for i, diameter_list in enumerate(lists):
        #Name of file being saved
        name = output_path + "D_"+ os.path.basename(names[i]) + '.csv'
        diameter_list_reshaped = np.array(diameter_list).reshape(-1, 1)
        df = pd.DataFrame( diameter_list_reshaped , columns = ['Diameters'])
        df.index.name = os.path.basename(names[i]) + " Diameters" # Set dataframe title
        df.to_csv(name, index=False) # Save dataframe to CSV file
        print(f'Saved {df.index.name} to {name}')
    return df
        
#Array where each entry is one of the images in the directory


#Main function call, gets images, processes them, and outputs particle csv file accordingly
def main():
    #Images from Directory and their path names
    original_image,original_names = getImgData() 
    #Grey Scaled Images
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in original_image] 
    #Black and white Images
    thresholded_images = [cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1] for img in gray_images] 
    #Array of double (original,thresholded)
    original_and_thresh = zip(original_image,thresholded_images)
    #Array where each elements are (image with contours, contour information)
    total_information = [draw_contours(image_thresh) for image_thresh in original_and_thresh]
    #Respective contains images with contour drawn on them, information about said contours
    contour_images, contour_information = zip(*total_information)
    #Array where each element is an array that contains the areas of each contours of the appropriate image
    contour_area = [get_contour_area(contours[0][0]) for contours in contour_information]
    #Array where each element is a list of diameters for particles of the corresponding image
    contour_approx_diam = [approximate_diameter(areas, epsilon) for areas in contour_area]
    save_lists_as_csv(contour_approx_diam, original_names)



    ##Display Images To see What they look like during processing
    #displayImages(original_image, 5)      #ORIGINAL
    #displayImages(gray_images, 5)        #GREY SCALED
    #displayImages(thresholded_images, 5) #BLACK AND WHITE
    #displayImages(contour_images, 5)      #ORIGINAL WITH CONTOURS



    ###IMPORTANT USE THIS LINE WHENEVER TO REMOVE IMAGES THAT AREN'T LEAVING

    #cv2.destroyAllWindows()
    
    

if __name__ == "__main__":
    main()




















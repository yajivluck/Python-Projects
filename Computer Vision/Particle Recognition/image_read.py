# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:17:50 2022

@author: Alena Gracheva
"""

import numpy as np
import cv2
import skimage
#from skimage import io, color, measure
#import matplotlib.pyplot as plt
from scipy import ndimage
#import csv




#step 1: Image reading
#fname="data/05_1.png"
fname = "data/05_1.png"
image = cv2.imread(fname)



#Step 1.5: Resize the image for better readability
image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5)


#step 2: Determine the image scale
# mouse call back function
def click_event(event, x, y,
        flags, params):

  # if the left button of mouse
  # is clicked then this
  # condition executes
  if event == cv2.EVENT_LBUTTONDOWN:
  
    # appending the points we
    # clicked to list
    points.append((x,y))
    
    # marking the point with a circle
    # of center at that point and
    # small radius
    cv2.circle(img,(x,y), 4,
        (0, 255, 0), -1)
    
    # if length of points list
    # greater than2 then this
    # condition executes
    if len(points) >= 2:
    
      # joins the current point and
      # the previous point in the
      # list with a line
      cv2.line(img, points[-1], points[-2],
          (0, 255, 255), 1)
      
    # displays the image
    cv2.imshow('image', img)
    
# making an black image
# of size (512,512,3)
# create 3-d numpy
# zeros array
img = image

# declare a list to append all the
# points on the image we clicked
points = []



# show the image
cv2.imshow('image',img)

# setting mouse call back
cv2.setMouseCallback('image',
          click_event)

# no waiting
cv2.waitKey(0)

# To close the image
# window that we opened
cv2.destroyAllWindows()#step 2: Define the scale
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        cv2.circle(image,(x,y), 4, (0, 255, 0), -1)
        if len(points) >= 2:
            cv2.line(image, points[-1], points[-2],(0, 255, 255), 1)
            cv2.imshow('image', image)

point1 = points[-1]
point2 = points[-2]
dist = ((point1[0]-point2[0])**2+(point1[0]-point2[0])**2)**(0.5)

#Scale of the image
print('disctance between points in pixels of the image', dist)
# Put finger width in um
finger_width = 8
pixels_to_um = finger_width/dist
print('scale of the image',pixels_to_um)
#step N: wait time
cv2.imshow('processed image',image)
cv2.waitKey(0)




#Step 3: Define the amount and the size of particles
new_shape = (image.shape[0] // 5, image.shape[1] // 5, image.shape[2])
small = skimage.transform.resize(image=image, output_shape=new_shape)
small = skimage.img_as_ubyte(small)

A=image
B=image.shape
print(type(B))
a,b,c=B
print(a,b,c)
print(type(a))

C=np.zeros((a,b,3),dtype=int)

C=A[:,:,:3]


H=0
G=0
F=0

for i in A:
    for j in i:
        F=F+j[0]
        G=G+j[1]
        H=H+j[2]
F=F/(B[0]*B[1]) 
G=G/(B[0]*B[1])
H=H/(B[0]*B[1])     
print("F=",F," ","G=",G," ","F=",F)
      
                    


total_pix=0
total_dust=0

for ii in C:

    for jj in ii:
        total_pix=total_pix+1
        
        if jj[0]<=F*0.65:
            if jj[1]<=G*0.65:
                if jj[2]<=H*0.65:
                    
                    jj[0]=255
                    jj[1]=255
                    jj[2]=255
                    total_dust=total_dust+1
                else: 
                    jj[0]=0
                    jj[1]=0
                    jj[2]=0
            else: 
                jj[0]=0
                jj[1]=0
                jj[2]=0
        else: 
             jj[0]=0
             jj[1]=0
             jj[2]=0
             
  
                    
print("total_dust=",total_dust)                  
print("total_pix=",total_pix)
rat_dust=total_dust/total_pix*100
print("dirt percantage =",rat_dust,"%")

label_im, nb_labels = ndimage.label(C)

 
sizes_pix = ndimage.sum_labels(C, label_im, range(1,nb_labels + 1))
sizes = sizes_pix*(pixels_to_um**2)/765

print('number of partcles:', nb_labels)

part_diameters = (sizes*4/3.14)**(0.5)

print('size of dust particles in pixels: ',sizes_pix)
print('size of dust particles in um: ',sizes)
print('diameter of dust particles in um: ',part_diameters)



#Step 4: Create a csv file
propList = ['Number','Area [um^2]','Rep Diameter [um]']
output_file=open('image_measurements.csv',str(fname),'w')
output_file.write(','+",".join(propList)+'\n')

for count in range(nb_labels):
#    print('1111111',count)
#    print('R',enumerate(propList))
    for i,prop in enumerate(propList):
        if(prop == 'Number'):
            to_print = count+1
#            print('count1',str(to_print))
        elif(prop == 'Area [um^2]'):
            to_print = sizes[int(count)]
#            print('count2',str(to_print))
        else:
            to_print = part_diameters[int(count)]
#            print('count3',str(to_print))
        output_file.write(','+str(to_print))
    output_file.write('\n')


    
    
    
    
    
    
    
    
    
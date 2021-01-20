# -*- coding: utf-8 -*-
"""
Test colors utilities script

@author: aantoun
"""

import colors_utilities as cu
import cv2
import edge_detection_utilities as ed
import numpy as np

#%% Test object creation
trackbar_fig_name = 'test'
cv2.namedWindow(trackbar_fig_name, cv2.WINDOW_GUI_EXPANDED)
img = cv2.imread('test_images/test2.jpg')

r = cu.R(img, trackbar_fig_name, bounds=[112, 255], create_trackbar=False) # create R channel

# h = cu.H(img, trackbar_fig_name)
sob_x = ed.SobelX(img, trackbar_fig_name)
sob_y = ed.SobelY(img, trackbar_fig_name)
sob_mag = ed.SobelMag(img, trackbar_fig_name)
sob_dir = ed.SobelDir(img, trackbar_fig_name)

#%% Show image channel for debugging
while(1):

    sob_mask = (sob_x.value_mask & sob_y.value_mask) | (sob_mag.value_mask & sob_dir.value_mask) 
                              
    final_image = cu.mask_image(r.values, sob_mask)
    
    cv2.imshow('test 2', cu.mask_to_img_8bit(sob_mask))
    cv2.imshow('final result', final_image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
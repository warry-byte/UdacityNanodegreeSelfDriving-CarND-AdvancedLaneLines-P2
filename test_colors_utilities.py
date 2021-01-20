# -*- coding: utf-8 -*-
"""
Test colors utilities script

@author: aantoun
"""

import colors_utilities as cu
import cv2
import edge_detection_utilities as ed
import numpy as np
import glob

#%% Methods
def switch_image(pos):
    global test_images, img
    
    img = cv2.imread(test_images[pos])
    
#%% Create main window
trackbar_fig_name = 'Image channels'
cv2.namedWindow(trackbar_fig_name, cv2.WINDOW_GUI_EXPANDED)

#%% Get input images and create trackbar to switch images
test_images_folder = 'test_images/'
test_images_pattern = "*"
test_images = glob.glob(test_images_folder + test_images_pattern)
img_trackbar_init_pos = 0

# Add trackbar for the image selection
cv2.createTrackbar('Image', 
                   trackbar_fig_name, 
                   0, 
                   len(test_images), 
                   switch_image)  # pass the update method as callback when the user moves the trackbar

cv2.setTrackbarPos('Image', trackbar_fig_name, img_trackbar_init_pos)


#%% Create image channels
img = cv2.imread(test_images[img_trackbar_init_pos])
r = cu.R(img, trackbar_fig_name) # create R channel

# h = cu.H(img, trackbar_fig_name)
sob_x = ed.SobelX(img, trackbar_fig_name)
# sob_y = ed.SobelY(img, trackbar_fig_name)
# sob_mag = ed.SobelMag(img, trackbar_fig_name)
# sob_dir = ed.SobelDir(img, trackbar_fig_name)

#%% Show image channel for debugging

# Switch image depending on trackbar pos

while(1):

    # sob_mask = (sob_x.value_mask & sob_y.value_mask) | (sob_mag.value_mask & sob_dir.value_mask) 
    
    # cv2.imshow('test 2', cu.mask_to_img_8bit(sob_mask))
    r.update_bgr(img)
    
    col = r.values
    sob_mask = cu.mask_to_img_8bit(sob_x.value_mask)
    
    cv2.imshow('final result', col & sob_mask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
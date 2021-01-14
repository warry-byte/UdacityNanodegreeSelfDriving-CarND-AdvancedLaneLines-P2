# -*- coding: utf-8 -*-
"""
Test colors utilities script

@author: aantoun
"""

import colors_utilities as cu
import cv2

#%% Test object creation
trackbar_fig_name = 'test'
cv2.namedWindow(trackbar_fig_name, cv2.WINDOW_GUI_EXPANDED)
img = cv2.imread('test_images/straight_lines1.jpg')
h = cu.H(img, trackbar_fig_name)
s = cu.S(img, trackbar_fig_name)

#%% Show image channel for debugging
while(1):
    
    cv2.imshow('test 2', s.values & h.values)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
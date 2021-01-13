"""
Created on Wed Jan 13 15:07:11 2021

@author: aantoun
"""

import cv2

class ImageParameter():
    def __init__(self, 
                 window_name, 
                 parameter_name='', 
                 min_value=0, 
                 max_value=255, 
                 start_value=255):
        self.window_name = window_name
        self.name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        
        cv2.createTrackbar(self.name, self.window_name, self.min_value, self.max_value, self.update())
        
        
    def update(self): 
        '''
        Callback function called when the trackbar is updated.

        Returns
        -------
        None.

        '''
        
        
        
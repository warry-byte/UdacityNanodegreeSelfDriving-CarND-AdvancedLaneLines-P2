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
        self.__window_name = window_name
        self.__name = parameter_name
        self.__min_value = min_value
        self.__max_value = max_value
        self.__value = start_value
        
        cv2.createTrackbar(self.name, 
                           self.window_name, 
                           self.min_value, 
                           self.max_value, 
                           self.update)
        
    @property
    def value(self):
        return self.__value    
    
    @value.setter
    def value(self, val):
        '''
        Called when manually setting the position of the trackbar

        Parameters
        ----------
        val : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        cv2.setTrackbarPos(self.__name, self.__window_name, val)
        
        # not sure if needed if setTrackbarPos triggers a callback 
        self.__value = cv2.getTrackbarPos(self.__name, self.__window_name)  # the trackbar object checks for min and max values 
        
    def update(self, value): 
        '''
        Callback function called when the trackbar is updated.

        Returns
        -------
        None.

        '''
        self.__value = value  # this is possible because the trackbar values are already constrained
        
        
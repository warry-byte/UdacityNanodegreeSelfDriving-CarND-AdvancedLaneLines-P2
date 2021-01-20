"""
Abstract class that defines an image channel. 
Will encapsulate all possible channels of an image, created as separate class instances (R, G, B, H, S, V, Sobel gradients, etc)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod

class ImageChannel(ABC):
    ''' Abstract class for all image channels (R, G, B, Sobel magnitude, etc)
        All channel data will be updated through calling the self.update() method. 
        The chain of calls is:
            update():
                save bgr image
                bounds: filter image to get channel data with parameter bounds
                
        The bounds is a property of the class. The setter method takes care of setting the __values argument of the channel by taking the pixels in the original channel image that are in the bounds set by the user (normally with trackbar)
        The user has access to the initial value of the channel or to the values as filtered with the image channel bounds.
        
    '''    
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255]):
                
        '''
        Common constructor for all image channel derived classes.
        This class is based on OpenCV. It will create by default two trackbars for each channel, allowing filtering of each channel according to given bounds.
        The trackbars will be added to the window which name is given in input to this method. 
        The limits of both trackbars are also specified in input.
        
        Example of channels: R, G, B, etc
        
        This method assumes that the bgr image in input is an nd-array of shape (W, H, 3).
        
        Parameters
        ----------
        bgr_img : nd-array
            Original image from which the current channel was extracted
        window_name : String
            Name of the window on which the trackbars are added.
        limits : 2D nd-array, optional
            Absolute limits of current image channel. Default is (0, 255). 
        bounds : 1-D nd-array, optional
            Bounds of current image channel. Default is (0, 255). bounds[0] will be associated with trackbar1, and bounds[1] with the second trackbar

        Returns
        -------
        None.

        '''
        self.__bgr = bgr_img
        self.__window_name = window_name
        self.__limits = limits  # set the parameters absolute limits - immuable
        
        # Create trackbars for min and max values of image parameter
        # The name of the child class will be shown and give the name of the trackbar thanks to the name class attribute
        cv2.createTrackbar(self.__class__.__name__ + 'min', 
                           self.__window_name, 
                           self.__limits[0], 
                           self.__limits[1], 
                           self.update_min)  # pass the update method as callback when the user moves the trackbar
        
        cv2.createTrackbar(self.__class__.__name__ + 'max', 
                           self.__window_name, 
                           self.__limits[0], 
                           self.__limits[1], 
                           self.update_max)  
        
        self.__values = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))  # by default, the values will be an nd-array of shape (N, M)
        self.__value_mask = np.empty_like(self.__values, dtype = bool)
        self.__bounds = np.array(bounds)
        self.update_bgr(bgr_img) # this will translate the bgr data into this channel's data
        self.update_min(self.__bounds[0])  # a bit redundant
        self.update_max(self.__bounds[1])
        
    def update_min(self, pos):
        if(pos < self.__bounds[1]): # this will ensure that the trackbar does not move if the min position is equal or high than the high position
            cv2.setTrackbarPos(self.__class__.__name__ + 'min', self.__window_name, pos)
            self.__bounds[0] = pos
            self.update_channel()
        
    def update_max(self, pos):
        if(pos > self.__bounds[0]):
            cv2.setTrackbarPos(self.__class__.__name__ + 'max', self.__window_name, pos)
            self.__bounds[1] = pos
            self.update_channel()
    
    # TODO set as property for the bgr member
    def update_bgr(self, input_bgr_img):
        self.__bgr = input_bgr_img # update initial bgr array to be used when filtering the image to retrieve channel
        self.update_channel() # update channel data after the image has been changed
        
        
    def update_channel(self): 
        '''
        Callback function called when the image channel needs to be updated (new bounds).
        Updating the current channel values to match the input BGR image (e.g.: If current channel is H, input image will be converted to Hue)

        Returns
        -------
        None.

        '''
        self.__initial_values = self.conversion_from_bgr(self.__bgr) # re-initialize initial values on which the current channel will work
        self.__values = np.copy(self.__initial_values)  # the values member needs to be initialized to initial values before filtering, otherwise it is always 0!
        self.filter_data()
        

    def filter_data(self):
        '''
        The methods checks if the bounds do not overlap each other (upper bound should always be higher than the lower one)
        The method then filters the self.__initial_values field and store the result in the self.values present in the self.values field.
        The __bounds member must be updated prior to calling this method.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        '''        
        # Filter initial values
        self.__value_mask = ((self.__initial_values >= self.__bounds[0]) & 
                      (self.__initial_values <= self.__bounds[1]))
        self.__values[ self.__value_mask == False ] = 0  # Zero-out all elements that are not in the range given by the bounds

        
    @property
    def values(self):
        return self.__values
    
    
    @property
    def initial_values(self):
        return self.__initial_values
        
    
    @property
    def value_mask(self):
        return self.__value_mask
    
        
    @abstractmethod
    def conversion_from_bgr(self, bgr_img):
        pass        
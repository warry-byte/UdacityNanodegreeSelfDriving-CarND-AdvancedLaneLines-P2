# Edge detection utility module

from image_channel import ImageChannel
import cv2
import numpy as np
from abc import ABC, abstractmethod

class FilterParameter(ImageChannel):
    ''' FilterParameter class. Inherits from ImageChannel, with added kernel size class member.
    This assumes that the filter uses convolution or kernel size of some sorts.
    The class can create an OpenCV trackbar for the kernel size on a given OpenCV window.
    '''
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255],
                 create_trackbar=True,
                 kernel_size=3):
        
        self.__kernel_size = 3  # initializing to default value in case below code is not run
        
        if(kernel_size % 2 == 1): # if valid, update kernel size
            self.__kernel_size = kernel_size # needs to be done before calling the super constructor,
        # as the constructor will try to convert the input image into sobel using the kernel

        # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits,  
                         bounds=bounds,
                         create_trackbar=create_trackbar)
        
        # Create trackbar for kernel size
        # Kernel size is arbitrarily limited to 11.
        self.__contains_trackbars = create_trackbar
        
        if self.__contains_trackbars:
            
            cv2.createTrackbar(self.__class__.__name__ + '_kernel_size', 
                               window_name, 
                               1, 
                               11, 
                               self.update_kernel_size_member)    
            cv2.setTrackbarPos(self.__class__.__name__ + '_kernel_size', window_name, self.__kernel_size)

    @property
    def kernel_size(self):
        return self.__kernel_size
    
    @kernel_size.setter
    def kernel_size(self, new_size):
        # Setter method for the __kernel_size member
        self.update_kernel_size(new_size)
            

    # Private function - will be called by the setter and by the init method (needed to be separated at the class instance creation)
    def update_kernel_size_member(self, new_size):
        
        if(new_size % 2 == 1 and new_size < 11 ): # kernel size needs to be odd
            self.__kernel_size = new_size
            
            # update trackbar
            if self.__contains_trackbars:
                cv2.setTrackbarPos(self.__class__.__name__ + '_kernel_size', self.window_name, new_size)
                
            self.update_channel()
          
    @abstractmethod
    def conversion_from_bgr(self, bgr_img):
        pass
    

class SobelX(FilterParameter):
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255],
                 create_trackbar=True,
                 kernel_size=3):
                
    # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits,  
                         bounds=bounds,
                         create_trackbar=create_trackbar)
    
    def conversion_from_bgr(self, bgr_img):
        '''
        Abstract method to get current channel from an input BGR image. 
        This method must be reimplemented for all ImageChannel child classes.
        
        This method is called after each update_channel() call, internal to this image channel.
        
        Value mask will contain the binary output (already a property of ImageChannel)

        Parameters
        ----------
        bgr_img : nd-array, shape (W, H, 3)
            BGR input image.

        Returns
        -------
        Image channel nd-array, shape (W, H, 1).

        '''
        return sobel_1d_gradient(bgr_img, 
                                 orient='x', 
                                 sobel_kernel = self.kernel_size, 
                                 thresh=self.bounds)
    
class SobelY(FilterParameter):
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255],
                 create_trackbar=True,
                 kernel_size=3):
                
    # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits,  
                         bounds=bounds,
                         create_trackbar=create_trackbar)
    
    def conversion_from_bgr(self, bgr_img):
        '''
        Abstract method to get current channel from an input BGR image. 
        This method must be reimplemented for all ImageChannel child classes.
        
        This method is called after each update_channel() call, internal to this image channel.
        
        Value mask will contain the binary output (already a property of ImageChannel)

        Parameters
        ----------
        bgr_img : nd-array, shape (W, H, 3)
            BGR input image.

        Returns
        -------
        Image channel nd-array, shape (W, H, 1).

        '''
        return sobel_1d_gradient(bgr_img, 
                                 orient='y', 
                                 sobel_kernel = self.kernel_size, 
                                 thresh=self.bounds)
    
class SobelMag(FilterParameter):
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255],
                 create_trackbar=True,
                 kernel_size=3):
                
    # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits,  
                         bounds=bounds,
                         create_trackbar=create_trackbar)
    
    def conversion_from_bgr(self, bgr_img):
        '''
        Abstract method to get current channel from an input BGR image. 
        This method must be reimplemented for all ImageChannel child classes.
        
        This method is called after each update_channel() call, internal to this image channel.
        
        Value mask will contain the binary output (already a property of ImageChannel)

        Parameters
        ----------
        bgr_img : nd-array, shape (W, H, 3)
            BGR input image.

        Returns
        -------
        Image channel nd-array, shape (W, H, 1).

        '''
        return sobel_gradient_mag(bgr_img, 
                                 sobel_kernel = self.kernel_size, 
                                 thresh=self.bounds)
    
class SobelDir(FilterParameter):
    
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits_deg=[0, 90],  
                 bounds_deg=[0, 90],
                 create_trackbar=True,
                 kernel_size=3):
                
    # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits_deg,  
                         bounds=bounds_deg,
                         create_trackbar=create_trackbar)
    
    def conversion_from_bgr(self, bgr_img):
        '''
        Abstract method to get current channel from an input BGR image. 
        This method must be reimplemented for all ImageChannel child classes.
        
        This method is called after each update_channel() call, internal to this image channel.
        
        Value mask will contain the binary output (already a property of ImageChannel)

        Parameters
        ----------
        bgr_img : nd-array, shape (W, H, 3)
            BGR input image.

        Returns
        -------
        Image channel nd-array, shape (W, H, 1).

        '''
        return sobel_gradient_dir_deg(bgr_img, 
                                 sobel_kernel = self.kernel_size, 
                                 thresh=self.bounds)  # works because bounds is a property of grand-parent class
    
# From Udacity
def sobel_1d_gradient(bgr_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2GRAY)
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    sobel_gradient_8bit = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Return the result
    return sobel_gradient_8bit

def sobel_gradient_mag(img, sobel_kernel=3, thresh=(0, 255)):

    # Take the gradient in x and y separately
    grad_x = sobel_1d_gradient(img, 'x', sobel_kernel, thresh)
    grad_y = sobel_1d_gradient(img, 'y', sobel_kernel, thresh)

    # Calculate the magnitude
    mag = np.sqrt(grad_x**2 + grad_y ** 2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    mag_img_8bit = np.uint8( mag/np.max(mag) * 255 )

    return mag_img_8bit

def sobel_gradient_dir_deg(img, sobel_kernel=3, thresh=(0, 90)):
    
    thresh = thresh * np.pi / 180 # deg to rad

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    grad_x = np.abs(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    grad_y = np.abs(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(grad_y, grad_x)
    
    grad_dir_deg = grad_dir * 180 / np.pi # rad to deg
    
    return grad_dir_deg

# Edge detection utility module

from image_channel import ImageChannel
import cv2
import numpy as np

class SobelX(ImageChannel):
    def __init__(self, 
                 bgr_img,
                 window_name, 
                 limits=[0, 255],  
                 bounds=[0, 255],
                 create_trackbar=True,
                 kernel_size=3):
        
        self.update_kernel_size_member(kernel_size) # set kernel size - needs to be done before calling the super constructor,
        # as the constructor will try to convert the input image into sobel using the kernel

        # Call parent constructor
        super().__init__(bgr_img,
                         window_name, 
                         limits=limits,  
                         bounds=bounds,
                         create_trackbar=create_trackbar)
        
    
    
    @property
    def kernel_size(self):
        return self.__kernel_size
    
    @kernel_size.setter
    def kernel_size(self, new_size):
        # Setter method for the __kernel_size member
        success = self.update_kernel_size(new_size)
        
        if success:
            self.update_channel()

    # Private function - will be called by the setter and by the init method (needed to be separated at the class instance creation)
    def update_kernel_size_member(self, new_size):
        success = False
        
        if(new_size % 2 == 1): # kernel size needs to be odd
            self.__kernel_size = new_size
            success = True
            
        return success
                
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
        Image channel nd-array, shape (W, H, 3).

        '''
        return sobel_1d_gradient(bgr_img, 
                                 orient='x', 
                                 sobel_kernel = self.kernel_size, 
                                 thresh=self.bounds)
    
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

def sobel_mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Take the gradient in x and y separately
    grad_x = sobel_1d_gradient(img, 'x', sobel_kernel, thresh)
    grad_y = sobel_1d_gradient(img, 'y', sobel_kernel, thresh)

    # Calculate the magnitude
    mag = np.sqrt(grad_x**2 + grad_y ** 2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    mag_img_8bit = np.uint8( mag/np.max(mag) * 255 )

    return mag_img_8bit

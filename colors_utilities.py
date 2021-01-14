# Color utilities module

from image_channel import ImageChannel
import cv2

class H(ImageChannel):

    def __init__(self, 
             bgr_img,
             window_name, 
             limits=(0, 255),  
             bounds=(0, 255) ):
    
        super().__init__(bgr_img,
                         window_name, 
                         limits=(0, 255),  
                         bounds=(0, 255) )
        
        # super.__initial_values = self.conversion_from_bgr(bgr_img) # not needed - see init constructor of parent class

    def conversion_from_bgr(self, bgr_img):
        '''
        Abstract method to get current channel from an input BGR image. 
        This method must be reimplemented for all ImageChannel child classes.

        Parameters
        ----------
        bgr_img : nd-array, shape (W, H, 3)
            BGR input image.

        Returns
        -------
        Image channel nd-array, shape (W, H, 3).

        '''
        return color_hue(bgr_img)
    
    
# Color utilities methods

def bgr_to_hsv(bgr_img):
    '''
    Assumes that the input image is an nd-array of shape (W, H, 3) and of format BGR.

    Parameters
    ----------
    bgr_img : nd-array, shape (W, H, 3)
        Input image.

    Returns
    -------
    h : nd-array, shape (W, H)
        H channel.
    s : nd-array, shape (W, H)
        S channel.
    v : nd-array, shape (W, H)
        V channel.

    '''
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        
    return hsv

def bgr_to_hls(bgr_img):
    '''
    Assumes that the input image is an nd-array of shape (W, H, 3) and of format BGR.

    Parameters
    ----------
    bgr_img : nd-array, shape (W, H, 3)
        Input image.

    Returns
    -------
    h : nd-array, shape (W, H)
        H channel.
    l : nd-array, shape (W, H)
        S channel.
    s : nd-array, shape (W, H)
        V channel.

    '''
    hls = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
        
    return hls

def color_hue(bgr_img):
    hsv = bgr_to_hsv(bgr_img)
    
    return hsv[:,:,0]

def color_saturation(bgr_img):
    hsv = bgr_to_hsv(bgr_img)
    
    return hsv[:,:,1]
    
def color_value(bgr_img):
    hsv = bgr_to_hsv(bgr_img)
    
    return hsv[:,:,2]

def color_red(bgr_img):
    return bgr_img[:,:,2]

def color_green(bgr_img):
    return bgr_img[:,:,1]

def color_blue(bgr_img):
    return bgr_img[:,:,0]

def color_lightness(bgr_img):
    hls = bgr_to_hls(bgr_img)
    
    return hls[:,:,1]
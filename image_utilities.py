# Image utility module

import numpy as np
import cv2
import matplotlib.pyplot as plt

def warp_test_images(img):
    '''
    Method used to warp images present in the test_images/ folder. 
    The source and destination points are hardcoded in this method. 

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    warped : TYPE
        DESCRIPTION.

    '''
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    
    s = np.float32(
    [[(img_size[0] / 2) - 70, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
    
    # New points calculated with straight_lines1.jpg
    s = np.float32([[285, 670],
                    [531, 498],
                    [756, 498],
                    [1019, 670]])
    
    # sape = np.copy(s).astype(np.int32) 
    # sape = sape.reshape((-1, 1, 2))
    # color = (255, 0, 255) 
    # thickness = 10 # px
    # i = cv2.polylines(img, sape, True, color, thickness, cv2.LINE_AA) # uncomment to check points
    # cv2.namedWindow('Source points')
    # cv2.imshow('Source points', i)
    
    d = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    
    M = cv2.getPerspectiveTransform(s, d)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

# Main
if __name__ == '__main__':

    img = cv2.imread('test_images/straight_lines1.jpg')
    # plt.imshow(img) # uncomment to check points
    
    w = warp_test_images(img)
    cv2.imshow('Original image', img)
    cv2.imshow('Warped image', w)
    
    while(1):
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

cv2.destroyAllWindows()

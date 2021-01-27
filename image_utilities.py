# Image utility module

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    
    # New points calculated with straight_lines1.jpg
    # s = np.float32([[285, 670],
    #                 [531, 498],
    #                 [756, 498],
    #                 [1019, 670]])
    
    # Plot points
    # sape = np.copy(s).astype(np.int32) 
    # sape = sape.reshape((-1, 1, 2))
    # color = (255, 0, 255) 
    # thickness = 10 # px
    # i = cv2.polylines(img, sape, True, color, thickness, cv2.LINE_AA) # uncomment to check points
    # cv2.namedWindow('Source points')
    # cv2.imwrite('output_folder/unwrap_points.jpg', i)
    
    d = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    
    M = cv2.getPerspectiveTransform(s, d)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return M, warped

def unwarp_image_and_plot_lines(warped, undist, M, left_fit, right_fit):
    '''
    Visualize lane lines on image.

    Returns
    -------
    None.

    ''' 
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Invert transform matrix
    Minv = np.linalg.inv(M)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    
    return result

# Main
if __name__ == '__main__':

    test_filename = 'test_images/straight_lines1.jpg'
    output_folder = "output_images/"
    
    img = cv2.imread(test_filename)
    # plt.imshow(img) # uncomment to check points
    
    
    
    M, w = warp_test_images(img)
    cv2.imshow('Original image', img)
    cv2.imshow('Warped image', w)
    
    cv2.imwrite(output_folder + os.path.splitext(os.path.basename(test_filename))[0] + '_unwarped_bw.jpg', w)

    
    while(1):
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

cv2.destroyAllWindows()

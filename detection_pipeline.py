# Pipeline for advanced lane lines detection
#%% Imports
import glob
import numpy as np
import cv2
import os
import camera as cam
import colors_utilities as cu
import edge_detection_utilities as edu
import image_utilities as iu
import lane_lines_utilities as llu
checkerboard_x_size = 9
checkerboard_y_size = 6
checkerboard_images_folder = 'camera_cal/'
calib_images_name_pattern = "calibration*.jpg"
camera = None

# Creating color channel and gradient objects
img = cv2.imread('test_images/test2.jpg') # dummy image used to create the objects
color_channel = cu.R(img, '', bounds= [223, 255], create_trackbar=False) # create R channel
grad_x = edu.SobelX(img, '', bounds=[0, 52], create_trackbar=False)
    
#%% Methods
def pipeline(input_img, filename):
    '''
    Image detection pipeline for line detection using advanced techniques (Project 2)

    Parameters
    ----------
    input_image : Array of uint8, shape: (size_x, size_y, 3)
        Input image on which to apply the detection pipeline.

    Returns
    -------
    None.

    '''
    
    global camera, color_channel, grad_x
        
    # Undistort image
    undist_img = cv2.undistort(input_img, camera.mtx, camera.dist, None, camera.mtx)
    
    cv2.imwrite(output_folder + os.path.splitext(os.path.basename(filename))[0] + '_undist.jpg', undist_img)

    # Update color channel and gradient channel with input image
    color_channel.update_bgr(undist_img)
    grad_x.update_bgr(undist_img)
    
    # final filtered image: color channel AND gradient channel
    filt_img = cu.mask_image(color_channel.values, grad_x.value_mask)
    
    # Perspective transform
    M, warp_img = iu.warp_test_images(filt_img)
    
    cv2.imwrite(output_folder + os.path.splitext(os.path.basename(filename))[0] + '_unwarped_bw.jpg', warp_img)

    # Create lane lines by fitting polynomial
    out_img, left_fitx, right_fitx, ploty = llu.fit_polynomial(warp_img)
    
    output_unwarped = iu.unwarp_image_and_plot_lines(warp_img, undist_img, M, left_fitx, right_fitx, ploty)
    
    return out_img, output_unwarped, left_fitx, right_fitx, ploty

# Main
if __name__ == '__main__':

    #%% Variables definition
    test_images_folder = 'test_images/'
    test_images_pattern = "*"
    test_images = glob.glob(test_images_folder + test_images_pattern)
    output_folder = "output_images/"
    
    #%% Get camera calibration
    camera = cam.CustomCamera(calib_images_folder_path=checkerboard_images_folder, 
                                 calib_images_name_pattern=calib_images_name_pattern, 
                                 checkerboard_x_size=checkerboard_x_size, 
                                 checkerboard_y_size=checkerboard_y_size)
    
    #%% Analyze all files
    for f in test_images:
        current_img = cv2.imread(f)
        output_img, output_img_unwarped, left_fitx, right_fitx, ploty = pipeline(current_img, f) # Execute pipeline
        
        # save to file
        # cv2.imshow('Test pipeline: ' + os.path.basename(f), output_img)
        cv2.imwrite(output_folder + os.path.basename(f), output_img)
        cv2.imwrite(output_folder + os.path.splitext(os.path.basename(f))[0] + '_output.jpg', output_img_unwarped)

    
    # cv2.destroyAllWindows()
    
    #%% Single image pipeline
   # output_img, output_img_unwarped = pipeline(current_img, test_images[0]) # Execute pipeline
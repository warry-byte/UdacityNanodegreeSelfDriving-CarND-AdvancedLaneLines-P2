# Pipeline for advanced lane lines detection
#%% Imports
import glob
import numpy as np
import cv2
import os
import camera as cam

checkerboard_x_size = 9
checkerboard_y_size = 6
checkerboard_images_folder = 'camera_cal/'
calib_images_name_pattern = "calibration*.jpg"
camera = None
    
#%% Methods
def pipeline(input_img):
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
    
    global camera
        
    # Undistort image
    undist_img = cv2.undistort(input_img, camera.mtx, camera.dist, None, camera.mtx)
    
    # Threshold image in HSV space
    # Set minimum and maximum HSV values to display
    lower = np.array([0, 0, 180])
    upper = np.array([70, 255, 255])

    # Convert to HSV format and threshold with HSV mask (lower and upper values)
    hsv = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output_img = cv2.bitwise_and(undist_img, undist_img, mask=mask)
    
    return output_img

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
    
    camera.calculate_intrinsics()
    
    #%% Analyze all files
    for f in test_images:
        current_img = cv2.imread(f)
        output_img = pipeline(current_img) # Execute pipeline
        
        # save to file
        # cv2.imshow('Test pipeline: ' + os.path.basename(f), output_img)
        cv2.imwrite(output_folder + os.path.basename(f), output_img)

    
    # cv2.destroyAllWindows()
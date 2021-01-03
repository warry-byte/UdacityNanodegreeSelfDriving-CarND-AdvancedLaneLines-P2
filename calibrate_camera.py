#%% Imports
import matplotlib.pyplot as plt
import camera
import cv2

#%% Variables
checkerboard_x_size = 9
checkerboard_y_size = 6
checkerboard_images_folder = 'camera_cal/'
calib_images_name_pattern = "calibration*.jpg"


camera = camera.CustomCamera(calib_images_folder_path=checkerboard_images_folder, 
                             calib_images_name_pattern=calib_images_name_pattern, 
                             checkerboard_x_size=checkerboard_x_size, 
                             checkerboard_y_size=checkerboard_y_size)

#%% Calibrate camera
camera.calculate_intrinsics()

#%% Testing camera calibration on calibration1.jpg
test_img_filename = "calibration1.jpg"
test_img = cv2.imread(checkerboard_images_folder + test_img_filename)
nx = 9
ny = 5
undist_img, m = camera.corners_unwarp(test_img, nx, ny)

# cv2.imshow('Undistorted image - file: ' + test_img_filename, undist_img)

cv2.imwrite("output_images/" + test_img_filename + "_undistort.jpg", undist_img)
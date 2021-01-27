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
from moviepy.editor import VideoFileClip

#%% Variables definition
checkerboard_x_size = 9
checkerboard_y_size = 6
checkerboard_images_folder = 'camera_cal/'
calib_images_name_pattern = "calibration*.jpg"
camera = None
test_images_folder = 'test_images/'
test_images_pattern = "*"
test_images = glob.glob(test_images_folder + test_images_pattern)
output_folder = "output_images/"
current_filename = None
save_intermediate_steps = False
analyze_images = False
left_fit = None
right_fit = None
lanes_mid_pos = None
out_img = None # TODO should probably be renamed to output_warped_img or output_warp_debug_img...

# Creating color channel and gradient objects
img = cv2.imread('test_images/test2.jpg') # dummy image used to create the objects
color_channel = cu.R(img, '', bounds= [220, 255], create_trackbar=False) # create R channel
grad_x = edu.SobelX(img, '', bounds=[0, 52], create_trackbar=False)
    
#%% Methods
def pipeline(img):
    '''
    Image detection pipeline for line detection using advanced techniques (Project 2)

    Parameters
    ----------
    input_image : Array of uint8, shape: (size_x, size_y, 3)
        Input image on which to apply the detection pipeline.
        This pipeline assumes a BGR image. If an RGB image is provided, flag analyze_images must be set to False for this method to convert the image to BGR. 

    Returns
    -------
    output_unwarped: initial image with identified lane lines and lane space in alpha

    '''
    
    global camera, color_channel, grad_x
    global output_folder
    global current_filename
    global save_intermediate_steps
    global left_fit 
    global right_fit 
    global lanes_mid_pos 
    global out_img
    
    if not analyze_images: # this means that we have a video output --> input is rgb
        input_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert to BGR
    else:
        input_img = img
        
    # Undistort image
    undist_img = cv2.undistort(input_img, camera.mtx, camera.dist, None, camera.mtx)
    
    # Update color channel and gradient channel with input image
    color_channel.update_bgr(undist_img)
    grad_x.update_bgr(undist_img)
    
    # final filtered image: color channel AND gradient channel
    filt_img = cu.mask_image(color_channel.values, grad_x.value_mask)
    
    # Perspective transform
    M, warp_img = iu.warp_test_images(filt_img)

    # Create lane lines by fitting polynomial
    out_img, left_fit, right_fit, lanes_mid_pos = llu.fit_polynomial(warp_img)
    
    output_unwarped = iu.unwarp_image_and_plot_lines(warp_img, undist_img, M, left_fit, right_fit)
    
    if save_intermediate_steps:
        cv2.imwrite(output_folder + os.path.splitext(os.path.basename(current_filename))[0] + '_undist.jpg', undist_img)
        cv2.imwrite(output_folder + os.path.splitext(os.path.basename(current_filename))[0] + '_unwarped_bw.jpg', warp_img)
        
    if not analyze_images: # this means that we have a video output --> input is rgb
        out = cv2.cvtColor(output_unwarped, cv2.COLOR_BGR2RGB) # convert to BGR
    else:
        out = output_unwarped
    
    
    return out


# Main
if __name__ == '__main__':
    
    #%% Get camera calibration
    camera = cam.CustomCamera(calib_images_folder_path=checkerboard_images_folder, 
                                 calib_images_name_pattern=calib_images_name_pattern, 
                                 checkerboard_x_size=checkerboard_x_size, 
                                 checkerboard_y_size=checkerboard_y_size)
    
    if analyze_images:
        #%% Analyze all files
        for f in test_images:
            current_filename = f
            current_img = cv2.imread(f)
            output_img_unwarped = pipeline(current_img) # Execute pipeline
            
            # TODO remove (debug only)
            # Text on image: display coefficients
            # font = cv2.FONT_HERSHEY_SIMPLEX  
            # org = (50, 50) 
            # fontScale = 1
            # color = (255, 0, 255) 
            # thickness = 2 # px
            
            # # Display image name
            # txt = "xL = " + str(left_fit[0]) + " y^2 + " + str(left_fit[1]) + " y + " + str(left_fit[2])
            # cv2.putText(output_img, txt, org, font, fontScale, color, thickness, cv2.LINE_AA) 
            
            cv2.imwrite(output_folder + os.path.basename(f), output_img)
            cv2.imwrite(output_folder + os.path.splitext(os.path.basename(f))[0] + '_output.jpg', output_img_unwarped)
            
            left_radius, right_radius = llu.measure_curvature_real(left_fit, right_fit, output_img.shape[0]-1)
            
            print("File: " + f + ", radii: " + str(left_radius) + ", " + str(right_radius))
            print("Lanes middle position: " + str(lanes_mid_pos))
        
    
    else:
        input_video = 'project_video.mp4'
        output_video_filename = 'output_videos/' + os.path.splitext(os.path.basename(input_video))[0] + '_output.mp4'
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1_RGB = VideoFileClip(input_video)
        output_clip = clip1_RGB.fl_image(pipeline) #NOTE: this function expects color images!!
        # %time white_clip.write_videofile(white_output, audio=False)
        output_clip.write_videofile(output_video_filename, audio=False)
   
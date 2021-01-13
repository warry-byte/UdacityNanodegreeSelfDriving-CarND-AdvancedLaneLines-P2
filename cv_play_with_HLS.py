#%% Inspired from https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv/48367205#48367205

#%% Imports
import cv2
import numpy as np
import glob
import  grad_amplitude 
import  grad_direction 
import time

#%% Methods
def nothing(x):
    pass


def image_processing_pipeline(input_img, 
                              sobel_lower,
                              sobel_upper,
                              color_lower, 
                              color_upper, 
                              color_conversion=cv2.COLOR_BGR2HSV):
    ''' 
    Image processing pipeline
    '''
    # color thresholding
    color_img = cv2.cvtColor(input_img, color_conversion)
    color_mask = cv2.inRange(color_img, color_lower, color_upper)
    color_masked_img = cv2.bitwise_and(input_img, input_img, mask=color_mask)
    
    # gradient magnitude filtering using sobel
    sob_thresh_mag_min = sobel_lower[0]
    sob_thresh_dir_min = sobel_lower[1]
    sob_thresh_mag_max = sobel_upper[0]
    sob_thresh_dir_max = sobel_upper[1]
    result_mag_bin = grad_amplitude.mag_thresh(color_masked_img, 
                                               thresh=(sob_thresh_mag_min, sob_thresh_mag_max))  # [0, 1]
            
    # gradient direction filtering - USE COLOR FILTERED IMAGE and bitwise-AND mask with magnitude
    result_dir_bin = grad_direction.dir_thresh(color_masked_img, 
                                               thresh=(sob_thresh_dir_min, sob_thresh_dir_max)) 
    
    final_mask = result_mag_bin.astype(np.uint8) & result_dir_bin.astype(np.uint8)
    output_img = cv2.bitwise_or(color_masked_img, color_masked_img, mask=final_mask )
    
    return output_img

#%% Init
trackbar_fig_name = 'HSV_val'
test_images_folder = 'test_images/'
test_images_pattern = "*"
test_images = glob.glob(test_images_folder + test_images_pattern)

#%% Create a window with trackbars
cv2.namedWindow(trackbar_fig_name, cv2.WINDOW_GUI_EXPANDED)
# cv2.namedWindow(trackbar_fig_name, cv2.WINDOW_AUTOSIZE)

# Create trackbars for color change
# Hue is from 0-179 for Opencv
trackbar_names = ['HMin', 'SMin', 'VMin', 
                  'HMax', 'SMax', 'VMax', 
                  'Sob Mag Min', 'Sob Mag Max', 
                  'Sob Dir Min', 'Sob Dir Max']

for t in range(0, len(trackbar_names)-1):
    if('Sob Dir' in trackbar_names[t]):
        max_trackbar_value = 90
    else:
        max_trackbar_value = 255
    
    cv2.createTrackbar(trackbar_names[t], trackbar_fig_name, 0, max_trackbar_value, nothing)
    
    if('Max' in trackbar_names[t]):
        cv2.setTrackbarPos(trackbar_names[t], trackbar_fig_name, max_trackbar_value)
    else: # Min value is all 0 - to be changed later if implementing pickling of the values 
        cv2.setTrackbarPos(trackbar_names[t], trackbar_fig_name, 0)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

#%% Load images into list, scale down and display as figures. Figures names will have the same name as the relative path to the images
image_list = []  # will share the same indices as test_images list, which contains the figures file names
scale_percent = 60 # percent of original size

# TODO proper scaling inside while 1 to account for image rescaling by the user
for f in test_images:
    image = cv2.imread(f)

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image and append to list
    scaled_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image_list.append(scaled_image)
    
    cv2.namedWindow(f, cv2.WINDOW_GUI_EXPANDED)  # name of the figure is the relative path file name
    cv2.imshow(f, scaled_image) 


#%% While 1 loop: modify all images according to HSV trackbars 
while(1):
    # Get current positions of all trackbars    
    hMin = cv2.getTrackbarPos('HMin', trackbar_fig_name)
    sMin = cv2.getTrackbarPos('SMin', trackbar_fig_name)
    vMin = cv2.getTrackbarPos('VMin', trackbar_fig_name)
    hMax = cv2.getTrackbarPos('HMax', trackbar_fig_name)
    sMax = cv2.getTrackbarPos('SMax', trackbar_fig_name)
    vMax = cv2.getTrackbarPos('VMax', trackbar_fig_name)
    sob_thresh_mag_min = cv2.getTrackbarPos('Sob Mag Min', trackbar_fig_name)
    sob_thresh_mag_max = cv2.getTrackbarPos('Sob Mag Max', trackbar_fig_name)
    sob_thresh_dir_min = cv2.getTrackbarPos('Sob Dir Min', trackbar_fig_name) * np.pi/180
    sob_thresh_dir_max = cv2.getTrackbarPos('Sob Dir Max', trackbar_fig_name) * np.pi/180 # scale to first quadrant angle
    
    # Set minimum and maximum HSV values to display
    color_lower = np.array([hMin, sMin, vMin])
    color_upper = np.array([hMax, sMax, vMax])
    sobel_lower = np.array([sob_thresh_mag_min, sob_thresh_dir_min])
    sobel_upper = np.array([sob_thresh_mag_max, sob_thresh_dir_max])

    # Convert images to HSV format, and apply color and sobel thresholds
    # Detection pipeline: 
    for i in range(len(image_list)):
        
        start_time = time.time()
        
        res = image_processing_pipeline(image_list[i], 
                                  sobel_lower, 
                                  sobel_upper, 
                                  color_lower, 
                                  color_upper)
        # # color thresholding
        # hsv = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, color_lower, color_upper)
        # color_thresh = cv2.bitwise_and(image_list[i], image_list[i], mask=mask)
        
        # # gradient magnitude filtering using sobel
        # result_mag_bin = sob_mag.mag_thresh(color_thresh, thresh=(sob_thresh_mag_min, sob_thresh_mag_max))  # [0, 1]
                
        # # gradient direction filtering - USE COLOR FILTERED IMAGE and bitwise-AND mask with magnitude
        # result_dir_bin = sob_dir.dir_thresh(color_thresh, thresh=(sob_thresh_dir_min, sob_thresh_dir_max)) 
        
        # final_mask = result_mag_bin.astype(np.uint8) & result_dir_bin.astype(np.uint8)
        # result = cv2.bitwise_or(src1, src2)(color_thresh, color_thresh, mask=final_mask )
        
        # Display result images
        print("--- Pipeline for one image: %s seconds ---" % (time.time() - start_time))
        
        start_time = time.time()
        cv2.imshow(test_images[i], res) # figure name was set to the relative path name # [0, 254]
        print("--- Show one image: %s seconds ---" % (time.time() - start_time))
        
        

    # Print if there is a change in HSV value - useful when debugging
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
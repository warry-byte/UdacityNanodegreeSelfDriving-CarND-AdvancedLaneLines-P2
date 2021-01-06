#%% Inspired from https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv/48367205#48367205



import cv2
import numpy as np
import glob
import apply_sobel as sob

def nothing(x):
    pass


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
cv2.createTrackbar('HMin', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('SMin', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('VMin', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('HMax', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('SMax', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('VMax', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('Sob Min', trackbar_fig_name, 0, 255, nothing)
cv2.createTrackbar('Sob Max', trackbar_fig_name, 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', trackbar_fig_name, 255)
cv2.setTrackbarPos('SMax', trackbar_fig_name, 255)
cv2.setTrackbarPos('VMax', trackbar_fig_name, 255)
cv2.setTrackbarPos('Sob Max', trackbar_fig_name, 255)

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
    sob_thresh_min = cv2.getTrackbarPos('Sob Min', trackbar_fig_name)
    sob_thresh_max = cv2.getTrackbarPos('Sob Max', trackbar_fig_name)

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert images to HSV format, and apply color and sobel thresholds
    # Detection pipeline: 
    for i in range(len(image_list)):
        # color thresholding
        hsv = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        color_thresh = cv2.bitwise_and(image_list[i], image_list[i], mask=mask)
        
        # gradient filtering using sobel
        result_bin = sob.abs_sobel_thresh(color_thresh, thresh=(sob_thresh_min, sob_thresh_max))  # [0, 1]
        result = cv2.bitwise_and(color_thresh, color_thresh, mask=result_bin)
        
        # Display result images
        cv2.imshow(test_images[i], result) # figure name was set to the relative path name # [0, 254]
        
        

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
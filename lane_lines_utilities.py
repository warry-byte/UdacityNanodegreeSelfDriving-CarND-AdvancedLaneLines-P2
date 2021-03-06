# Lane line module

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
# From Udacity
def find_lane_pixels_get_pos(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Define vehicle position as the midpoint of the histograms bottom
    lanes_mid_pos = midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, lanes_mid_pos


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, lanes_mid_pos = find_lane_pixels_get_pos(binary_warped)

    # we fit x values as fct of y values, as the lines are considered vertical in this detection algorithm
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # try:
    #     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # except TypeError:
    #     # Avoids an error if `left` and `right_fit` are still none or incorrect
    #     print('The function failed to fit a line!')
    #     left_fitx = 1*ploty**2 + 1*ploty
    #     right_fitx = 1*ploty**2 + 1*ploty

    # ## Visualization ##
    # # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    return out_img, left_fit, right_fit, lanes_mid_pos

def measure_curvature_pixels(left_fit, right_fit, y_eval):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''    
    # Compute radii
    left_curverad = np.power((1+(2*left_fit[0]*y_eval + left_fit[1])**2), 1.5) / np.abs(2*left_fit[0])  ## Implement the calculation of the left line here
    right_curverad = np.power((1+(2*right_fit[0]*y_eval + right_fit[1])**2), 1.5) / np.abs(2*right_fit[0])  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def measure_curvature_real(left_fit_px, right_fit_px, y_eval):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30/720 # meters per pixel in y dimension - considering 30 m line of sight
    ym_per_pix = 30/260 # meters per pixel in y dimension - considering 30 m line of sight
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_px[0]*y_eval*ym_per_pix + left_fit_px[1])**2)**1.5) / np.absolute(2*left_fit_px[0])
    right_curverad = ((1 + (2*right_fit_px[0]*y_eval*ym_per_pix + right_fit_px[1])**2)**1.5) / np.absolute(2*right_fit_px[0])
    
    return left_curverad, right_curverad

# Main
if __name__ == '__main__':

    # Load our image
    binary_warped = mpimg.imread('output_images/straight_lines1.jpg')
    
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels_get_pos(binary_warped)
    out_img = fit_polynomial(binary_warped)
    
    plt.figure()
    plt.imshow(out_img)
    plt.show()
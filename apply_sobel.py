import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image and grayscale it
# image = mpimg.imread('test_images\test2.jpg')


# solution
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=20, thresh_max=100
# should produce output like the example image shown above this quiz.
# NOTE: did not seem to produce the same output....
def abs_sobel_thresh_mine(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        deriv_img = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif(orient == 'y'):
        deriv_img = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print("Error: Gradient direction should be x or y.")

    # 3) Take the absolute value of the derivative or gradient
    abs_deriv_img = np.absolute(deriv_img)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    abs_deriv_img_8bit = np.uint8( abs_deriv_img/np.max(abs_deriv_img) * 255 )

    # 5) Create a mask of 1's where the scaled gradient magnitude
    #binary_output = np.where(np.logical_and(abs_deriv_img_8bit <= thresh_max, abs_deriv_img_8bit >= thresh_min) == True, 1, 0)

    binary_output = np.zeros(abs_deriv_img_8bit.shape)
    binary_output[ (abs_deriv_img_8bit <= thresh[1]) & (abs_deriv_img_8bit >= thresh[0]) ] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


# Run the function
# grad_binary = abs_sobel_thresh(image, orient='x', thresh=(20,100))
# grad_binary_mine = abs_sobel_thresh_mine(image, orient='x', thresh=(20,100))
#
# # Compare solution with my results - breakpoint after this
# compar = grad_binary - grad_binary_mine
# a = plt.imshow(compar)
# plt.show()


# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(grad_binary, cmap='gray')
# ax2.set_title('Thresholded Gradient', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

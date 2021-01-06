import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image
# image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, sobel_kernel)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, sobel_kernel)

    # 3) Calculate the magnitude
    grad = np.sqrt(grad_x**2 + grad_y ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    grad_img_8bit = np.uint8( grad/np.max(grad) * 255 )

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros(grad_img_8bit.shape)
    binary_output[ (grad_img_8bit <= thresh[1]) & (grad_img_8bit >= thresh[0]) ] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Run the function
# mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(mag_binary, cmap='gray')
# ax2.set_title('Thresholded Magnitude', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
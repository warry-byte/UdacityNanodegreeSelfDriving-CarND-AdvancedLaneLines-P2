---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: /output_images/calibration1.jpg_undistort.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### List of files
#### Detection pipeline
- detection_pipeline.py: entry point of the project. Contains the whole pipeline to process images and eventually find lane lines polynomials. 

#### Utility files
- camera.py: contains camera information and calibration methods. 
- image_utilities.py: warping methods
- image_channel.py: Abstract parent class for all image channels (colors, gradients). It contains all the logic to handle channels thresholding and the user interface on the form of OpenCV trackbars for min and max thresholds. 

#### Other files: 
- color_utilities.py: contains color channel classes and methods to get all colors (H, S, V, L, and R G B). 
- edge_detection_utilities.py: contains class FilterParameter (inherits from ImageChannel), and child classes Sobel- (Sobel gradients, etc)

- lane_line_utilities.py: contains all the processing to find the line polynomials

#### Test files
- test_color_utilities.py: Allows to use color and sobel classes to combine the different colors together. 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Note that images calibration1.jpg, calibration4.jpg and calibration5.jpg could not be used to calculate the distortion coefficients due to occluded points. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

<img src="file:///output_images/calibration1.jpg_undistort.jpg" alt="drawing" width="200"/>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

<img src="file:///output_images/straight_lines1.jpg" alt="drawing" width="200"/>
<img src="file:///output_images/straight_lines1_undist.jpg" alt="drawing" width="200"/>

The difference between distorted and undistorted image can be noticed especially at the bottom left corner of the image. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

##### Color thresholding

Color thresholding in HSV space + gradient thresholding using Sobel filter. 
The script named "cv_play_with_HLS.py" allows to modify the color thresholds and Sobel thresholds with trackbars.

Color thresholding: Observations
* Vmin: high values seem to isolate the lines reasonably well. However a high value has a tendancy to suppress the parts of the lane that are further away.
* S: thresholding on S does not seem to provide good results, as the lines are either plain white or plain yellow
* Hmin: destroys the lines
* Hmax: removes the sky and a portion of the gray component of the road. Same comment as for Vmin
* R values: seem to delimitate the lines well. 

R values: (220, 255) (to leave as much lines as possible)

Color thresholding processing time: about 2 ms (ballpark value - might depend on the chosen thresholding)

##### Gradient filtering
Magnitude threshold: takes about 16 ms to run
Direction threshold: takes about 27 ms to run --> unable to be used for real-time acquisition

Exploration is done on the following pipeline:
- Gradient and color filtering logic: grad_mask = (GradX & GradY) | (Grad_mag & Grad_dir) 
- Final image: R & grad_mask

Observations on test2.jpg:
- Important loss of information in the image when filtering in y direction --> x filtering checks for gradients (edges) along X
- Sobel magnification and direction kernel of more than 1 is unnecessary
- Did not find grad mag and grad dir filtering outstanding
- X gradient kernel size as high as possible. Not much difference above 11
- Results might be improved by taking the R channel as input image for the gradient filtering

Results: 
- gradient and color filtering: GradX & R
- Threshold values:
	12, 255
- Kernel value: 11


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_test_images()` based on warper(), which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py).  The function takes as inputs an image (`img`).  
As there is no information with respect to the camera extrinsics parameters in the car, the source points were chosen on the image straight_lines1.jpg, under the assumption that the lines are straight and that the camera position is the same for all recordings:

```python

s = np.float32(
[[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
[((img_size[0] / 6) - 10), img_size[1]],
[(img_size[0] * 5 / 6) + 60, img_size[1]],
[(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    
	
d = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following destination points (slightly different from Udacity initial points):

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `s` and `d` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="file:///output_images/straight_lines1_unwarped_bw.jpg" alt="drawing" width="200"/>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the lanes in the image and apply the polynomial fit, the following operations are executed:
- Input image: color + gradient filtered image binary
- Sum pixels along the X dimension --> histogram of the bottom half of the image
- Calculate left and right peak points of the histogram --> this will be the starting point of the lane line search
The image will be divided in the X dimension into equal "windows" (rectangle) --> sliding windows
- For window i:
	* Identify boundaries of window i according to window parameters
	* Identify non-zero pixels in current left and right windows
	* Save indices of pixels that are in the current windows
	* If the current window contains more pixels than minpix threshold: recenter current window

The result of this algorithm is depicted below, where the pixels identified as part of the lane lines are colored in blue and red for each lane:

<img src="file:///output_images/test1.jpg" alt="drawing" width="200"/>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:


![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

import numpy as np
import cv2
import glob
import pickle

class CustomCamera():

    def __init__(self, 
                 calib_images_folder_path='', 
                 calib_images_name_pattern="calibration*.jpg", 
                 checkerboard_x_size=5, 
                 checkerboard_y_size=5):
        '''
        Constructor of the camera class. 

        Parameters
        ----------
        calib_images_folder_path : String, optional
            Path to the calibrated images folder. The default is ''.
        images_name_pattern : String, optional
            Calibration images filename pattern. The default is "calibration*.jpg".
        checkerboard_x_size : int, optional
            Number of corners of the checkerboard in the image along the X dimension.. The default is 5.
        checkerboard_y_size : int, optional
            Number of corners of the checkerboard in the image along the Y dimension.. The default is 5.

        Returns
        -------
        None.

        '''
        self.mtx = None
        self.dist = None
        self.calib_images_folder_path = calib_images_folder_path
        self.images_name_pattern = calib_images_name_pattern
        self.nx = checkerboard_x_size
        self.ny = checkerboard_y_size
        
    def calculate_intrinsics(self):
        '''
        Calculate camera intrinsics (mtx and dist) based on calibration images.
        We assume that the calibration images folder path and images name pattern have been correctly initialized..

        Returns
        -------
        None.

        '''
        
        #%% Initialize variables
        ''' Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) 
        Format of objp: 
            [[0, 0, 0],
             [1, 0, 0],
             ...
             [Nx-1, 0, 0],
             
             [0, 1, 0],
             [1, 1, 0],
             ...
             [Nx-1, Ny-1, 0]
             
             The Z coordinate is always 0 for a 2D calibration pattern
             
        '''
        objp = np.zeros((self.ny*self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        images = glob.glob(self.calib_images_folder_path + self.images_name_pattern)
        
        #%% Step through the list and search for checkerboard corners
        for idx, fname in enumerate(images):
            print("Reading calibration image: " + fname)
            
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, 
                                                     (self.nx, self.ny), 
                                                     None)
        
            # If found, add object points, image points
            if ret == True:
                print("Corners found. Append points...")
                
                objpoints.append(objp)
                imgpoints.append(corners)
        
                # Draw and display the corners
                cv2.drawChessboardCorners(img, 
                                          (self.nx, self.ny), 
                                          corners, 
                                          ret)
                
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            
                
        #%% Calibrate camera
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                           imgpoints, 
                                                           img_size, 
                                                           None, 
                                                           None)
        # Store camera intrinsics
        self.mtx = mtx
        self.dist = dist

    def corners_unwarp(self, img, nx, ny):
        '''
        Undistort input image (supposedly containing a checkerboard) using camera intrinsics and warp according to calculated perspective.
    
        Parameters
        ----------
        img : Array of uint8, shape: (size_x, size_y, 3)
            Image to unwarp.
        nx : int
            Number of corners of the checkerboard in the image along the X dimension.
        ny : int
            Number of corners of the checkerboard in the image along the Y dimension.
    
        Returns
        -------
        warped_img : Array of uint8, shape: (size_x, size_y, 3)
            Output image, unwarped and put to perspective.
        m : Array
            Transformation matrix.
    
        '''
        # 1) Undistort using mtx and dist
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
        # 2) Convert to grayscale
        gray_undist_img = gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    
        # img_size = gray_undist_img.shape()
    
        # 3) Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_undist_img, (nx, ny), None)
    
        # 4) If corners found:
        if(len(corners) > 0):
            # a) draw corners
            undist_gray_corner_img = cv2.drawChessboardCorners(gray_undist_img, (nx,ny), corners, ret)
    
            img_x = undist_gray_corner_img.shape[0]
            img_y = undist_gray_corner_img.shape[1]
    
            # b) define 4 source points - TL TR BR BL
            src_pts = np.float32([corners[0],corners[nx-1], corners[-1], corners[-nx]])
    
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            dst_pts = np.float32([[0,0],[0,img_x-1],[img_y-1,img_x-1],[img_y-1,0]])
    
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
            # e) use cv2.warpPerspective() to warp your image to a top-down view
            warped_img = cv2.warpPerspective(undist_gray_corner_img, m,
                                             (img_y,img_x), flags=cv2.INTER_LINEAR)
        # M = None
        # warped = np.copy(img)
        return warped_img, m
        
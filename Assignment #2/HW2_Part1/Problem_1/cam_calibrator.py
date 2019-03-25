#!/usr/bin/python

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np
import glob

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]


    def genCornerCoordinates(self, u_meas, v_meas):
        '''
        Inputs:
            u_meas: a list of arrays where each array are the u values for each board.
            v_meas: a list of arrays where each array are the v values for each board.
        Output:
            corner_coordinates: a tuple (Xg, Yg) where Xg/Yg is a list of arrays where each array are the x/y values for each board.

        HINT: u_meas, v_meas starts at the blue end, and finishes with the pink end
        HINT: our solution does not use the u_meas and v_meas values 
        HINT: it does not matter where your frame it, as long as you are consistent! 
        '''

        ########## Code starts here ##########

        #Since the frame doesn't matter; Let us use our frame as 
        #    	^
        #		|
        #	y	|
        #		|
        #		----------->
        #			x
        # Let's code the grid as 6 even spaces between 7 corners
        # np.linspace(start, stop, num_of_samples)
        rows = np.linspace(0.0, self.d_square*(self.n_corners_x-1), num = self.n_corners_x)
        # Let's code the grid as 8 even spaces between 9 corners
        # Note the use of ....corners_x-1; as n points create n-1 intervals between them
        columns = np.linspace(0.0, self.d_square*(self.n_corners_y-1), num = self.n_corners_y)
        # Let's reshape the columns as a column vector
        columns = columns.reshape(self.n_corners_y,1)
        # The measured velocities as shown in the figure start from bottom to top;
        # But my convention is opposite to this
        # So, I will reverse the column matrix I have just found
        columns = np.flipud(columns)
        # Now we will need 9 instances of our x_cordinates for 9 corners vertically
        corner_x = np.broadcast_to(rows, (self.n_corners_y, self.n_corners_x))
      	#Collapsing the matrix structure into a single row
        corner_x = corner_x.reshape(-1)
        # We will also need 7 instances of our y_cordinates for 7 corners horizontally
        corner_y = np.broadcast_to(columns, (self.n_corners_y, self.n_corners_x))
        corner_y = corner_y.reshape(-1)
        # Multiplying with no. of images to get tuples which contains list of arrays for each image
        x = [corner_x] * self.n_chessboards
        y = [corner_y] * self.n_chessboards
        #Finally, I return the tuple
        corner_coordinates = (x,y)  # UPDATE ME

        ########## Code ends here ##########

        return corner_coordinates

    def estimateHomography(self, u_meas, v_meas, X, Y):    # Zhang Appendix A
        '''
        Inputs:
            u_meas: an array of the u values for a board. -> 7
            v_meas: an array of the v values for a board. -> 9
            X: an array of the X values for a board. (from genCornerCoordinates) -> 63
            Y: an array of the Y values for a board. (from genCornerCoordinates) -> 63
        Output:
            H: the homography matrix. its size is 3x3
        
        HINT: What is the size of the matrix L? -> 2n x 9
        HINT: What are the outputs of the np.linalg.svd function? Based on this, where does the eigenvector corresponding to the smallest eigen value live?
        HINT: np.stack and/or np.hstack may come in handy here.
        '''
        ########## Code starts here ##########

        # Each image has 7*9 = 63 points
        n = u_meas.shape[0]  
        
        L = np.zeros((2*n, 9))
        for i in range(n):
            L[2*i] = [X[i], Y[i], 1, 0, 0, 0, -u_meas[i]*X[i], -u_meas[i]*Y[i], -u_meas[i]]
            L[2*i+1] = [0, 0, 0, X[i], Y[i], 1, -v_meas[i]*X[i], -v_meas[i]*Y[i], -v_meas[i]]
        
        _, _, v_T = np.linalg.svd(L)
        H = v_T[-1, :]  # Reversing the array as per our requirement for reshaping
        H = np.reshape(H, (3, 3))

        ########## Code ends here ##########
        return H

    def getCameraIntrinsics(self, H):    # Zhang 3.1, Appendix B
        '''
        Input:
            H: a list of homography matrices for each board
        Output:
            A: the camera intrinsic matrix

        HINT: MAKE SURE YOU READ SECTION 3.1 THOROUGHLY!!! V. IMPORTANT
        HINT: What is the definition of h_ij?  
        HINT: It might be cleaner to write an inner function (a function inside the getCameraIntrinsics function)
        HINT: What is the size of V? 2n x 6
        '''
        ########## Code starts here ##########


        # We will implement what is mentioned inn section 3.1
        num = len(H)

        # Let's construct V matrix 
        V = np.zeros((2*num,6))

        # In order to find V; I need to find v12, v11 and v22| This will become very long if we want to type this 3 different times
        # Let's write a lambda function for v
        # In python indices start from 0, so we modify our function as given in equation (7)
        v = lambda h,i,j: np.array([h[i-1, 0]*h[j-1, 0],
                                          h[i-1, 0]*h[j-1, 1]+h[i-1, 1]*h[j-1, 0],
                                          h[i-1, 1]*h[j-1, 1], 
                                          h[i-1, 2]*h[j-1, 0]+h[i-1, 0]*h[j-1, 2],
                                          h[i-1, 2]*h[j-1, 1]+h[i-1, 1]*h[j-1, 2], 
                                          h[i-1, 2]*h[j-1, 2]])
        # Let's construct our matrix V
        for i in range(num):
            h = np.transpose(H[i])      # Constructing column vector from H
            V[2*i] = v(h,1,2)
            V[2*i+1] = v(h,1,1) - v(h,2,2)  # As shown in equation (8)

        # Finally running SVD as
        _,_,v_T = np.linalg.svd(V)
        B = v_T[-1,:]
        # Extracting elements of matrix B as 6D vectors in equation (6)
        B11, B12, B22, B13, B23, B33 = B

        # Now we will implement Appendix B to extract intrinsic parameters
        v0 = (B12*B13 - B11*B23)/(B11*B22-np.power(B12,2))
        lamb = B33 - ((np.power(B12,2) + v0*(B12*B13 - B11*B23))/B11)
        alpha = np.sqrt(lamb/B11)
        beta = np.sqrt((lamb*B11)/(B11*B22 - np.power(B12,2)))
        gamma = -B12 * np.power(alpha,2)*beta/lamb
        u0 = (gamma*v0/beta) - (B13*np.power(alpha,2)/lamb)

        # Finally forming matrix A
        A = np.array([[alpha,gamma,u0], [0, beta, v0], [0, 0, 1]]) # UPDATE ME from equation (1)


        ########## Code ends here ##########
        return A

    def getExtrinsics(self, H, A):    # Zhang 3.1, Appendix C
        '''
        Inputs:
            H: a single homography matrix
            A: the camera intrinsic matrix
        Outputs:
            R: the rotation matrix
            t: the translation vector
        '''
        ########## Code starts here ##########
        # UPDATE ME
       # Writing the constraints from equation (4)
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]

        # In section 3.1; we have lambda1 = 1/||A-1h1|| = lambda2 = 1/||A-1h2||
        # Let's verify if they are same
        lambda1 = 1/ np.linalg.norm(np.linalg.inv(A).dot(h1))
        lambda2 = 1/ np.linalg.norm(np.linalg.inv(A).dot(h2))

        # Let's choose which lambda we will use 
        # If there is a large difference we will take average; else go with lambda1
        if (lambda1-lambda2)>1e-6:
            lamb = (lambda1+lambda2)/2
        else:
            lamb = lambda1

        # Now, that we have lambda lets implement the functions as given in the bottom of section 3.1
        r1 = lamb*np.linalg.inv(A).dot(h1)
        r2 = lamb*np.linalg.inv(A).dot(h2)
        r3 = np.cross(r1,r2)    # Cross product
        t = lamb*np.linalg.inv(A).dot(h3)
        # Compound matrix R = [r1,r2,r3] doesn't satisfy the requirements

        # Now, we implement appendix C to get 
        Q = np.hstack((r1.reshape(-1,1),r2.reshape(-1,1),r3.reshape(-1,1)))
        # Using SVD to solve
        U, _,v_T = np.linalg.svd(Q)

        # UPDATE ME
        R = np.matmul(U,v_T)


        ########## Code ends here ##########
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x, y: the coordinates in the ideal normalized image plane

        '''
        ########## Code starts here ##########
        # UPDATE ME
        num = X.shape[0]    
        
        dummy = np.array([X[0],Y[0],Z[0],1]).reshape(4,1)
        for i in range(1,num):
            dummy = np.hstack((dummy, np.array([X[i], Y[i], Z[i], 1]).reshape(4, 1)))
           
        
        Rt = np.hstack((R, t.reshape(-1, 1))) # Rt matrix
        p = np.matmul(Rt, dummy) # 3*4 x 4*63 = 3*63
        p = np.divide(p, dummy[-1, :]) # Normalizing by last row
        
        x = p[0, :].tolist()
        y = p[1, :].tolist()
        z = Z # Z is an array of zeros

        ########## Code ends here ##########
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u, v: the coordinates in the ideal pixel image plane
        '''
        ########## Code starts here ##########
        # UPDATE ME
        num_points = X.shape[0]
        
        dummy = np.array([X[0], Y[0], Z[0], 1.]).reshape(4, 1)
        for i in range(1, num_points):
            dummy = np.hstack((dummy, np.array([X[i], Y[i], Z[i], 1.]).reshape(4, 1)))
            
        
        T = np.matmul(A, np.hstack((R, t.reshape(-1, 1)))) # 3x4
        p = np.matmul(T, dummy) # 3x63
        p = np.divide(p, p[-1, :])
        
        u = p[0, :].tolist()
        v = p[1, :].tolist()
        ########## Code ends here ##########
        return u, v

    def transformWorld2NormImageDist(self, X, Y, R, t, k):    # Zhang 3.3
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x_br, y_br: the real, normalized image coordinates         
        '''
        ########## Code starts here ##########        
        x, y = self.transformWorld2NormImageUndist(X, Y, np.zeros_like(X), R, t) # Creating a zeros array of same size as X
        x = np.array(x)
        y = np.arra(y)
        x_br = x + x * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2)) # Estimating new distance by adding distortion
        y_br = y + y * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        ########## Code ends here ##########        
        return x_br, y_br

    def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):    # Zhang 3.3
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters                     
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u_br, v_br: the real, observed image coordinates         
        '''
        ########## Code starts here ##########                
        # UPDATE ME
        u, v = self.transformWorld2PixImageUndist(X, Y, np.zeros_like(Z), R, t, A)
        u_0, v_0 = A[0, 2], A[1, 2]
        x, y = self.transformWorld2NormImageUndist(X, Y, np.zeros_like(Z), R, t)
        
        u = np.array(u)
        v = np.array(v)
        x = np.array(x) 
        y = np.array(y)
        u_0, v_0 = np.array([u_0]*u.shape[0]), np.array([v_0]*v.shape[0])
        u_br = u + (u - u_0) * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        v_br = v + (v - v_0) * (k[0]*(np.power(x, 2) + np.power(y, 2)) + k[1]*np.power((np.power(x, 2) + np.power(y, 2)), 2))
        ########## Code ends here ##########
        return u_br, v_br

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            if k[0] != 0:
                u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
                ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)



# coding: utf-8

# In[1]:

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import collections
from PIL import Image
get_ipython().magic('matplotlib qt')
# %matplotlib inline


# In[2]:

# Saves useful informations for both lines
class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last 50 fits of the line
        self.recent_xfitted = collections.deque(50*[0], 50)
        # average x values of the fitted line over the last 50 iterations
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        # polynomial coefficients over the last 5 iterations
        self.last_fits = collections.deque(5*[3*[0]], 5)
        # polynomial coefficients averaged over the last 5 iterations
        self.last_fits_average = np.mean(self.last_fits, axis=0)
        # polynomial coefficients for the most recent fit
        self.current_fit = collections.deque(1*[3*[0]], 1)
        # radius of curvature of the line in some units
        self.radius_of_curvature = collections.deque(5*[0], 5)
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # Take an array and multiply its elements by 1000
#         self.fois_mille = lambda x: [[1000*n[0],1000*n[1],1000*n[2]] for n in x]
        # difference in fit coefficients between last and new fits
        self.diffs = []
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.min_width = 10000
        self.max_width = 0


# In[3]:

# Saves parameters of image transformations
class image_transformation():

    def __init__(self):
        self.objpoints = []
        self.imgpoints = []
        self.nx = None
        self.ny = None
        self.gray = np.uint8()


# In[4]:

def set_objpoints_imgpoints(images):

    # Arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []

    # Prepare object points
    nx = 6
    ny = 9
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)  # x, y coordinates

    for fname in images:
        # Read in each image
        image = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)

    return objpoints, imgpoints, gray


# In[5]:

# Takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints, gray):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
#     dst = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# In[6]:

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(nx, ny, undist):
    # Use the OpenCV undistort() function to remove distortion
    #     undist = cv2.undistort(img, mtx, dist, None, mtx)

    #     img = cv2.imread(image)
    # RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    offset = 300  # offset for dst points
    # Grab the image shape
    img_size = (undist.shape[1], undist.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([(612, 440), (667, 440), (1100, 720), (210, 720)])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([[offset, -900], [img_size[0]-offset, -900],
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, Minv, undist


# In[7]:

def find_lanes(binary_warped, margin=150):

    mask_margin = 50
    left_fit = left_lane.last_fits[-1]
    right_fit = right_lane.last_fits[-1]
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    if left_lane.detected:

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - mask_margin)) & (
            nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + mask_margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        left_lane.last_fits.append(left_fit)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - mask_margin)) & (
            nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + mask_margin)))
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane.last_fits.append(right_fit)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    else:

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
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
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[
                            0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Save the current lane lines informations
        left_lane.detected = True
        left_lane.recent_xfitted.append(left_fitx)
        right_lane.recent_xfitted.append(right_fitx)
        left_lane.last_fits.append(left_fit)
        right_lane.last_fits.append(right_fit)
        left_lane.current_fit.append(left_fit)
        right_lane.current_fit.append(right_fit)
        left_lane.diffs = abs(
            left_lane.last_fits_average - left_lane.current_fit)
        right_lane.diffs = abs(
            right_lane.last_fits_average - right_lane.current_fit)
        left_lane.allx = leftx
        left_lane.ally = lefty
        right_lane.allx = rightx
        right_lane.ally = righty
        if abs(left_fitx[max(ploty)] - right_fitx[max(ploty)]) < right_lane.min_width:
            right_lane.min_width = abs(
                left_fitx[max(ploty)] - right_fitx[max(ploty)])
        if abs(left_fitx[max(ploty)] - right_fitx[max(ploty)]) > right_lane.max_width:
            right_lane.max_width = abs(
                left_fitx[max(ploty)] - right_fitx[max(ploty)])

    return left_fit, right_fit, left_lane_inds, right_lane_inds, nonzeroy, nonzerox, leftx, rightx, lefty, righty


# In[8]:

def calculate_radius_distance(img_width, ploty, leftx, rightx, lefty, righty):
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/730  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[
                      1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    # Calculate the distance between the center of the vehicule and the lane
    # lines
    distance = img_width - rightx[0] - leftx[0]
    text_right = 'Vehicule is {:.2f} m right of center'.format(
        float(abs(distance)*xm_per_pix))
    text_left = 'Vehicule is {:.2f} m left of center'.format(
        float(abs(distance)*xm_per_pix))
    text_distance = text_right if distance >= 0 else text_left

    # Save the datas for both lane lines
    left_lane.radius_of_curvature.append(left_curverad)
    right_lane.radius_of_curvature.append(right_curverad)
    left_lane.line_base_pos = 3.7/2 + distance*xm_per_pix
    right_lane.line_base_pos = 3.7/2 + distance*xm_per_pix

    return left_curverad, right_curverad, text_distance


# In[45]:

def process_image(img):

    kernel_size = 3

    img_undist = cal_undistort(img, objpoints, imgpoints, gray)

    top_down, Minv, undist = corners_unwarp(nx, ny, img_undist)

    gray_image = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    blur_gray = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    # Sobel x
    sobelx = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0,
                       ksize=3)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 80
    thresh_max = 250
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    test3 = np.uint8(255*sxbinary)

    hls = cv2.cvtColor(top_down, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    rgb = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)
    r_channel = rgb[:, :, 0]
    g_channel = rgb[:, :, 1]
    b_channel = rgb[:, :, 2]

    r_thresh_min = 0
    r_thresh_max = 30
    r_binary = np.ones_like(r_channel)
    r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 0
#     r_binary2 = np.uint8(255*r_binary)

    v_thresh_min = 110
    v_thresh_max = 255
    v_thresh = np.zeros_like(s_channel)
    v_thresh[(v_channel >= v_thresh_min) & (v_channel <= v_thresh_max)] = 1
#     v_thresh2 = np.uint8(255*v_thresh)

    v_sobelx = cv2.Sobel(v_thresh2, cv2.CV_64F, 1, 0,
                         ksize=21)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    v_abs_sobelx = np.absolute(v_sobelx)
    v_scaled_sobel = np.uint8(255*v_abs_sobelx/np.max(v_abs_sobelx))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(v_thresh == 1) | (sxbinary == 1)] = 1
    combined_binary[(r_binary == 0) & (combined_binary == 1)] = 0
#     combined_binary2 = np.uint8(255*combined_binary)

    binary_warped = np.copy(combined_binary)

    margin = 150

    left_fit, right_fit, left_lane_inds, right_lane_inds, nonzeroy, nonzerox, leftx, rightx, lefty, righty = find_lanes(
        binary_warped, margin)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_curverad, right_curverad, text_distance = calculate_radius_distance(
        binary_warped.shape[1], ploty, leftx, rightx, lefty, righty)

    warped = np.copy(binary_warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    text_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    trust = lambda t: (abs(t[-1][0] - t[-2][0]) > 0.0001) & (
        abs(t[-1][1] - t[-2][1]) > 0.01) & (abs(t[-1][2] - t[-2][2]) > 25)

    if trust(left_lane.last_fits):
        correction1 += "Left lane: probleme"
        couleur1 = (0, 0, 255)
    else:
        correction1 = "Left lane: tout va bien"
        couleur1 = (0, 255, 0)

    if trust(right_lane.last_fits):
        correction2 += "Right lane: probleme"
        couleur2 = (0, 0, 255)
    else:
        correction2 += "Right lane: tout va bien"
        couleur2 = (0, 255, 0)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))
#     vertices = np.array([[(612, 440),
#                 (667, 440),
#                 (1100, 720),
#                 (210, 720)]], dtype=np.int32)
#     cv2.fillPoly(img_undist,vertices, (0,0, 255))
    radius = ('Radius of curvature = {} m').format(np.int_((np.mean(
        left_lane.radius_of_curvature, axis=0) + np.mean(right_lane.radius_of_curvature, axis=0))/2))
    cv2.putText(text_warp, radius, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    cv2.putText(text_warp, text_distance, (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    cv2.putText(text_warp, correction1, (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 2, couleur1)
    cv2.putText(text_warp, correction2, (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 2, couleur2)

#     # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result, 1, text_warp, 1, 0)

    return result


# In[10]:

images = glob.iglob('camera_cal/calibration*.jpg')

init_parameters = image_transformation()

nx = init_parameters.nx
ny = init_parameters.ny
gray = init_parameters.gray
objpoints = init_parameters.objpoints
imgpoints = init_parameters.imgpoints

# Prepare object points
nx = 6
ny = 9
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)  # x, y coordinates

for fname in images:
    # Read in each image
    image = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)


# In[46]:

left_lane = Line()
right_lane = Line()

img1 = cv2.imread("test_images/Capture4.jpg")
im1 = process_image(img1)
plt.imshow(im1)
cv2.imshow('im2', im1)


# In[1]:

yellow_output = 'Advanced_Lane_Line2.mp4'
clip2 = VideoFileClip('project_video.mp4')

yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

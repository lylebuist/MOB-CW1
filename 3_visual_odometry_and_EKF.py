#!/usr/bin/env python
# coding: utf-8

# ## (Stereo) Visual Odometry

# In[1]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import cv2


# ### Dataset Handling
# Let's make a dataset handling object to make our data more accessible as we complete our task.

# In[2]:


class Dataset_Handler():
    def __init__(self, lidar=False, progress_bar=True):
        import pandas as pd
        import os
        import cv2

        self.left_image_files = os.listdir('./data/image_02/data')
        self.right_image_files = os.listdir('./data/image_03/data')
        self.left_image_files.sort()
        self.right_image_files.sort()

        self.num_frames = len(self.left_image_files)

        self.reset_frames()
            # Store original frame to memory for testing functions
        self.first_image_left = cv2.imread('./data/image_02/data/' 
                                               + self.left_image_files[0], 0)
        
        self.first_image_right = cv2.imread('./data/image_03/data/' 
                                               + self.right_image_files[0], 0)
        self.second_image_left = cv2.imread('./data/image_02/data/' 
                                               + self.left_image_files[1], 0)
        self.imheight = self.first_image_left.shape[0]
        self.imwidth = self.first_image_left.shape[1]
            
            
    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        self.images_left = (cv2.imread('./data/image_02/data/'+ name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread('./data/image_03/data/' + name_right, 0)
                            for name_right in self.right_image_files)


# #### disparity map and depth map
# Unlike the simulation dataset we used in the tutorial, this time we are going to use a real driving dataset. Therefore we don't have depth images that you need to use for PnP directly, however, we have stereo images in this dataset, so the depth map can be generated from that.

# In[3]:


def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
    '''
    Takes a left and right stereo pair of images and computes the disparity map for the left
    image. Pass rgb=True if the images are RGB.
    
    Arguments:
    img_left -- image from left camera
    img_right -- image from right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    rgb -- (bool) set to True if passing RGB images as input
    verbose -- (bool) set to True to report matching type and time to compute
    
    Returns:
    disp_left -- disparity map for the left camera image
    
    '''
    # Feel free to read OpenCV documentation and tweak these values. These work well
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * sad_window ** 2,
                                        P2 = 32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                       )
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)
    
    return disp_left


# In[4]:


handler = Dataset_Handler()
left = handler.first_image_left
right = handler.first_image_right
disp = compute_left_disparity_map(left, 
                                  right, 
                                  matcher='sgbm',
                                  verbose=True)
plt.figure(figsize=(11,7))
plt.imshow(disp)


# In[5]:


def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)
    
    Arguments:
    p -- projection matrix to be decomposed
    
    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector
    
    '''
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    '''
    Calculate depth map using a disparity map, intrinsic camera matrix, and translation vectors
    from camera extrinsic matrices (to calculate baseline). Note that default behavior is for
    rectified projection matrix for right camera. If using a regular projection matrix, pass
    rectified=False to avoid issues.
    
    Arguments:
    disp_left -- disparity map of left camera
    k_left -- intrinsic matrix for left camera
    t_left -- translation vector for left camera
    t_right -- translation vector for right camera
    
    Optional Arguments:
    rectified -- (bool) set to False if t_right is not from rectified projection matrix
    
    Returns:
    depth_map -- calculated depth map for left camera
    
    '''
    # Get focal length of x axis for left camera
    f = k_left[0][0]
    
    # Calculate baseline of stereo pair
    if rectified:
        b = t_right[0] - t_left[0] 
    else:
        b = t_left[0] - t_right[0]
        
    # Avoid instability and division by zero
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    # Make empty depth map then fill with depth
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left
    
    return depth_map
def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, 
                   rectified=True):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. If your projection
    matrices are not rectified, set rectified=False.
    
    Arguments:
    img_left -- image of left camera
    img_right -- image of right camera
    P0 -- Projection matrix for the left camera
    P1 -- Projection matrix for the right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM
    rgb -- (bool) set to True if images passed are RGB. Default is False
    verbose -- (bool) set to True to report computation time and method
    rectified -- (bool) set to False if P1 not rectified to P0. Default is True
    
    Returns:
    depth -- depth map for left camera
    
    '''
    # Compute disparity map
    disp = compute_left_disparity_map(img_left, 
                                      img_right, 
                                      matcher=matcher, 
                                      rgb=rgb, 
                                      verbose=verbose)
    # Decompose projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    # Calculate depth map for left camera
    depth = calc_depth_map(disp, k_left, t_left, t_right)
    
    return depth


# In[6]:


P0 = np.array([[7.188560e+02 ,   0.    ,  6.071928e+02,   4.538225e+01],
               [  0.    ,7.188560e+02 , 1.852157e+02,   -1.130887e-01],
               [  0.    ,   0.    ,   1.    ,   3.779761e-03]])
P1 = np.array([[7.188560e+02 ,   0.    ,  6.071928e+02,   -3.372877e+02],
               [  0.    ,7.188560e+02 , 1.852157e+02,   2.369057e+00],
               [  0.    ,   0.    ,   1.    ,   4.915215e-03]])


# In[7]:


depth = stereo_2_depth(left, 
                       right, 
                       P0, 
                       P1,
                       matcher='sgbm',
                       verbose=True)
plt.imshow(depth)


# In[8]:


# We can constuct a mask using this information like so
mask = np.zeros(handler.first_image_left.shape[:2], dtype=np.uint8)


# ### Your task:

# In[9]:


def extract_features(image, detector='orb', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image
    mask -- Masks for each input image specifying where to look for keypoints

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    if detector == 'orb':
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(image, mask)
    elif detector == 'sift':
        sift = cv2.SIFT_create(500)
        kp = sift.detect(image, mask)
        kp, des = sift.compute(image, kp)
    
    return kp, des


# In[10]:


def match_features(des1, des2, matching='BF', detector='sift', k=2):
    """
    Match features from two images
    You can set nfeatures to 500 for a faster computation

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for 
               the same query descriptor
    """
    
    if matching == 'FLANN':
        if detector == 'orb':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6,
                                key_size = 12,
                                multi_probe_level = 1)
            
        elif detector == "sift":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k)
    elif matching == 'BF':
        if detector == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
        else:    
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k)
        

    return matches


# In[11]:


def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    
    try: 
        for m,n in matches:
            if m.distance < dist_threshold * n.distance:
                filtered_match.append(m)
    except:
        for i in range(0, len(matches)-1):
            m1 = matches[i]
            m2 = matches[i + 1]
            if (m1.distance < dist_threshold*m2.distance):
                filtered_match.append(m1)
            
    return filtered_match


# In[12]:


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


# In[13]:


def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix 
    
    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image. 
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image. 
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    

    objectpoints = []
    for m in match:
        # Coordinates of m
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt
        
        # Get depth
        s = depth1[int(v1), int(u1)]
        
        # Check for valid scale values
        if s < max_depth:
            # Transform pixel coordinates to camera coordinates using the pinhole camera model
            p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))
            
            # Save the results
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)
        
    # Convert lists to numpy arrays
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)
    
    # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
    # Convert rotation vector to rotation matrix
 
    rmat, _ = cv2.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points


# ### check your implementation here

# In[14]:


image_left = handler.first_image_left
plt.imshow(image_left)
image_right = handler.first_image_right 
image_plus1 = handler.second_image_left 
depth = stereo_2_depth(image_left, image_right, P0,  P1, matcher='sgbm', verbose=True) 
kp0, des0 = extract_features(image_left, 'orb') 
kp1, des1 = extract_features(image_plus1, 'orb') 
matches = match_features(des0, des1, matching='BF', detector='orb') 
print('Number of matches before filtering:', len(matches)) 
matches = filter_matches_distance(matches, 0.8) 
print('Number of matches after filtering:', len(matches)) 
visualize_matches(image_left, kp0, image_plus1, kp1, matches)


# In[15]:


def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=None, 
                    stereo_matcher='sgbm', mask=None, depth_type='stereo'):
    '''
    Function to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Takes as input a Data_Handler object and optional parameters.
    
    Arguments:
    handler -- Data_Handler object instance
    
    Optional Arguments:
    detector -- (str) can be 'sift' or 'orb'. Default is 'sift'.
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'. Default is 'BF'.
    filter_match_distance -- (float) value for ratio test on matched features. Default is None.
    stereo_matcher -- (str) can be 'bm' (faster) or 'sgbm' (more accurate). Default is 'bm'.
    mask -- (array) mask to reduce feature search area to where depth information available.
    depth_type -- (str) can be 'stereo' or set to None to use Essential matrix decomposition.
                        Note that scale will be incorrect with no depth used.
   
    Returns:
    trajectory -- Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    
    '''
    P0 = np.array([[7.188560e+02 ,   0.    ,  6.071928e+02,   4.538225e+01],
               [  0.    ,7.188560e+02 , 1.852157e+02,   -1.130887e-01],
               [  0.    ,   0.    ,   1.    ,   3.779761e-03]])
    P1 = np.array([[7.188560e+02 ,   0.    ,  6.071928e+02,   -3.372877e+02],
               [  0.    ,7.188560e+02 , 1.852157e+02,   2.369057e+00],
               [  0.    ,   0.    ,   1.    ,   4.915215e-03]])
    
    # Report methods being used to user
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector), 
                                                                   matching))
    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))
        
    image_left = handler.first_image_left
    
    image_right = handler.first_image_right 
    image_left2 = handler.second_image_left 
    
    kp0, des0 = extract_features(image_left, detector, mask) 
    kp1, des1 = extract_features(image_left2, detector, mask) 
    
    matches = match_features(des0, des1, matching, detector) 
    
    matches = filter_matches_distance(matches, filter_match_distance) 
    
    depth = stereo_2_depth(image_left, image_right, P0, P1,stereo_matcher, verbose=False)
    
    # Create variables for computation
    trajectory = np.zeros((handler.num_frames+1, 3, 4))
    robot_pose = np.zeros((handler.num_frames + 1, 4, 4))
    
    k, _, _ = decompose_projection_matrix(P0)
    
    # Initialize camera pose
    robot_pose[0] = np.eye(4)
    
    # Estimate camera motion between a pair of images
    rmat, tvec, image1_points, image2_points = estimate_motion(matches, kp0, kp1, k, depth)
        
    # Determine current pose from rotation and translation matrices
    current_pose = np.eye(4)
    current_pose[0:3, 0:3] = rmat
    current_pose[0:3, 3] = tvec.T
        
    # Build the robot's pose from the initial position by multiplying previous and current poses
    robot_pose[1] = robot_pose[0] @ np.linalg.inv(current_pose)
        
    # Calculate current camera position from origin
   # position = robot_pose[1] @ np.array([0., 0., 0., 1.])
        
    # Build trajectory
    trajectory[0] = robot_pose[1][0:3]
    
    for i in range(1, handler.num_frames - 1):
        image_left = cv2.imread('./data/image_02/data/' + handler.left_image_files[i], 0)
        image_right = cv2.imread('./data/image_03/data/' + handler.right_image_files[i], 0)
        image_left2 = cv2.imread('./data/image_02/data/' + handler.left_image_files[i+1], 0)
        
        depth = stereo_2_depth(image_left, image_right, P0, P1,stereo_matcher, verbose=False)
        
        kp0, des0 = extract_features(image_left, detector, mask) 
        kp1, des1 = extract_features(image_left2, detector, mask) 
    
        matches = match_features(des0, des1, matching, detector) 
    
        matches = filter_matches_distance(matches, filter_match_distance) 
    
        #depth = stereo_2_depth(image_left, image_right, P0, P1, stereo_matcher, verbose=True)
        
        rmat, tvec, image1_points, image2_points = estimate_motion(matches, kp0, kp1, k, depth)
        
         # Determine current pose from rotation and translation matrices
        current_pose = np.eye(4)
        current_pose[0:3, 0:3] = rmat
        current_pose[0:3, 3] = tvec.T
        
        # Build the robot's pose from the initial position by multiplying previous and current poses
        robot_pose[i+1] = robot_pose[i] @ np.linalg.inv(current_pose)
        
        # Calculate current camera position from origin
  #      position = robot_pose[i+1] @ np.array([0., 0., 0., 1.])
        
        # Build trajectory
        trajectory[i] = robot_pose[i+1][0:3]

    return trajectory


# In[16]:


mask = np.zeros(handler.first_image_left.shape[:2], dtype=np.uint8)
ymax = handler.first_image_left.shape[0]
xmax = handler.first_image_left.shape[1]
cv2.rectangle(mask, (96,0), (xmax,ymax), (255), thickness = -1)
plt.imshow(mask)
# You can set nfeatures to 500 and use FLANN based matcher for a faster computation
trajectory = visual_odometry(handler,
                                        filter_match_distance=0.5, 
                                        detector='orb', # you can use your favourite detector and matcher
                                        matching='BF',
                                        stereo_matcher='bm',
                                        mask=mask)


# #### Load  gps data
# Given that GPS have a noise following a normal distribution of 0 mean and 0.5 standard deviation 

# In[17]:


gps_data = np.load('data/gps.npy')
gps_data.shape


# In[18]:


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.plot(trajectory[:, :, 3][:, 0], 
        trajectory[:, :, 3][:, 2], 
        label='estimated', color='orange')

ax.scatter(gps_data[:,0],gps_data[:,1],
        label='gps',color='red')
ax.legend()


# ##### You might need this in following part

# In[19]:


delta_pose = np.zeros((handler.num_frames-1,2))
bearing = np.zeros(handler.num_frames)
relative_angle = np.zeros(handler.num_frames-1)
distance = np.zeros(handler.num_frames-1)
for i in range(1,handler.num_frames):
    delta_pose[i-1,0] = trajectory[:, :, 3][i,0]-trajectory[:, :, 3][i-1,0]
    delta_pose[i-1,1] = trajectory[:, :, 3][i,2]-trajectory[:, :, 3][i-1,2]
    bearing[i] = np.arctan2(delta_pose[i-1,0],delta_pose[i-1,1])
    relative_angle[i-1] = bearing[i] - bearing[i-1]
    distance[i-1] = np.sqrt(np.square(delta_pose[i-1,0]) + np.square(delta_pose[i-1,1]))


# ## Extended Kalman Filter
# 
# Now please try to use EKF to fuse the visual odometry from cameras with gps.
# 
# For simplicity, set the state vector X = \begin{bmatrix}
#  b \\
#  x \\
#  y
# \end{bmatrix}
# ### Motion Model
# \begin{align}
# \mathbf{x}_{k} &= 
# \begin{bmatrix}
# 1 &0 & 0 \\
# 0 &1 & 0\\
# 0 &0 & 1
# \end{bmatrix}
# \mathbf{x}_{k-1} +
# \begin{bmatrix}
# 1 &0 \\
# 0 &sin(b)\\
# 0 &cos(b)
# \end{bmatrix}
# \left(
# \begin{bmatrix}
# \theta \\
# d
# \end{bmatrix}
# + \mathbf{w}_k
# \right)
# \, , \, \, \, \, \, \mathbf{w}_k = \mathcal{N}\left(\mathbf{0}, \mathbf{Q}\right)
# \end{align}
# 
# - $\mathbf{x}_k = \left[ b \, x \, y  \right]^T$ is the current bearing and 2d position of the vehicle
# - $\theta $ is the change in bearing between frame k-1 and k, data is stored in "relative_angle"
# - $d$ is the distance traveled between frame k-1 and k, data is stored in "distance"
# 
# The process noise $\mathbf{w}_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{Q}$.
# 
# 
# ### Measurement Model
# 
# The measurement model from gps $\mathbf{y}_k = \left[x \, y \right]^T$.
# 
# \begin{align}
# \mathbf{y}_k =
# \begin{bmatrix}
# 0 & 1 & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# x_k
# + \mathbf{n}_k
# \, , \, \, \, \, \, \mathbf{n}_k = \mathcal{N}\left(\mathbf{0}, \mathbf{R}\right)
# \end{align}
# 
# 
# The gps measurement noise $\mathbf{n}_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{R}$.

# In[20]:


from numpy.linalg import inv
init_x = np.array([0,0,0])
x_check_hist = [init_x]
P_init = np.diag([0, 0, 0])
P_check_hist = [P_init]
th_var = 0.01
d_var = 0.01
Q_km = np.diag([th_var, d_var])
cov_y = np.diag([1, 1])

for i in range(handler.num_frames-1):
    b = x_check_hist[-1][0]
    d = distance[i]
    x_check = np.zeros((3,))
    x_check[0] = b + relative_angle[i]
    x_check[1] = x_check_hist[-1][1] + d*np.sin(b)
    x_check[2] = x_check_hist[-1][2] + d*np.cos(b)
    

    F_km = np.mat([[1,0,0],
                   [d*np.cos(b),1,0],
                   [-d*np.sin(b),0,1]])
    L_km = np.mat([[1, 0],
                   [0, np.sin(b)],
                   [0, np.cos(b)]])

    P_check = F_km.dot(P_check_hist[-1]).dot(F_km.T) + L_km.dot(Q_km).dot(L_km.T)
    
    H_k = np.matrix([[0, 1, 0],
                  [0, 0, 1]])
    M_k = np.eye(2)
    K_k = P_check.dot(H_k.T).dot(inv(H_k.dot(P_check).dot(H_k.T) + M_k.dot(cov_y).dot(M_k.T)))
    
    y_k_l_predict = np.zeros([2, 1])
    y_k_l_predict[0] = x_check_hist[-1][1] + d*np.sin(b)
    y_k_l_predict[1] = x_check_hist[-1][2] + d*np.cos(b)
    y_k_l = np.mat([[gps_data[i,0]], [gps_data[i,1]]])
    add = K_k.dot(y_k_l - y_k_l_predict)
    add = np.array(add).flatten()
    x_check = x_check + add
    P_check = (np.eye(3) - K_k.dot(H_k)).dot(P_check)
    x_check_hist.append(x_check)
    P_check_hist.append(P_check)


# In[26]:


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.plot(trajectory[:, :, 3][:, 0], 
        trajectory[:, :, 3][:, 2], 
        label='estimated', color='orange')
ax.scatter(gps_data[:,0],gps_data[:,1],
        label='gps',color='red',s=1)

xs = [x[1] for x in x_check_hist]
ys = [y[2] for y in x_check_hist]

ax.plot(xs, ys,label='EKF')#plot your EKF result here
ax.legend()


# In a similar vein, now consider a real-life fusion problem for IMU and WIFI localization. Unlike outdoor localization tasks, indoor localization tasks lack reliable GPS results and therefore often use technologies such as bluetooth, WiFi, etc. to aid localization. You are given the Informatics Forum first floor dense IMU odometer results and sparse WiFi positioning results, and use EKF to fuse them. Optionally you can incorporate the uncertainty from the WiFi localization result to the filter.
# 

# In[27]:


imu = pd.read_csv("./data/WiFi_IMU/pdr.csv")
imu = imu[['tmsp','x','y']].to_numpy() #time, odometery frame x, odometery frame y
wifi = pd.read_csv("./data/WiFi_IMU/wifi.csv")
wifi = wifi[['time','x','y','uncertainty_normalized']].to_numpy() #time, map frame x, map frame y, uncertainty


# In[28]:


### your implementation:


# Plot your result on the floor plan:

# In[29]:


def meter2pixel(x,y,fig_resolution=72,fig_scale=100):
    pix_x=x/0.0254/fig_scale*fig_resolution
    pix_y=-y/0.0254/fig_scale*fig_resolution
    return pix_x,pix_y

floorplan = plt.imread("./data/WiFi_IMU/F1.png")
gt = np.load("./data/WiFi_IMU/gt_9.npy")
gtx_pixel,gty_pixel= meter2pixel(gt[:,1],gt[:,2])

plt.imshow(floorplan)
plt.plot(gtx_pixel,gty_pixel,label="GT")
plt.legend()


# In[ ]:





# In[ ]:





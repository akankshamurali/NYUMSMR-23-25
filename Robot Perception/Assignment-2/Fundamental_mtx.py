import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
 
path = './data/Task3/'

def calib_camera():
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 8)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
     
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
     
     
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
     
    # Extracting path of individual image stored in a given directory
    for i in range(1,12):
        img = cv.imread(path+str(i)+'.jpg')
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
         
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
             
            imgpoints.append(corners2)
     
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
         
        #cv.imshow('img',img)
        #cv.waitKey(0)
     
    cv.destroyAllWindows()
     
    h,w = img.shape[:2]
     
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx



r_image = cv.imread(path+'right.jpg', cv.COLOR_BGR2GRAY)
l_image = cv.imread(path+'left.jpg', cv.COLOR_BGR2GRAY)


# Detect Aruco markers and retrive corners and marker IDs
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
parameters =  cv.aruco.DetectorParameters_create()
markerCorners_l, markerIds_l, _ = cv.aruco.detectMarkers(l_image, dictionary, parameters = parameters)
markerCorners_r, markerIds_r, _ = cv.aruco.detectMarkers(r_image, dictionary, parameters = parameters)

# Type cast IDs and corners to np array
markerIds_r = np.squeeze(markerIds_r.astype(np.int32))      
markerIds_l = np.squeeze(markerIds_l.astype(np.int32))

for j in range(len(markerCorners_r)):
    markerCorners_r[j] = np.squeeze(markerCorners_r[j])
    markerCorners_l[j] = np.squeeze(markerCorners_l[j])
    
# Identify Aruco marker with same ID
for i, id in enumerate(markerIds_l):
    if (markerIds_l[i] != markerIds_r[i]):
        j = np.where(markerIds_r == markerIds_l[i])
        j = np.squeeze(j)
        markerIds_r[i], markerIds_r[j] = markerIds_r[j], markerIds_r[i]
        markerCorners_r[i], markerCorners_r[j] = markerCorners_r[j], markerCorners_r[i]
        
# Extract match points in order        
l_pts = []
r_pts = []

for k in range(len(markerCorners_r)):
    a = np.squeeze(markerCorners_l[k])
    b = np.squeeze(markerCorners_r[k])

    n_itr = a.shape[0]
    for t in range(n_itr):
        l_pts.append(a[t])
        r_pts.append(b[t])
        
l_pts = np.int32(np.asarray(l_pts))
r_pts = np.int32(np.asarray(r_pts))        

# Construct A matrix        
r = 32                  # Toatal no. of corners/ match points
A = np.zeros((r, 9))
for itr in range(r):
    x = l_pts[itr, 0]
    y = l_pts[itr, 1]
    x_bar = r_pts[itr, 0]
    y_bar = r_pts[itr, 1]
    A[itr, :] = np.array([x_bar * x, x_bar * y, x_bar, y_bar * x, y_bar * y, y_bar, x, y, 1])
    
# Generate Fundamental mtx using SVD 
u, s, vt = np.linalg.svd(A)

v = vt[-1, :].reshape(3, 3)

u_2, s_2, vt_2 = np.linalg.svd(v)
s_2[2] = 0
F = u_2 @ np.diag(s_2) @ vt_2
F = F / F[2, 2]
print("Fundamental Matrix:")
print(F)

# Section 2 Draw Epipolar lines
def drawlines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
          
        color = tuple(np.random.randint(0, 255, 3).tolist())
          
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
          
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

lines1 = cv.computeCorrespondEpilines(r_pts.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img1,img2 = drawlines(l_image, r_image, lines1, l_pts, r_pts)

lines2 = cv.computeCorrespondEpilines(l_pts.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3,img4 = drawlines(r_image, l_image, lines2, r_pts, l_pts)

plt.imshow(img1)
plt.imshow(img3)


# Get Intrinsic
K = calib_camera()

# Find Essential Mtx
E = np.transpose(K) @ F @ K

U, S, Vt = np.linalg.svd(E)
u3 = (U @ np.transpose([0, 0, 1])).reshape(3, 1)
W = [[0, -1, 0], 
     [1, 0, 0], 
     [0, 0, 1]]

# Get Pose
_, R, T, _ = cv.recoverPose(E, l_pts, r_pts)

print("Rotation:") 
print(R)
print("Translation:")
print(T)

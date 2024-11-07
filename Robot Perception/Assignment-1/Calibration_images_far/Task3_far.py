import numpy as np
import cv2 as cv
import glob
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*8,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:8].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    print(fname)
    # cv.waitKey(0)
    img=cv.resize(img,dsize=[720,720],fx=1,fy=1,interpolation=cv.INTER_AREA)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('img',gray)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(grey, (4,8), None)
    # print(img.shape)
    # print(ret)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(grey,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (4,8), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
 
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, grey.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("\n")
print("dist : \n")
print(dist)
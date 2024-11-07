import cv2
import numpy as np

# Load the image where the ArUco marker is located
img = cv2.imread('aruco1.jpg')  #Replace with different aruco images for more solutions 

#Resize and convert the image to grayscale
img = cv2.resize(img,dsize=[720,720], fx=1, fy=1, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Detect the markers in the image
detect= cv2.aruco.ArucoDetector(aruco_dict,parameters)
corners, ids, rejectedImgPoints = detect.detectMarkers(gray)

# Check if any markers are detected
if ids is not None:
    #Caliberated camera matrix and distortion coefficients
    K = np.array([[649.45, 0, 364.017],   
                  [0, 649.84, 371.88],
                  [0,   0,   1]])
    dist_coeffs = np.array([0.124, -0.280, 0, 0, 0.216])  

    #define the marker size
    marker_size = 0.01  

    # Define the 3D points for the corners of the ArUco marker in world coordinates
    marker_corners_3d = np.array([[-marker_size / 2, -marker_size / 2, 0],
                                  [ marker_size / 2, -marker_size / 2, 0],
                                  [ marker_size / 2,  marker_size / 2, 0],
                                  [-marker_size / 2,  marker_size / 2, 0]], dtype=np.float32)

    # Loop through detected markers
    for i in range(len(corners)):
        # Reshape the corners of the marker for use with solvePnP
        marker_corners_2d = corners[i].reshape((4, 2))

        # Estimate the pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(marker_corners_3d, marker_corners_2d, K, dist_coeffs)

        if success:
            # Draw the detected marker
            img = cv2.aruco.drawDetectedMarkers(img, corners)

            # Draw the 3D axis on the marker
            cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 0.001)

            # Define the 3D points of a cube (same size as the ArUco marker)
            pnt=np.array([
                [-0.005, -0.005, 0],
                [0.005, -0.005, 0],
                [0.005, 0.005, 0],
                [-0.005, 0.005, 0],
                [-0.005, -0.005, -0.01],
                [0.005, -0.005, -0.01],
                [0.005, 0.005, -0.01],
                [-0.005, 0.005, -0.01]
                ], dtype=np.float32)
            projected_points, _ = cv2.projectPoints(pnt, rvec, tvec, K, dist_coeffs)
            projected_points=np.round(projected_points,0).astype(int)
            for i in range(4):
                cv2.line(img, projected_points[i][0], projected_points[(i+1)%4][0], (0, 255, 0), 2)
                cv2.line(img, projected_points[i+4][0], projected_points[(i+1)%4 + 4][0], (0, 0, 255), 2)
                cv2.line(img, projected_points[i][0], projected_points[i+4][0],(255, 0, 0), 1)

    # Show the result with the 3D cube
    cv2.imshow('3D Cube on Aruco Marker', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No Aruco markers detected.")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# First ensure the path exists and is correct
path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\Task3_imgs"
if not os.path.exists(path):
    raise FileNotFoundError(f"The directory {path} does not exist. Please check your path.")

def get_approximate_K(img):
    """
    Returns an approximate camera intrinsic matrix based on image dimensions.
    Args:
        img: Input image to get dimensions from
    Returns:
        K: 3x3 camera intrinsic matrix
    """
    img_size = img.shape
    focal_length = img_size[1]  # Approximate focal length as image width
    center_x = img_size[1] / 2
    center_y = img_size[0] / 2
    
    # Create approximate intrinsic matrix
    K = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ])
    return K

def read_and_check_image(image_path):
    """
    Reads and validates an image file
    Args:
        image_path: Path to the image file
    Returns:
        img: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img

def drawlines(img1, img2, lines, pts1, pts2):
    """Draw epipolar lines and points."""
    height, width = img1.shape[:2]
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    # Define a color palette for consistent visualization
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = color_palette[i % len(color_palette)]  # Cycle through colors
        
        # Calculate line endpoints across the image width
        x0, y0 = 0, int(-r[2] / r[1]) if r[1] != 0 else 0
        x1, y1 = width, int(-(r[2] + r[0] * width) / r[1]) if r[1] != 0 else 0
        
        # Draw the line and points
        img1_copy = cv.line(img1_copy, (x0, y0), (x1, y1), color, 2)
        img1_copy = cv.circle(img1_copy, tuple(pt1), 6, color, -1)
        img2_copy = cv.circle(img2_copy, tuple(pt2), 6, color, -1)
    
    return img1_copy, img2_copy


# Read stereo images
right_path = os.path.join(path, 'right.jpg')
left_path = os.path.join(path, 'left.jpg')
r_image = read_and_check_image(right_path)
l_image = read_and_check_image(left_path)

# Detect Aruco markers and retrieve corners and marker IDs
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Detect markers
markerCorners_l, markerIds_l, _ = detector.detectMarkers(l_image)
markerCorners_r, markerIds_r, _ = detector.detectMarkers(r_image)

if markerIds_l is None or markerIds_r is None:
    raise ValueError("No markers detected in one or both images")

# Type cast IDs and corners to np array
markerIds_r = np.squeeze(markerIds_r.astype(np.int32))      
markerIds_l = np.squeeze(markerIds_l.astype(np.int32))

# Convert marker corners from tuple to list for modification
markerCorners_l = list(markerCorners_l)
markerCorners_r = list(markerCorners_r)

# Process marker corners
for j in range(len(markerCorners_r)):
    markerCorners_r[j] = np.squeeze(markerCorners_r[j])
    markerCorners_l[j] = np.squeeze(markerCorners_l[j])
    
# Identify Aruco marker with same ID
for i, id in enumerate(markerIds_l):
    if (markerIds_l[i] != markerIds_r[i]):
        # Find the index where the matching ID exists in markerIds_r
        j = int(np.where(markerIds_r == markerIds_l[i])[0][0])  # Get single integer index
        
        # Swap the IDs and corners
        markerIds_r[i], markerIds_r[j] = int(markerIds_r[j]), int(markerIds_r[i])
        markerCorners_r[i], markerCorners_r[j] = markerCorners_r[j].copy(), markerCorners_r[i].copy()
        
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
r = len(l_pts)  # Total no. of corners/ match points
A = np.zeros((r, 9))
for itr in range(r):
    x = l_pts[itr, 0]
    y = l_pts[itr, 1]
    x_bar = r_pts[itr, 0]
    y_bar = r_pts[itr, 1]
    A[itr, :] = np.array([x_bar * x, x_bar * y, x_bar, y_bar * x, y_bar * y, y_bar, x, y, 1])

def visualize_epipolar_lines(left_img_path, right_img_path):
    # Calculate fundamental matrix and get corresponding points
    F, l_pts, r_pts, l_image, r_image = calculate_fundamental_matrix(left_img_path, right_img_path)
    
    # Define intrinsic matrix (replace with your camera's calibration values)
    K = np.array([
        [1000, 0, 640],  # fx = 1000, cx = 640
        [0, 1000, 360],  # fy = 1000, cy = 360
        [0, 0, 1]
    ])

# Generate Fundamental matrix using SVD 
u, s, vt = np.linalg.svd(A)
v = vt[-1, :].reshape(3, 3)
u_2, s_2, vt_2 = np.linalg.svd(v)
s_2[2] = 0
F = u_2 @ np.diag(s_2) @ vt_2
F = F / F[2, 2]
print("Fundamental Matrix:")
print(F)

# Draw epipolar lines
lines1 = cv.computeCorrespondEpilines(r_pts.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img1, img2 = drawlines(l_image, r_image, lines1, l_pts, r_pts)

lines2 = cv.computeCorrespondEpilines(l_pts.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(r_image, l_image, lines2, r_pts, l_pts)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.title('Left Image with Epipolar Lines')
plt.subplot(122)
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title('Right Image with Epipolar Lines')
plt.show()

# Get Intrinsic matrix using image dimensions
K = get_approximate_K(l_image)

# Find Essential Matrix
E = np.transpose(K) @ F @ K

# Decompose Essential Matrix
U, S, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], 
              [1, 0, 0], 
              [0, 0, 1]])

# Get Pose
_, R, T, _ = cv.recoverPose(E, l_pts, r_pts, K)

print("\nRotation Matrix:") 
print(R)
print("\nTranslation Vector:")
print(T)
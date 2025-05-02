import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_approximate_K(img):
    """
    Returns an approximate camera intrinsic matrix based on image dimensions.
    """
    img_size = img.shape
    focal_length = img_size[1]  # Approximate focal length as image width
    center_x = img_size[1] / 2
    center_y = img_size[0] / 2
    
    K = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ])
    return K

def estimate_fundamental_matrix(img1_path, img2_path):
    """
    Estimate the fundamental matrix with focus on AR marker corners
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use SIFT with parameters tuned for high-contrast markers
    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.02,  # Lower threshold to detect marker corners
        edgeThreshold=20
    )
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # FLANN matching parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Filter matches focusing on marker regions
    good_matches = []
    pts1 = []
    pts2 = []
    
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            # Get point coordinates
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            
            # Only keep points that are likely on markers (you might need to adjust these thresholds)
            if (100 < pt1[0] < img1.shape[1]-100 and  # x coordinates
                100 < pt1[1] < img1.shape[0]-100):    # y coordinates
                good_matches.append(m)
                pts1.append(pt1)
                pts2.append(pt2)
    
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    # Compute fundamental matrix with stricter parameters
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )
    
    # Enforce rank-2 constraint
    U, S, Vh = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vh
    
    # Normalize F
    F = F / np.linalg.norm(F)
    
    # Select inlier points
    mask = mask.ravel() == 1
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    
    # Select points more likely to be on markers
    if len(pts1) > 8:
        # Compute point distances from image center
        center = np.array([img1.shape[1]/2, img1.shape[0]/2])
        distances = np.linalg.norm(pts1 - center, axis=1)
        
        # Select points closer to the center where markers are likely to be
        closest_indices = np.argsort(distances)[:8]
        pts1 = pts1[closest_indices]
        pts2 = pts2[closest_indices]
    
    return F, pts1, pts2, img1, img2


    # Normalize points before computing F matrix
    
    def normalize_points(pts):
        centroid = np.mean(pts, axis=0)
        std_dev = np.std(pts[:, 0]**2 + pts[:, 1]**2)
        norm_factor = np.sqrt(2) / std_dev
        
        T = np.array([
            [norm_factor, 0, -norm_factor*centroid[0]],
            [0, norm_factor, -norm_factor*centroid[1]],
            [0, 0, 1]
        ])
        
        pts_homog = np.column_stack([pts, np.ones(len(pts))])
        normalized_pts = (T @ pts_homog.T).T
        return normalized_pts[:, :2], T
    
    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Calculate fundamental matrix using normalized points
    F_norm, mask = cv2.findFundamentalMat(
        pts1_norm, 
        pts2_norm, 
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=0.5,  # Reduced threshold for stricter RANSAC
        confidence=0.999
    )
    
    # Denormalize F matrix
    F = T2.T @ F_norm @ T1
    
    # Enforce rank-2 constraint using SVD
    U, S, Vh = np.linalg.svd(F)
    S[2] = 0  # Force smallest singular value to 0
    F = U @ np.diag(S) @ Vh
    
    # Normalize F matrix
    F = F / F[2,2]
    
    # Select only inlier points
    mask = mask.ravel() == 1
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    
    # Keep only a subset of points for visualization
    if len(pts1) > 20:  # Increased number of visualization points
        indices = np.random.choice(len(pts1), 20, replace=False)
        pts1_vis = pts1[indices]
        pts2_vis = pts2[indices]
    else:
        pts1_vis = pts1
        pts2_vis = pts2
    
    return F, pts1_vis, pts2_vis, img1, img2

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    """
    Draw epipolar lines in both images with improved visualization
    """
    def draw_lines(img, lines, pts):
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Define a fixed set of colors for consistency
        colors = [
            (0, 0, 255),   # Red
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0), # Cyan
            (128, 0, 0),   # Dark Red
            (0, 128, 0)    # Dark Green
        ]
        
        # Draw lines extending across the full image width
        for idx, (r, pt) in enumerate(zip(lines, pts)):
            color = colors[idx % len(colors)]
            # Extend lines to image boundaries
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [w, -(r[2] + r[0]*w)/r[1]])
            
            # Draw the extended line
            cv2.line(img_copy, (x0, y0), (x1, y1), color, 2)
            
            # Draw point with matching color
            cv2.circle(img_copy, tuple(map(int, pt)), 5, color, -1)
        
        return img_copy

    # Find epilines corresponding to points in right image, and draw them on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5 = draw_lines(img1, lines1, pts1)

    # Find epilines corresponding to points in left image, and draw them on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3 = draw_lines(img2, lines2, pts2)

    # Convert from BGR to RGB for matplotlib
    img5_rgb = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    # Create figure with specified size
    plt.figure(figsize=(15, 6))
    
    # Left image
    plt.subplot(121)
    plt.imshow(img5_rgb)
    plt.title('Left Image with Epipolar Lines')
    plt.axis('off')
    
    # Right image
    plt.subplot(122)
    plt.imshow(img3_rgb)
    plt.title('Right Image with Epipolar Lines')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

def recover_pose(F, K, pts1, pts2):
    """
    Recover relative pose (R, t) from fundamental matrix and calibration matrix
    Returns the most likely solution based on cheirality check
    """
    # Calculate essential matrix
    E = K.T @ F @ K
    
    # Recover pose using cv2.recoverPose which automatically selects the best solution
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t

def save_results(F, R, t, output_path="results.txt"):
    """
    Save the results to a text file
    """
    with open(output_path, 'w') as f:
        f.write("Fundamental Matrix:\n")
        f.write(str(F))
        f.write("\n\nRotation Matrix:\n")
        f.write(str(R))
        f.write("\n\nTranslation Vector:\n")
        f.write(str(t))

def main():
    # Image paths - replace with your image paths
    img1_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\Task3_imgs\\left.jpg"
    img2_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\Task3_imgs\\right.jpg"
    
    # Step 1: Estimate fundamental matrix and get matching points
    print("Estimating fundamental matrix...")
    F, pts1, pts2, img1, img2 = estimate_fundamental_matrix(img1_path, img2_path)
    print("\nFundamental Matrix:")
    print(F)
    
    # Step 2: Draw and display epipolar lines
    print("\nDrawing epipolar lines...")
    draw_epipolar_lines(img1, img2, pts1, pts2, F)
    
    # Step 3: Get camera intrinsic matrix K
    K = get_approximate_K(img1)
    print("\nCamera Intrinsic Matrix K:")
    print(K)
    
    # Step 4: Recover pose
    print("\nRecovering relative pose...")
    R, t = recover_pose(F, K, pts1, pts2)
    print("\nRotation Matrix R:")
    print(R)
    print("\nTranslation Vector t:")
    print(t)
    
    # Step 5: Save results
    print("\nSaving results to file...")
    save_results(F, R, t)
    print("Results saved to results.txt")

if __name__ == "__main__":
    main()

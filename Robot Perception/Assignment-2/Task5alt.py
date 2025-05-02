import cv2
import os
import shutil

def calculate_similarity(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Find keypoints and descriptors with SIFT in both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    
    # Calculate the percentage of good matches
    if len(kp1) == 0:
        return 0
    similarity = len(good_matches) / len(kp1)
    return similarity

# Paths to the datasets
database_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\database"
query_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\query"
detections_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\figs\\Task5\\detections"

# Create the detections directory if it doesn't exist
if not os.path.exists(detections_path):
    os.makedirs(detections_path)

# Load the query images in color (using cv2.IMREAD_COLOR)
query_images = []
for query_image_filename in os.listdir(query_path):
    query_image_path = os.path.join(query_path, query_image_filename)
    query_images.append((query_image_filename, cv2.imread(query_image_path, cv2.IMREAD_COLOR)))  # Load as color

# Process each query image independently
for query_image_name, query_image in query_images:
    match_found = False  # Track if a match is found for the current query image
    
    # Compare the query image with database images
    for database_image_filename in os.listdir(database_path):
        database_image_path = os.path.join(database_path, database_image_filename)
        database_image = cv2.imread(database_image_path, cv2.IMREAD_COLOR)  # Load as color

        # Calculate similarity
        similarity = calculate_similarity(query_image, database_image)
        if similarity > 0.9:
            print("Query Image: ", query_image_name)
            print("High similarity found: ", database_image_filename)
            print("Similarity Index: ", similarity)
            
            # Copy the matching database image to the detections folder
            detection_image_path = os.path.join(detections_path, database_image_filename)
            shutil.copy(database_image_path, detection_image_path)  # Copy the file
            
            match_found = True  # Mark that a match was found
            break  # Stop checking this query image once a match is found

    if match_found:
        print(f"Stopped further comparisons for query image: {query_image_name}")

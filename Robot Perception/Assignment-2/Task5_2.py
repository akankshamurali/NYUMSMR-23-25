import cv2 as cv
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
strt=time.time()
# Get image paths from database and target directories
images = glob.glob('C:\\Users\\heman\\Downloads\\task_5\\database\\*jpg')
target = glob.glob('C:\\Users\\heman\\Downloads\\task_5\\query\\*jpg')

# Initialize threshold and path lists
t = [0, 0, 0]
path = [None, None, None]
sift = cv.SIFT_create()
bf = cv.BFMatcher()

# Function to compare images
def compare_images(image_path, target_paths):
    results = []
    for i, target_path in enumerate(target_paths):
        if t[i]>0.85:
            continue
        img_db = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_target = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
        kp1, des1 = sift.detectAndCompute(img_db, None)
        kp2, des2 = sift.detectAndCompute(img_target, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        if len(kp2) == 0:
            return 0
        similarity = len(good_matches) / len(kp2)
        return similarity
        results.append((i, r, image_path))
    return results

# Compare images using parallel processing

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(compare_images, img, target) for img in images]
    for future in as_completed(futures):
        results = future.result()
        for i, r, img_path in results:
            if r > t[i]:
                t[i] = r
                path[i] = img_path

# Print the paths of the closest matching images
print(path)
print(time.time()-strt)
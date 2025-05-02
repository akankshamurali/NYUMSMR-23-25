import cv2 as cv
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Get image paths from database and target directories
images = glob.glob("C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\database\\*jpg")
target = glob.glob('C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\query\\*jpg')

# Initialize threshold and path lists
t = [100, 100, 100]
path = [None, None, None]

# Function to compare images
def compare_images(image_path, target_paths):
    results = []
    for i, target_path in enumerate(target_paths):
        if t[i]<15:
            continue
        img_db = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_target = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
        r = (img_db - img_target).mean()
        results.append((i, r, image_path))
    return results

# Compare images using parallel processing
strt=time.time()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(compare_images, img, target) for img in images]
    for future in as_completed(futures):
        results = future.result()
        for i, r, img_path in results:
            if r < t[i]:
                t[i] = r
                path[i] = img_path

# Print the paths of the closest matching images
print(path)
print(time.time()-strt)
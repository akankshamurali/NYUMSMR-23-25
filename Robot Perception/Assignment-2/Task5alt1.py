import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

class ImageMatcher:
    def __init__(self, similarity_threshold: float = 0.9):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.similarity_threshold = similarity_threshold
        
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract SIFT features from an image."""
        return self.sift.detectAndCompute(image, None)
    
    def calculate_similarity(self, des1: np.ndarray, des2: np.ndarray, kp1_len: int) -> Tuple[float, List]:
        """Calculate similarity between two sets of descriptors and return good matches."""
        if des1 is None or des2 is None or kp1_len == 0:
            return 0.0, []
            
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        similarity = len(good_matches) / kp1_len
        return similarity, good_matches

    def process_query_batch(self, query_batch: List[Tuple[str, np.ndarray, List, np.ndarray]], 
                          database_image: np.ndarray, database_filename: str,
                          detections_path: str) -> List[Tuple[str, str, float]]:
        """Process a batch of query images against one database image."""
        kp2, des2 = self.extract_features(cv2.cvtColor(database_image, cv2.COLOR_BGR2GRAY))
        results = []
        
        for query_filename, query_image, kp1, des1 in query_batch:
            similarity, good_matches = self.calculate_similarity(des1, des2, len(kp1))
            
            if similarity > self.similarity_threshold:
                results.append((query_filename, database_filename, similarity))
                
                # Copy the matched original database image to the detections directory
                matched_image_path = os.path.join(detections_path, database_filename)
                cv2.imwrite(matched_image_path, database_image)
                
                if similarity >= 0.9:
                    break
                
        return results

def load_images(directory: str) -> List[Tuple[str, np.ndarray]]:
    """Load all images from a directory with error handling."""
    images = []
    for filename in os.listdir(directory):
        try:
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load in color for visualization
            if image is not None:
                images.append((filename, image))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return images

def main():
    # Paths configuration
    database_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\database"
    query_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\query"
    detections_path = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\task_5\\figs\\Task5\\detections"
    
    # Create detections directory if needed
    os.makedirs(detections_path, exist_ok=True)
    
    # Initialize matcher
    matcher = ImageMatcher(similarity_threshold=0.9)
    
    # Load and preprocess query images
    query_images = load_images(query_path)
    preprocessed_queries = [
        (filename, image, *matcher.extract_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
        for filename, image in query_images
    ]
    
    # Process database images in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for db_filename, db_image in load_images(database_path):
            future = executor.submit(
                matcher.process_query_batch,
                preprocessed_queries,
                db_image,  # Pass the original database image
                db_filename,
                detections_path
            )
            futures.append(future)
        
        # Collect and print results
        for future in as_completed(futures):
            for query_filename, db_filename, similarity in future.result():
                print(f"Query Image: {query_filename}")
                print(f"High similarity found: {db_filename}")
                print(f"Similarity Index: {similarity}")

if __name__ == "__main__":
    main()
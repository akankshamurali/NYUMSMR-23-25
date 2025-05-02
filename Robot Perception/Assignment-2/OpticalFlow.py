import numpy as np
import cv2
import os

lk_params = dict(winSize=(20, 20),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=25,
                      qualityLevel=0.5,
                      minDistance=5,
                      blockSize=7)

trajectory_len = 80
detect_interval = 1
trajectories = []
frame_idx = 1

file_name = os.path.join(os.getcwd(),"C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\tracking.mp4")
cap = cv2.VideoCapture(file_name)

detections_path = os.path.join(os.getcwd(),"figs/")


while True:
    
    # Read frame
    captured, frame = cap.read()
    
    # Check if frame read
    if not captured:
        print("Error reading frame!")
        break
    
    # Create copy for manipulation
    img = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Optical flow claculation with LK Tracker
    if len(trajectories) > 0:
        # Get points to track
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        # Calculate optical flow from prev to current
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        # Calculate reverse optical flow from current to prev
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
        
        # Identifying good flow points
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = gray

    cv2.imshow('Optical Flow', img)
    
    '''
    if frame_idx == 50 or frame_idx == 51:
        dest_file_name = os.path.join(detections_path,"opticFlow_fr"+str(frame_idx)+".jpg")
        cv2.imwrite(img, dest_file_name)
        print("Write Successful!")
    '''
    
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
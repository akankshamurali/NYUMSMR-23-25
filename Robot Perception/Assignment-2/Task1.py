import math
import numpy as np
import open3d as o3d


# function to get the equation of a plane from 3 points
def plane(p1, p2, p3):
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = ((-1 * (a * p1[0])) - (b * p1[1]) - (c * p1[2]))
    return a, b, c, d


# RANSAC function to get the inlers and outleirs of the point cloud to fit the best plane
def RANSAC_plane(data, p, threshold):
    s = 3
    i_inl = []
    i_outl = []
    max_inlier_count = 0
    no_of_inliers = 0
    no_of_points = data.shape[0]
    N = 10000
    itr = 0
    
    while N > itr:
        
        # Select Random points
        idx = np.random.randint(no_of_points, size = s)
        p1 = data[idx[0], :]
        p2 = data[idx[1], :]
        p3 = data[idx[2], :]
        
        # Get Co-eff of Plane equation for random points
        a, b, c, d = plane(p1, p2, p3)

        # Get total inliers
        for i in range(no_of_points):
            nr = abs(((a * data[i, 0]) + (b * data[i, 1]) + (c * data[i, 2]) + d))
            dr = (math.sqrt((a * a) + (b * b) + (c * c)))
            dist = nr / dr
            if(abs(dist) < threshold):
                no_of_inliers += 1
                i_inl.append(i)
            else:
                i_outl.append(i)
        
        # Calculate iteration limit     
        ep = (1 - ((no_of_inliers) / (no_of_points)))
        N = min(N, int(math.log(1 - p) / math.log(1 - (1 - ep) ** s)))
        itr += 1
        
        # Reassign set with max inliers
        if(no_of_inliers > max_inlier_count):
            best_plane = [a, b, c, d]
            max_inlier_count = no_of_inliers
            idx_inliers = i_inl
            idx_outliers = i_outl
        
        # Reset all parameters
        no_of_inliers = 0
        i_inl = []
        i_outl = []
        
    # Return the best set of points.    
    idx_inliers = np.asarray(idx_inliers)
    idx_outliers = np.asarray(idx_outliers)
    return best_plane, max_inlier_count, idx_inliers, idx_outliers


# read demo point cloud provided by Open3D
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

# # function to visualize the point cloud
o3d.visualization.draw_geometries([pcd],
                                   zoom = 1,
                                   front = [0.4257, -0.2125, -0.8795],
                                   lookat = [2.6172, 2.0475, 1.532],
                                   up = [-0.0694, -0.9768, 0.2024])


data = np.asarray(pcd.points)

# RABSAC parametrs
p = 0.95
threshold = 0.03409
bestplane, inliers_count, inliers_index, outliers_index = RANSAC_plane(data, p, threshold)

# visulaizing after getting the inliers and outliers from RANSAC
inlier_cloud = pcd.select_by_index(inliers_index)
outlier_cloud = pcd.select_by_index(outliers_index)
inlier_cloud.paint_uniform_color([1, 0, 0])            # Set Inlier color to red

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                   zoom = 1,
                                   front = [0.4257, -0.2125, -0.8795],
                                   lookat = [2.6172, 2.0475, 1.532],
                                   up = [-0.0694, -0.9768, 0.2024])

import open3d as o3d
import copy
import numpy as np
import sys

# Load point clouds
#demo_icp_pcds = o3d.data.DemoICPPointClouds()
#source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
#target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

source = o3d.io.read_point_cloud("C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\kitti_frame1.pcd")
target = o3d.io.read_point_cloud("C:\\Desktop\\NYU\\3rd_sem\\Perception\\Assignment_2\\kitti_frame2.pcd")

def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    zoom=0.4459,
                                    front=[0.9288, -0.2951, -0.2242],
                                    lookat=[1.6784, 2.0612, 1.4451],
                                    up=[-0.3402, -0.9189, -0.1996])

def ICP(source, target, max_iterations=1000):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_temp_ = np.asarray(target_temp.points)
    
    N = len(target_temp_)
    H = np.eye(4)
    
    # Initialize optimizer params
    prev_cost = float('inf')
    curr_cost = 0
    
    # Use numpy arrays directly for better performance
    source_points_array = np.asarray(source_temp.points)
    target_points_array = np.asarray(target_temp.points)
    
    # Iterate over convergence 
    diff_cost = float('inf')
    iteration = 0
    
    while diff_cost > 0.001 and iteration < max_iterations:
        source_points = []
        
        # Represent source data as KD Tree
        source_tree = o3d.geometry.KDTreeFlann(source_temp)
        
        # Vectorize the point matching using numpy operations
        source_indices = []
        for i in range(N):  # Iterating over target points
            [k, idx, _] = source_tree.search_knn_vector_3d(target_temp.points[i], 1)
            source_indices.append(idx[0])
        
        source_points = source_points_array[source_indices]
        target_points = target_points_array
        n = len(source_points)
        
        # Calculate COM of point clouds using numpy operations
        source_mean = np.mean(source_points, axis=0).reshape(3, 1)
        target_mean = np.mean(target_points, axis=0).reshape(3, 1)
        com_source = source_points.T - source_mean
        com_target = target_points.T - target_mean
        
        # Estimate rotation with SVD
        K = com_target @ com_source.T
        U, _, Vt = np.linalg.svd(K)
        R = U @ Vt
        
        # Estimate Translation
        T = target_mean.flatten() - (R @ source_mean).flatten()
        
        # Compute iteration Cost
        curr_cost = np.linalg.norm(com_target - (R @ com_source))
        diff_cost = abs(prev_cost - curr_cost)
        
        # Print current cost on same line
        sys.stdout.write(f'\rIteration {iteration} - Cost: {curr_cost:.6f}')
        sys.stdout.flush()
        
        prev_cost = curr_cost
        
        # Build Homogenous matrix
        H_temp = np.eye(4)
        H_temp[:3, :3] = R
        H_temp[:3, 3] = T
        
        # Cumulative H
        H = H_temp @ H
        
        # Transform the source toward destination
        source_temp.transform(H_temp)
        source_points_array = np.asarray(source_temp.points)
        
        iteration += 1
    
    # Print final newline and summary
    print(f"\nICP finished after {iteration} iterations")
    return H

# Initial visualization
init_trans = np.eye(4)
init_trans[:3,3] = np.array([1,0,0]).T
draw_registration_result(source, target, transformation=init_trans)

# Perform ICP and show results
#voxel_Size=0.1
#transformation = ICP(source.voxel_down_sample(voxel_Size), target.voxel_down_sample(voxel_Size))
transformation = ICP(source, target)
draw_registration_result(source, target, transformation)
import sys
import argparse
import numpy as np

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
from utils import *
from planner_wrapper_coveragev2 import TomogramCoveragePlanner
import math
import time

sys.path.append('../')
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
args = parser.parse_args()

cfg = Config()

if args.scene == 'Building':
    # tomo_file = 'building2_9'
    tomo_file = 'experiments/1F_2*1'
    # tomo_file = 'experiments/Becker_office_eval'
    # tomo_file = 'building_LEE_1F'
    # tomo_file = 'ETH_HPH'


path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)
explored_cells_pub = rospy.Publisher("/explored_cells", PointCloud2, latch=True, queue_size=1)
planner = TomogramCoveragePlanner(cfg)

sampled_points_pub = rospy.Publisher("/sampled_points", PointCloud2, latch=True, queue_size=1)

def pct_plan():
    planner.loadTomogram(tomo_file)
    voxel_map_resolution = 0.2
    voxel_map_path = f"/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/{tomo_file}.pcd"
    planner.loadVoxelMap(voxel_map_path, voxel_map_resolution)

    # sensor_ranges = [1.5,2,2.5,3,3.5,4,6,8,10]
    # fovs = [ 60, 70, 80, 90, 100, 120, 130, 140, 150, 180, 200, 210, 220, 240, 250, 270, 300, 330, 360]
    # for fov in fovs:
    #     planner.sensor_fov = fov
    #     planner.sensor_range = 4.0
    #     planner.loadVoxelMap(voxel_map_path, voxel_map_resolution)
    #     start_time = time.time()
    #     candidate_points_idx, candidate_angles, candidate_points_xyz = planner.nextBestView()
    #     # planner.visualizeExploredGrid()
    #     elapsed = time.time() - start_time
    #     print(f" {elapsed:.4f} seconds")
    #     # print(f"Sensor Range: {sensor_range} meters")
    #     print(f"Sensor FOV: {fov} degrees")
    #     print("Candidate points:", candidate_points_xyz.shape)

# ################################################################
    # num_runs = 1
    # timings = []
    # for i in range(num_runs):
    #     planner.loadVoxelMap(voxel_map_path, voxel_map_resolution)
    #     start_time = time.time()
    #     candidate_points_idx, candidate_angles, candidate_points_xyz = planner.nextBestView()
    #     elapsed = time.time() - start_time
    #     timings.append(elapsed)
    #     print(f"Run {i+1}: {elapsed:.4f} seconds")
    # timings = np.array(timings)
    # print(f"\nRaw times: {timings}")
    # print(f"Average time: {np.mean(timings):.4f} seconds")
    # print(f"Standard deviation: {np.std(timings):.4f} seconds")
    computeNBVpoints()
    candidate_points_xyz = np.load("sampled_points.npy")
    candidate_points_idx = np.load("sampled_points_idx.npy").astype(np.int32)
    candidate_angles = np.load("sampled_points_angles.npy")
    print("Candidate points:", candidate_angles.shape)
    # publish_points(candidate_points_xyz)
    

# #################### Compute adjacency matrix computation ##############################
    time_starta = time.time()
    adjacency = planner.compute_adjacency_matrix(candidate_points_idx)
    time_enda = time.time()
    print("Candidate points:", candidate_points_xyz.shape)
    print(f"Adjacency matrix computed in {time_enda - time_starta:.2f} seconds")
    np.save("adjacency_matrix.npy", adjacency)
# ############################# Solving TSP problem ##############################
    adjacency_matrix = np.load("adjacency_matrix.npy")  

#     ## Optioal sometimes: make sure that the first candidate point is a valid view point (reachabl)
    i,j = 0, 0
    adjacency_matrix[[i, j], :] = adjacency_matrix[[j, i], :]
    adjacency_matrix[:, [i, j]] = adjacency_matrix[:, [j, i]]
    candidate_points_idx[[i, j]] = candidate_points_idx[[j, i]]
    candidate_points_xyz[[i, j]] = candidate_points_xyz[[j, i]]
    candidate_angles[[i, j]] = candidate_angles[[j, i]]
    # publish start point:
    start_point = np.array([candidate_points_xyz[0][0], candidate_points_xyz[0][1], candidate_points_xyz[0][2]], dtype=np.float32)
    print("Viewpoints:", candidate_points_idx)
    publish_points(start_point.reshape(1, 3), frame_id="map")

    
    # np.set_printoptions(threshold=np.inf)
    # print("Adjacency matrix:", adjacency_matrix)  
    time_start = time.time()
    updated_adjacency_matrix, updated_sampled_points_idx, updated_sampled_points_angles, updated_sampled_points_xyz = \
    remove_unreachable_nodes(adjacency_matrix, candidate_points_idx, candidate_angles, candidate_points_xyz)    # remove unreachable nodes
    time_end = time.time()
    print("Updated adjacency matrix:", updated_adjacency_matrix.shape)
    np.save("reachable_adjacency_matrix.npy", updated_adjacency_matrix)
    np.save("reachable_sampled_points_idx.npy", updated_sampled_points_idx)
    np.save("reachable_sampled_points_angles.npy", updated_sampled_points_angles)
    np.save("reachable_sampled_points.npy", updated_sampled_points_xyz)
    updated_adjacency_matrix = np.load("reachable_adjacency_matrix.npy")
    time_start1 = time.time()
    # tsp_path, tsp_cost = solve_tsp_local_search(updated_adjacency_matrix, x0=0) 
    # tsp_path, tsp_cost = solve_tsp_nearest_neighbor(updated_adjacency_matrix, start_node=0)
    tsp_path, tsp_cost = solve_tsp_simulated_annealing(updated_adjacency_matrix, x0=0)
    time_end1 = time.time()
    
    np.save("shortest_path_idx.npy", tsp_path)
    print("TSP Path:", tsp_path)
    print("TSP Cost:", tsp_cost)
    global_path = compute_global_path_idx(tsp_path, updated_sampled_points_idx)
    print("Global path:", global_path)
    time_start2 = time.time()
    full_trajectory,segment_trajectory = generate_global_trajectory(global_path, planner)
    time_end2 = time.time()
    candidate_points_xyz_path = candidate_points_xyz[tsp_path]
    np.save("candidate_points_xyz_path.npy", candidate_points_xyz_path)
    np.save("segment_trajectory.npy", segment_trajectory, allow_pickle=True)
    if len(full_trajectory) > 0:
        path_pub.publish(traj2ros(full_trajectory))
        np.save("full_trajectory.npy", full_trajectory)
        print("Full 3D trajectory published")
    else:
        rospy.logwarn("Failed to generate a full 3D trajectory")
    full_trajectory = np.load("full_trajectory.npy")
    path_pub.publish(traj2ros(full_trajectory))
    length = compute_trajectory_length(full_trajectory)
    print(f"Trajectory Length: {length:.2f} meters")
    explored_cells = planner.compute_and_visualise_explored_voxels(updated_sampled_points_xyz, updated_sampled_points_angles, False)
    publish_points(updated_sampled_points_xyz)
    area = compute_covered_area(explored_cells, planner.resolution_raycast)
    print(f"Covered Area: {area:.2f} m^2")
    print(f"TSP solved in {time_end1 - time_start1:.4f} seconds")
    print(f"Time taken to remove unreachable nodes: {time_end - time_start:.4f} seconds")
    print(f"Full trajectory generated in {time_end2 - time_start2:.4f} seconds")
    # print(f"Adjacency matrix computed in {time_enda - time_starta:.2f} seconds")
    print(f"Number of candidate points: {len(updated_sampled_points_idx)}")

def compute_explored_region(points_idx):
    candidate_points_idx = np.load("reachable_sampled_points_idx.npy").astype(np.int32)
    candidate_angles = np.load("reachable_sampled_points_angles.npy")
    base_angles = [0]
    Explored_cells = planner.initExplorationGraph()
    for point_index in points_idx:
        current_height = planner.elev_g[point_index[0], point_index[1], point_index[2]]
        same_height_layers = np.where(np.abs(planner.elev_g[:, point_index[1], point_index[2]] - current_height) < 0.5)[0]
        for base_angle in base_angles:
            angles = np.deg2rad(np.arange(base_angle - planner.sensor_fov / 2, base_angle + planner.sensor_fov / 2, step=10))
            for angle in angles:
                x_min = point_index[1]
                x_max = point_index[1] + math.floor(planner.sensor_range * np.cos(angle) / planner.resolution)
                y_min = point_index[2]
                y_max = point_index[2] + math.floor(planner.sensor_range * np.sin(angle) / planner.resolution)
                x_step = 1 if x_max >= x_min else -1
                y_step = 1 if y_max >= y_min else -1

                for i_x in range(x_min, x_max + x_step, x_step): 
                    stop = False
                    for i_y in range(y_min, y_max + y_step, y_step): 
                        if 0 <= i_x < planner.map_dim[0] and 0 <= i_y < planner.map_dim[1]:
                            for layer in same_height_layers:  
                                if Explored_cells[layer, i_x, i_y] == 0:
                                    Explored_cells[layer, i_x, i_y] = 1
                                if planner.trav[layer, i_x, i_y] == planner.cost_barrier:
                                    stop = True
                                    break
                        if stop:
                            break
    explore_area = np.nansum(Explored_cells) * planner.resolution ** 2
    return explore_area, Explored_cells

def compute_covered_area(explored_cells, resolution):
    num_covered = np.count_nonzero(explored_cells)
    area = num_covered * (resolution ** 2)
    return area


    
    
    
def generate_global_trajectory(global_path, planner):
    full_trajectory = []
    segment_trajectories = {}

    for i in range(len(global_path) - 1):
        start_pos = global_path[i]
        end_pos = global_path[i + 1]

        # Compute the 3D trajectory between the two points
        traj_3d = planner.plan_with_idx_online(start_pos, end_pos)
        print(f"Segment {i} to {i+1}: Start {start_pos}, End {end_pos}, Trajectory length: {len(traj_3d) if traj_3d is not None else 'None'}")
        if traj_3d is not None:
            full_trajectory.extend(traj_3d) 
            # Store the segment trajectory
            segment_trajectories[(i, i+1)] = np.array(traj_3d)
        else:
            rospy.logwarn(f"Failed to compute trajectory between {start_pos} and {end_pos}")
    return np.array(full_trajectory), segment_trajectories


def find_non_diagonal_inf(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and np.isinf(adjacency_matrix[i, j]):
                print(f"Non-diagonal inf found at ({i}, {j})")


def remove_unreachable_nodes(adjacency_matrix, sampled_points_idx, sampled_points_angles, sampled_points_xyz):
    n = adjacency_matrix.shape[0]
    rows_to_remove = []
    for i in np.arange(0, n):
        if i ==0:
            adjacency_matrix[i,i] = 0
        else:
            adjacency_matrix[i,i] = 0
            if adjacency_matrix[0,i] == np.inf:
                rows_to_remove.append(i)
    # Remove rows and columns corresponding to isolated nodes
    updated_adjacency_matrix = np.delete(adjacency_matrix, rows_to_remove, axis=0)
    updated_adjacency_matrix = np.delete(updated_adjacency_matrix, rows_to_remove, axis=1)
    updated_sampled_points_idx = np.delete(sampled_points_idx, rows_to_remove, axis=0)
    updated_sampled_points_angles = np.delete(sampled_points_angles, rows_to_remove, axis=0)
    updated_sampled_points_xyz = np.delete(sampled_points_xyz, rows_to_remove, axis=0)
    
    return updated_adjacency_matrix, updated_sampled_points_idx, updated_sampled_points_angles, updated_sampled_points_xyz


def computeNBVpoints():
    # Compute the next best view points
    start_time = time.time()
    candidate_points_idx, candidate_angles, candidate_points_xyz = planner.nextBestView()
    end_time = time.time()
    print(f"Time taken to compute NBV points: {end_time - start_time:.2f} seconds")
    planner.visualizeExploredGrid()
    print("Candidate points:", candidate_points_xyz)
    np.save("sampled_points.npy", candidate_points_xyz)
    np.save("sampled_points_idx.npy", candidate_points_idx)
    np.save("sampled_points_angles.npy", candidate_angles)
    np.save("sampled_points.npy", candidate_points_xyz)
    np.save("sampled_points_idx.npy", candidate_points_idx)
    np.save("sampled_points_angles.npy", candidate_angles)

def solve_tsp_nearest_neighbor(adjacency_matrix, start_node=0):
    n = adjacency_matrix.shape[0]
    visited = [False] * n
    path = [start_node]
    total_cost = 0

    current_node = start_node
    visited[current_node] = True

    for _ in range(n - 1):
        nearest_neighbor = None
        min_cost = float('inf')
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current_node, neighbor] < min_cost:
                nearest_neighbor = neighbor
                min_cost = adjacency_matrix[current_node, neighbor]

    
        if nearest_neighbor is None:
            rospy.logerr("No valid neighbor found. The graph might be disconnected.")
            return path, float('inf')  

        path.append(nearest_neighbor)
        total_cost += min_cost
        visited[nearest_neighbor] = True
        current_node = nearest_neighbor

    total_cost += adjacency_matrix[current_node, start_node]
    path.append(start_node)

    return path, total_cost

def compute_global_path_idx(tsp_path, candidate_points_idx):
    global_path = []
    for idx in tsp_path:
        global_path.append(candidate_points_idx[idx])
    return np.array(global_path)

def publish_points(points_xyz, frame_id="map"):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]

    point_cloud_msg = pc2.create_cloud(header, fields, points_xyz)

    # Publish the message
    sampled_points_pub.publish(point_cloud_msg)

def compute_trajectory_length(trajectory):
    if len(trajectory) < 2:
        return 0.0 
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_length = np.sum(distances)
    return total_length
if __name__ == '__main__':
    rospy.init_node("pct_planner", anonymous=True)

    pct_plan()

    rospy.spin()
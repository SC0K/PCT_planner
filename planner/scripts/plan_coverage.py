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
from planner_wrapper_coverage import TomogramCoveragePlanner
import math
import time

sys.path.append('../')
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
args = parser.parse_args()

cfg = Config()

if args.scene == 'Spiral':
    tomo_file = 'spiral0.3_2'
    start_pos = np.array([-16.0, -6.0], dtype=np.float32)
    end_pos = np.array([-26.0, -5.0], dtype=np.float32)
elif args.scene == 'Building':
    # tomo_file = 'building2_9'
    tomo_file = 'building_2F_4R'
    # tomo_file = 'building_LEE'
    start_pos = np.array([5.0, 4.0, 5], dtype=np.float32)
    end_pos = np.array([-6.0, -1.0, 5], dtype=np.float32)
else:
    tomo_file = 'plaza3_10'
    start_pos = np.array([0.0, 0.0], dtype=np.float32)
    end_pos = np.array([23.0, 10.0], dtype=np.float32)

path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)
explored_cells_pub = rospy.Publisher("/explored_cells", PointCloud2, latch=True, queue_size=1)
planner = TomogramCoveragePlanner(cfg)

sampled_points_pub = rospy.Publisher("/sampled_points", PointCloud2, latch=True, queue_size=1)

def pct_plan():
    planner.loadTomogram(tomo_file)

    # traj_3d = planner.plan(start_pos, end_pos)
    # if traj_3d is not None:
    #     path_pub.publish(traj2ros(traj_3d))
    #     print("Trajectory published")
#########################  Test sampled points ################################
    # sampled_points_idx, sampled_points_xyz = planner.sampleTraversablePoints( num_samples=1000)
    # sampled_points_idx, sampled_points_xyz = planner.sampleUniformPointsInSpace()
    # print("Candidate points:", sampled_points_xyz.shape)
    # publish_points(sampled_points_xyz)
   
########################## Test path planning between any two points ##############################
    # candidate_points_idx = np.array([[  0, 250, 176], [  0, 400, 170]])
    # print(planner.trav.shape)
    # print("-----------------heights:", planner.elev_g[candidate_points_idx[0][0], candidate_points_idx[0][1], candidate_points_idx[0][2]], planner.elev_g[candidate_points_idx[1][0], candidate_points_idx[1][1], candidate_points_idx[1][2]])
    # candidate_points_xyz = np.zeros_like(candidate_points_idx, dtype=np.float32)
    # candidate_points_xyz[0] = planner.idx2pos_3D(candidate_points_idx[0])
    # candidate_points_xyz[1] = planner.idx2pos_3D(candidate_points_idx[1])
    # publish_points(candidate_points_xyz)
    
    # traj_3d = planner.plan_with_idx(candidate_points_idx[0], candidate_points_idx[1])
    # if traj_3d is not None:
    #     path_pub.publish(traj2ros(traj_3d))
    #     print("Trajectory published")
# ################################################################
    # start_time = time.time()
    # computeNBVpoints()
    # end_time = time.time()
    # print(f"Time taken to compute NBV points: {end_time - start_time:.2f} seconds")
    
    candidate_points_xyz = np.load("sampled_points.npy")
    candidate_points_idx = np.load("sampled_points_idx.npy").astype(np.int32)
    candidate_angles = np.load("sampled_points_angles.npy")
    print("Candidate points:", candidate_points_idx.shape)
    publish_points(candidate_points_xyz)

#################### Compute adjacency matrix computation ##############################
    # Computation time ~ 60s for 60 points
    # adjacency = planner.compute_adjacency_matrix(candidate_points_idx)
    # print("Adjacency matrix:", adjacency)
    # np.save("adjacency_matrix.npy", adjacency)
# ############################# Solving TSP problem ##############################
    adjacency_matrix = np.load("adjacency_matrix.npy")  

#     ## Optioal sometimes: make sure that the first candidate point is a valid view point (reachabl)
    # i,j = 0, 1
    # adjacency_matrix[[i, j], :] = adjacency_matrix[[j, i], :]
    # adjacency_matrix[:, [i, j]] = adjacency_matrix[:, [j, i]]
    # candidate_points_idx[[i, j]] = candidate_points_idx[[j, i]]
    # candidate_points_xyz[[i, j]] = candidate_points_xyz[[j, i]]
    # candidate_angles[[i, j]] = candidate_angles[[j, i]]
    # publish start point:
    start_point = np.array([candidate_points_xyz[0][0], candidate_points_xyz[0][1], candidate_points_xyz[0][2]], dtype=np.float32)
    print("Viewpoints:", candidate_points_idx)
    publish_points(start_point.reshape(1, 3), frame_id="map")

    
    # np.set_printoptions(threshold=np.inf)
    # print("Adjacency matrix:", adjacency_matrix)  

    updated_adjacency_matrix, updated_sampled_points_idx, updated_sampled_points_angles, updated_sampled_points_xyz = \
    remove_unreachable_nodes(adjacency_matrix, candidate_points_idx, candidate_angles, candidate_points_xyz)    # remove unreachable nodes
    print("Updated adjacency matrix:", updated_adjacency_matrix.shape)
    np.save("reachable_adjacency_matrix.npy", updated_adjacency_matrix)
    np.save("reachable_sampled_points_idx.npy", updated_sampled_points_idx)
    np.save("reachable_sampled_points_angles.npy", updated_sampled_points_angles)
    np.save("reachable_sampled_points.npy", updated_sampled_points_xyz)
    updated_adjacency_matrix = np.load("reachable_adjacency_matrix.npy")
    tsp_path, tsp_cost = solve_tsp_nearest_neighbor(updated_adjacency_matrix, start_node=0)
    tsp_path, tsp_cost = solve_tsp_simulated_annealing(updated_adjacency_matrix, x0=0)
    publish_points(updated_sampled_points_xyz)
    tsp_path, tsp_cost = solve_tsp_local_search(updated_adjacency_matrix, x0=0) 
    ## TODO: recompute the explored region because some candidates that are not reachable are removed   
    print("TSP Path:", tsp_path)
    print("TSP Cost:", tsp_cost)
    global_path = compute_global_path_idx(tsp_path, updated_sampled_points_idx)
    print("Global path:", global_path)
    candidate_points_xyz = np.array([candidate_points_xyz[tsp_path[0]],candidate_points_xyz[tsp_path[-2]]], dtype=np.float32)
    full_trajectory = generate_global_trajectory(global_path, planner)
    # np.save("full_trajectory.npy", full_trajectory)
    if len(full_trajectory) > 0:
        path_pub.publish(traj2ros(full_trajectory))
        print("Full 3D trajectory published")
    else:
        rospy.logwarn("Failed to generate a full 3D trajectory")
    length = compute_trajectory_length(full_trajectory)
    print(f"Trajectory Length: {length:.2f} meters")
    explored_area, explored_cells = compute_explored_region(updated_sampled_points_idx)
    print(f"Explored Area: {explored_area:.2f} m^2")
    publish_explored_cells(
        explored_cells,
        planner.elev_g,
        planner.resolution,
        planner.center,
        planner.offset
    )

def compute_explored_region(points_idx):
    """
    Compute the explored region based on the adjacency matrix and candidate points.
    """
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




    
    
    
def generate_global_trajectory(global_path, planner):
    """
    Generate a 3D trajectory for the global path by concatenating the trajectories
    between consecutive points. Set the diagonal entries to 0.

    Args:
        global_path (np.ndarray): The global path as a sequence of 3D points.
        planner (TomogramCoveragePlanner): The planner object to compute trajectories.

    Returns:
        np.ndarray: The concatenated 3D trajectory for the global path.
    """
    full_trajectory = []

    for i in range(len(global_path) - 1):
        start_pos = global_path[i]
        end_pos = global_path[i + 1]

        # Compute the 3D trajectory between the two points
        traj_3d = planner.plan_with_idx(start_pos, end_pos)
        if traj_3d is not None:
            full_trajectory.extend(traj_3d)  # Append the trajectory to the full trajectory
        else:
            rospy.logwarn(f"Failed to compute trajectory between {start_pos} and {end_pos}")

    return np.array(full_trajectory)


def find_non_diagonal_inf(adjacency_matrix):
    """
    Find and print the indices of non-diagonal entries in the adjacency matrix that are `inf`.
    For debugging purposes.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
    """
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and np.isinf(adjacency_matrix[i, j]):
                print(f"Non-diagonal inf found at ({i}, {j})")


def remove_unreachable_nodes(adjacency_matrix, sampled_points_idx, sampled_points_angles, sampled_points_xyz):
    """
    Remove isolated nodes (with all `inf` in both their row and column excluding the diagonal).
    """
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
    candidate_points_idx, candidate_angles, candidate_points_xyz = planner.nextBestView()
    print("Candidate points:", candidate_points_xyz)

    # Publish sampled points
    # candidate_points_idx = candidate_points_idx[:, [0, 2, 1]].astype(np.int32)  # Switch the order of x and y for planning and ensure integers

    # Filter out points with the same x, y values in layers with the same mode heights
    unique_points_idx = []
    unique_points_xyz = []
    unique_angles = []
    seen_xy = {}
    for idx, point in enumerate(candidate_points_idx):
        s, x, y = point
        s = int(s)
        x = int(x)
        y = int(y)
        xy_key = (x, y)
        height = planner.elev_g[s, x, y]
        if xy_key not in seen_xy or np.abs(height - seen_xy[xy_key]) > 0.05:  
            unique_points_idx.append(point)
            unique_points_xyz.append(candidate_points_xyz[idx])  
            unique_angles.append(candidate_angles[idx])  
            seen_xy[xy_key] = planner.elev_g[s,x,y]

    candidate_points_idx = np.array(unique_points_idx, dtype=np.int32)
    candidate_points_xyz = np.array(unique_points_xyz, dtype=np.float32)
    candidate_angles = np.array(unique_angles, dtype=np.float32)

    # print("Filtered sampled points (indices):", candidate_points_idx)
    # print("Filtered sampled points (xyz):", candidate_points_xyz)
    # print("Filtered sampled points (angles):", candidate_angles)
    # Save the sampled points to a file
    np.save("sampled_points.npy", candidate_points_xyz)
    np.save("sampled_points_idx.npy", candidate_points_idx)
    np.save("sampled_points_angles.npy", candidate_angles)

def solve_tsp_nearest_neighbor(adjacency_matrix, start_node=0):
    """
    Solve the TSP using the Nearest Neighbor Heuristic. Return to the starting node.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
        start_node (int): The starting node for the TSP.

    Returns:
        list: The order of nodes in the TSP path.
        float: The total cost of the TSP path.
    """
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
    """
    Compute the global path in 3D space based on the TSP path.

    Args:
        tsp_path (list): The order of nodes in the TSP path.
        candidate_points_xyz (np.ndarray): The 3D coordinates of the candidate points.

    Returns:
        np.ndarray: The global path as a sequence of 3D points.
    """
    global_path = []
    for idx in tsp_path:
        global_path.append(candidate_points_idx[idx])
    return np.array(global_path)

def publish_points(points_xyz, frame_id="map"):
    """
    Publish sampled points as a PointCloud2 message.

    Args:
        sampled_points (np.ndarray): Array of sampled points (x, y indices).
        resolution (float): Resolution of the grid.
        center (np.ndarray): The center of the grid in map coordinates.
        frame_id (str): The frame ID for the PointCloud2 message.
    """
    #     # Convert sampled points to 3D coordinates (x, y, z)
    # points_3d = []
    # for s, x, y in sampled_points_idx:
    #     # Convert grid indices to map coordinates
    #     map_x = (x -  // 2) * resolution + center[0]
    #     map_y = (y -  // 2) * resolution + center[1]
    #     map_z = 0.0  # Assume z = 0 for visualization
    #     points_3d.append([map_x, map_y, map_z])

    # Create a PointCloud2 message
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

def publish_explored_cells(explored_cells, elev_g, resolution, center, offset, frame_id="map"):
    """
    Publish the explored cells as a PointCloud2 message for visualization in RViz.

    Args:
        explored_cells (np.ndarray): The explored cells array.
        elev_g (np.ndarray): The elevation grid.
        resolution (float): The resolution of the grid.
        center (np.ndarray): The center of the map.
        offset (np.ndarray): The offset of the grid.
        frame_id (str): The frame ID for the PointCloud2 message.
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]

    points = []
    for s in range(explored_cells.shape[0]):
        for x in range(explored_cells.shape[1]):
            for y in range(explored_cells.shape[2]):
                if explored_cells[s, x, y] > 0: 
                    map_x = (x - offset[0]) * resolution + center[0]
                    map_y = (y - offset[1]) * resolution + center[1]
                    map_z = elev_g[s, x, y] 
                    points.append([map_x, map_y, map_z])

    point_cloud_msg = pc2.create_cloud(header, fields, points)
    explored_cells_pub.publish(point_cloud_msg)

def compute_trajectory_length(trajectory):
    """
    Compute the total length of a 3D trajectory in meters.

    Args:
        trajectory (np.ndarray): The trajectory as a sequence of 3D points (N x 3).

    Returns:
        float: The total length of the trajectory in meters.
    """
    if len(trajectory) < 2:
        return 0.0 
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_length = np.sum(distances)
    return total_length
if __name__ == '__main__':
    rospy.init_node("pct_planner", anonymous=True)

    pct_plan()

    rospy.spin()
import os
import sys
import pickle
import numpy as np
import math
from scipy.stats import mode
from utils import *
import open3d as o3d
import cupy as cp

sys.path.append('../')
from lib import a_star, ele_planner, traj_opt

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class TomogramCoveragePlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate

        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir

        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_h0 = None
        self.slice_dh = None
        self.map_dim = []
        self.offset = None

        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)

        self.cost_barrier = self.cfg.planner.cost_barrier
        self.elev_g = None
        self.trav = None
        self.explored = None
        self.sensor_range = self.cfg.sensor.sensor_range
        self.sensor_range_analysis = 4
        self.sensor_fov = self.cfg.sensor.sensor_fov
        self.layer_modes = None
        self.fov_vert = 90
        self.fov_hor = 80

    def loadVoxelMap(self, pcd_file, voxel_size=0.2):
        """
        Load and voxelize a point cloud to create a voxel map.

        Args:
            pcd_file (str): Path to the point cloud file.
            voxel_size (float): Size of each voxel.
        """
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        voxel_centers = np.array([
            voxel_grid.get_voxel_center_coordinate(v.grid_index)
            for v in voxel_grid.get_voxels()
        ])
        voxel_indices = np.floor(voxel_centers / voxel_size).astype(np.int32)
        self.min_idx = voxel_indices.min(axis=0)
        self.max_idx = voxel_indices.max(axis=0)
        self.grid_shape = self.max_idx - self.min_idx + 1

        # Create hash grid
        shifted_indices = voxel_indices - self.min_idx
        self.hash_grid = cp.zeros(tuple(self.grid_shape), dtype=cp.bool_)
        for idx in shifted_indices:
            self.hash_grid[tuple(idx)] = True

        # Initialize explored voxels
        self.explored_voxels = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        self.voxel_size = voxel_size

    def loadTomogram(self, tomo_file):
        with open(self.tomo_dir + tomo_file + '.pickle', 'rb') as handle:
            data_dict = pickle.load(handle)

            tomogram = np.asarray(data_dict['data'], dtype=np.float32)

            self.resolution = float(data_dict['resolution'])
            self.center = np.asarray(data_dict['center'], dtype=np.double)
            self.n_slice = tomogram.shape[1]
            self.slice_h0 = float(data_dict['slice_h0'])
            self.slice_dh = float(data_dict['slice_dh'])
            self.map_dim = [tomogram.shape[2], tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)


        self.trav = tomogram[0]
        self.trav_gx = tomogram[1]
        self.trav_gy = tomogram[2]
        self.elev_g = tomogram[3]
        self.elev_g = np.nan_to_num(self.elev_g, nan=-100)
        self.elev_c = tomogram[4]
        self.elev_c = np.nan_to_num(self.elev_c, nan=1e6)
        self.trav_raw = tomogram[5]         # Adding raw trav cost (no inflation) for reward calculation

        
        self.initPlanner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
        # exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)
        # layers_t : travel cost
        # trav_grad_x : gradient x
        # trav_grad_y : gradient y
        # layers_g : ground height
        # layers_c : ceiling height

        # Initialize the explored graph
        self.explored = self.initExplorationGraph()
        self.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd", 0.2)

    def initExplorationGraph(self):
        """
        Initialize a graph to track whether cells in the elevation grid (elev_g) are explored.
    
        Returns:
            np.ndarray: A float array where -100 indicates ignored cells, 0.0 indicates unexplored cells, 
                        and 1.0 indicates explored cells.
        """
        # Initialize the exploration graph with NaN values
        exploration_graph = np.full(self.elev_g.shape, np.nan, dtype=np.float32)
        # Set cells with elev_g != -100 to 0.0 (unexplored)
        valid_mask = self.elev_g != -100
        exploration_graph[valid_mask] = 0.0
        return exploration_graph
    

    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]       # difference of travel cost between two slices
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])   # difference of elevation between two slices

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[1:]))
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[:-1]))
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2    # Boolean indexing
        self.gateway = gateway

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        self.planner.init_map(
            20, 15, self.resolution, self.n_slice, 0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )
        # print("Dimention of the elevation map:", self.elev_g.shape)
        # print("Dimention of the travel cost map:", self.trav.shape)
    def init_planner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        """
        Initialize the planner with the required maps and parameters.
    
        Args:
            trav (np.ndarray): Traversability map.
            trav_gx (np.ndarray): Gradient in the x direction of the traversability map.
            trav_gy (np.ndarray): Gradient in the y direction of the traversability map.
            elev_g (np.ndarray): Ground elevation map.
            elev_c (np.ndarray): Ceiling elevation map.
        """
        self.planner.init_map(
            20, 15, self.resolution, self.n_slice, 0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            self.gateway.reshape(-1, self.gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )
        
    def compute_adjacency_matrix(self, sampled_points_idx):
        """
        Compute an adjacency matrix where each entry represents the path length between two sampled points.
    
        Args:
            sampled_points_idx (np.ndarray): Array of sampled points' grid indices (N x 3).
    
        Returns:
            np.ndarray: Adjacency matrix of size N x N with path lengths.
        """
        num_points = sampled_points_idx.shape[0]
        adj_matrix = np.full((num_points, num_points), np.inf, dtype=np.float32)  # Initialize with infinity
    
        for i in range(num_points):
            for j in range(i + 1, num_points):  # Only compute for upper triangle (symmetry)
                # self.initPlanner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
                self.init_planner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
                # Plan a path between the two points
                print("Planning path between points:", sampled_points_idx[i], sampled_points_idx[j])
                # self.planner.plan(sampled_points_idx[i], sampled_points_idx[j], True)
                # Swap x and y for planning
                start_idx = np.array([sampled_points_idx[i][0], sampled_points_idx[i][2], sampled_points_idx[i][1]], dtype=np.int32)
                end_idx = np.array([sampled_points_idx[j][0], sampled_points_idx[j][2], sampled_points_idx[j][1]], dtype=np.int32)
                self.planner.plan(start_idx, end_idx, True)
                path_finder: a_star.Astar = self.planner.get_path_finder()
                path = path_finder.get_result_matrix()
    
                if len(path) > 0:  # If a valid path exists
                    path_length = len(path)  # Use the number of steps as the path length
                    adj_matrix[i, j] = path_length
                    adj_matrix[j, i] = path_length  # Symmetry for undirected graph
    
        return adj_matrix
        
    def plan_with_idx(self, start_pos, end_pos):
        # self.start_idx[1:] = self.pos2idx(start_pos)
        # self.end_idx[1:] = self.pos2idx(end_pos)
        # self.start_idx[:] = self.pos2idx_3D(start_pos)
        # self.end_idx[:] = self.pos2idx_3D(end_pos)
        

        self.start_idx = np.array([start_pos[0], start_pos[2], start_pos[1]], dtype=np.int32)   # planner needs s,y,x whereas the grid index is s,x,y
        self.end_idx = np.array([end_pos[0], end_pos[2], end_pos[1]], dtype=np.int32)
        print("start_idx:", self.start_idx)
        print("end_idx:", self.end_idx)

        self.planner.plan(self.start_idx, self.end_idx, True)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None

        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()

        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        return traj_3d
    
    def plan(self, start_pos, end_pos):
        # TODO: calculate slice index. By default the start and end pos are all at slice 0
        # self.start_idx[1:] = self.pos2idx(start_pos)
        # self.end_idx[1:] = self.pos2idx(end_pos)
        self.start_idx[:] = self.pos2idx_3D(start_pos)
        self.end_idx[:] = self.pos2idx_3D(end_pos)
        

        self.planner.plan(self.start_idx, self.end_idx, True)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None

        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()

        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        return traj_3d
    
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32) # Swap x and y for grid indexing
        return idx
    def pos2idx_3D(self, pos):
        """
        Convert a 3D position (x, y, z) to grid indices (s, y, x), where s is the layer number.
        
        Args:
            pos (np.ndarray): The 3D position (x, y, z).
        
        Returns:
            np.ndarray: The grid indices (s, y, x).
        """
        # Subtract the center to align with the grid
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy - self.center
        
        # Calculate x and y indices
        idx_xy = np.round(pos_xy / self.resolution).astype(np.int32) + self.offset
        idx_xy = np.array([idx_xy[1], idx_xy[0]], dtype=np.int32)  # Swap x and y for grid indexing
        
        # Search for the z index (layer number) using the precomputed layer modes
        z_height = pos[2]  # Extract the z-coordinate
        z_idx = -1  # Default to -1 if no valid layer is found
        for s in range(self.elev_g.shape[0]):
            print(f"Layer height {s}: {self.elev_g[s, idx_xy[1], idx_xy[0]]}")
            if abs(z_height - self.elev_g[s, idx_xy[1], idx_xy[0]]) <= self.resolution*2:
                z_idx = s
                break
        
        # Combine z_idx with x and y indices
        idx = np.array([z_idx, idx_xy[0], idx_xy[1]], dtype=np.float32)
        return idx
    def sampleUniformPointsInSpace(self):
        """
        Sample points that are uniformly distributed in space with a fixed distance equal to the sensor range
        in the x and y directions, and a smaller fixed step in the vertical (slice) direction.

        Note:
            the grid is indexed as (slice, y, x), where slice is the first dimension. The notaion in this function is reversed such that x_indices is actually the y dimension.
    
        Returns:
            np.ndarray: Array of valid sampled points (s, x, y indices).
            np.ndarray: Array of valid sampled points in map coordinates (x, y, z).
        """
        step_x = max(1, int(1.5 / self.resolution))  # Step size in the x dimension
        step_y = max(1, int(1.5 / self.resolution))  # Step size in the y dimension
        slice_indices = np.arange(0, self.elev_g.shape[0], 1)
        x_indices = np.arange(0, self.elev_g.shape[1], step_x)
        y_indices = np.arange(0, self.elev_g.shape[2], step_y)
        sampled_indices = np.array(np.meshgrid(slice_indices, x_indices, y_indices, indexing="ij"))
        sampled_indices = sampled_indices.reshape(3, -1).T  # Reshape to (N, 3)
    
        # Filter out invalid or untraversable points
        valid_indices = []
        for s, x, y in sampled_indices:
            if self.trav[s, x, y] < 30 and self.elev_g[s, x, y] > -90:
                valid_indices.append([s, x, y])
    
        valid_indices = np.array(valid_indices)
    
    # Filter out points with the same x, y indices and the same exact height in the elevation map
        unique_points = []
        seen_xy = {}
        for s, x, y in valid_indices:
            xy_key = (x, y)
            height = self.elev_g[s, x, y]
            if xy_key not in seen_xy or np.abs(seen_xy[xy_key] - height) > 0.05:
                unique_points.append([s, x, y])
                seen_xy[xy_key] = height

        unique_indices = np.array(unique_points, dtype=np.int32)

    
        # Convert valid indices to map coordinates
        sampled_xyz = np.empty((len(unique_indices), 3), dtype=np.float32)
        for idx, (s, x, y) in enumerate(unique_indices):
            map_x = (x - self.offset[0]) * self.resolution + self.center[0]
            map_y = (y - self.offset[1]) * self.resolution + self.center[1]
            map_z = self.elev_g[s, x, y]
            sampled_xyz[idx] = [map_x, map_y, map_z]
    
        return unique_indices, sampled_xyz
    
    def sampleTraversablePoints_rad(self, num_samples):
        """
        Sample a uniform set of traversable points from the travel cost map.
    
        Args:
            num_samples (int): The number of points to sample.
    
        Returns:
            np.ndarray: Array of sampled traversable points (x, y, z indices).
        """
        traversable_mask = (self.trav < self.cost_barrier) & (self.elev_g >= 0) 
        traversable_indices = np.argwhere(traversable_mask)
    
        if len(traversable_indices) <= num_samples:
            sampled_xyz = np.empty((len(traversable_indices), 3), dtype=np.float32)
            for idx, (s, x, y) in enumerate(traversable_indices):
                map_x = (x - self.offset[0]) * self.resolution + self.center[0]
                map_y = (y - self.offset[1]) * self.resolution + self.center[1]
                map_z = self.elev_g[s, x, y]
                sampled_xyz[idx] = [map_x, map_y, map_z]
            return traversable_indices, sampled_xyz
        sampled_idx = traversable_indices[
            np.random.choice(len(traversable_indices), num_samples, replace=False)
        ]
    
        sampled_xyz = np.empty((num_samples, 3), dtype=np.float32)
    
        for idx, (s, x, y) in enumerate(sampled_idx):
            map_x = (x - self.offset[0]) * self.resolution + self.center[0]
            map_y = (y - self.offset[1]) * self.resolution + self.center[1]
            map_z = self.elev_g[s, x, y]
            sampled_xyz[idx] = [map_x, map_y, map_z]
    
        return sampled_idx, sampled_xyz
    def idx2pos_3D(self, idx):
        """
        Convert grid indices to map coordinates.

        Args:
            idx (np.ndarray): The grid indices (s, x, y).

        Returns:
            np.ndarray: The map coordinates (x, y, z).
        """
        # Convert grid indices to map coordinates
        map_x = (idx[1] - self.offset[0]) * self.resolution + self.center[0]
        map_y = (idx[2] - self.offset[1]) * self.resolution + self.center[1]
        map_z = self.elev_g[idx[0], idx[1], idx[2]]
        return np.array([map_x, map_y, map_z], dtype=np.float32)
    
    def visualizeExploredGrid(self):
        """
        Visualize the explored grid using Open3D.
    
        Explored voxels are shown in one color (e.g., red), and unexplored voxels are shown in another (e.g., gray).
        """
        # Convert the explored voxel grid to a NumPy array
        explored_np = cp.asnumpy(self.explored_voxels)
    
        # Get the indices of all voxels
        voxel_indices = np.argwhere(explored_np)
    
        # Convert voxel indices to world coordinates
        voxel_centers = (voxel_indices + self.min_idx) * self.voxel_size
    
        # Create a point cloud
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
        # Assign colors based on whether the voxel is explored
        colors = []
        for idx in voxel_indices:
            if explored_np[tuple(idx)]:
                colors.append([1.0, 0.0, 0.0])  # Red for explored
            else:
                colors.append([0.5, 0.5, 0.5])  # Gray for unexplored
        vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
        # Visualize the point cloud
        o3d.visualization.draw_geometries([vis_pcd], window_name="Explored Grid")

    def nextBestView(self):
        """
        Calculate the next best view using the voxel map and raycasting across multiple orientations.
        Iteratively selects the best candidate pose, updates the explored voxel map, and removes the selected candidate.

        Returns:
            list: List of tuples containing the best candidate poses and their orientations.
        """
        min_reward = 50
        best_idxs = []
        best_angles = []
        best_xyz = []
        finished = False

        # Sample candidate points
        sampled_points_idx, sampled_points_xyz = self.sampleUniformPointsInSpace()
        sampled_points_xyz = sampled_points_xyz + np.array([0,0,1]) 

        # Define multiple yaw angles (in radians) for testing orientations
        yaw_angles = np.radians([0, 90, 180, 270])  # Example: 8 orientations

        while not finished and len(sampled_points_xyz) > 0:
            best_reward = -1
            best_pose = None
            best_orientation = None
            best_visible = None
            best_index = -1

            # Iterate over all candidate poses
            for i, candidate_pose in enumerate(sampled_points_xyz):
                for yaw in yaw_angles:
                    # Generate orientation matrix for the current yaw angle
                    orientation = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0, 0, 1]
                    ])

                    # Perform raycasting for the current pose and orientation
                    reward, visible = calculate_rewards_raycast(
                        candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape,
                        self.hash_grid, self.explored_voxels, fov_deg=self.sensor_fov,
                        max_range=self.sensor_range, resolution=self.resolution, n_rays=50
                    )

                    # Update the best pose and orientation if the reward is higher
                    if reward > best_reward:
                        best_reward = reward
                        best_pose = candidate_pose
                        best_orientation = orientation
                        best_visible = visible
                        best_index = i

            # Check if the best reward is below the minimum threshold
            if best_reward < min_reward:
                finished = True
                break

            # Update explored voxels
            for v in best_visible:
                print(f"Exploring voxel: {v}")
                local_idx = tuple(v - self.min_idx)
                self.explored_voxels[local_idx] = True

            # Store the best pose and orientation
            best_idxs.append(sampled_points_idx[best_index])
            best_angles.append(best_orientation)
            best_xyz.append(best_pose)

            # Remove the selected best candidate from the list
            sampled_points_xyz = np.delete(sampled_points_xyz, best_index, axis=0)

            print(f"Best pose: {best_pose}, Best orientation: {best_orientation}, Reward: {best_reward}")

        return best_idxs, best_angles, best_xyz
    def compute_explored_voxels(self, candidate_points_xyz, angles):
        """
        Compute the explored voxels based on candidate points and raycasting.

        Args:
            candidate_points_xyz (np.ndarray): Candidate points in world coordinates.
            angles (np.ndarray): Angles for each candidate point.

        Returns:
            cp.ndarray: Updated explored voxel grid.
        """
        
        # explored_voxels = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        explored_voxels_max = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        candidate_points_xyz = candidate_points_xyz + np.array([0,0,0.6])  # Adjust candidate points for z-axis
        explored_voxels_candidate = cp.zeros((len(candidate_points_xyz),) + self.hash_grid.shape, dtype=cp.bool_)
        for i,candidate_pose in enumerate(candidate_points_xyz):
            # Perform raycasting for the current pose
            print(f"Exploring candidate pose: {candidate_pose}")
            orientation = np.array([
                [np.cos(np.radians(angles[i])), -np.sin(np.radians(angles[i])), 0],
                [np.sin(np.radians(angles[i])),  np.cos(np.radians(angles[i])), 0],
                [0, 0, 1]
            ])
            visible = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape,self.hash_grid,self.fov_vert,self.fov_hor,self.sensor_range_analysis,
                self.resolution, n_rays=50)
            # Maximum possible visibility
            visible_max = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape,self.hash_grid,self.fov_vert, 360 , 10,
                self.resolution, n_rays=50)
            for v in visible_max:
                local_idx = tuple(v - self.min_idx)
                explored_voxels_max[local_idx] = True
            for v in visible:
                    local_idx = tuple(v - self.min_idx)
                    # explored_voxels[local_idx] = True
                    explored_voxels_candidate[i][local_idx] = True
        
        return explored_voxels_max, explored_voxels_candidate

    def compute_and_visualise_explored_voxels(self, candidate_points_xyz, angles):
        """
        Compute the explored voxels based on candidate points and raycasting.

        Args:
            candidate_points_xyz (np.ndarray): Candidate points in world coordinates.
            min_idx (np.ndarray): Minimum voxel indices.
            grid_shape (tuple): Shape of the voxel grid.
            hash_grid (cp.ndarray): Hash grid of occupied voxels.
            explored_voxels (cp.ndarray): Boolean grid of explored voxels.
            fov_deg (float): Field of view in degrees.
            max_range (float): Maximum sensor range.
            resolution (float): Raycasting resolution.
            n_rays (int): Number of rays.

        Returns:
            cp.ndarray: Updated explored voxel grid.
        """
        explored_voxels = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        candidate_points_xyz = candidate_points_xyz + np.array([0,0,1])  # Adjust candidate points for z-axis
        for i,candidate_pose in enumerate(candidate_points_xyz):
            # Perform raycasting for the current pose
            print(f"Exploring candidate pose: {candidate_pose}")
            orientation = np.array([
                [np.cos(np.radians(angles[i])), -np.sin(np.radians(angles[i])), 0],
                [np.sin(np.radians(angles[i])),  np.cos(np.radians(angles[i])), 0],
                [0, 0, 1]
            ])
            visible = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape,self.hash_grid, self.fov_vert,self.fov_hor, self.sensor_range_analysis,
                self.resolution, n_rays=60)
            
            for v in visible:
                    local_idx = tuple(v - self.min_idx)
                    explored_voxels[local_idx] = True
        
        explored_np = cp.asnumpy(explored_voxels)
        
        # Get the indices of all occupied voxels in the hash grid
        voxel_indices = np.argwhere(cp.asnumpy(self.hash_grid))
        
        # Convert voxel indices to world coordinates
        voxel_centers = (voxel_indices + self.min_idx) * self.voxel_size
        
        # Create a point cloud
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
        
        # Assign colors based on whether the voxel is explored
        # Vectorized operation to assign colors
        explored_mask = explored_np[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
        colors = np.zeros((voxel_indices.shape[0], 3))  # Initialize color array
        colors[explored_mask] = [1.0, 0.0, 0.0]  # Red for explored
        colors[~explored_mask] = [0.5, 0.5, 0.5]  # Gray for unexplored
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([vis_pcd], window_name="Explored Grid")
        
        return explored_voxels

def calculate_rewards_raycast(candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid, explored_voxels,
                              fov_deg=100, max_range=10.0, resolution=0.2, n_rays=30):
    """
    Calculate rewards using raycasting to determine visible voxels.

    Args:
        candidate_pose (np.ndarray): Candidate pose (x, y, z).
        orientation (np.ndarray): Orientation matrix (3x3).
        voxel_size (float): Size of each voxel.
        min_idx (np.ndarray): Minimum voxel indices.
        grid_shape (tuple): Shape of the voxel grid.
        hash_grid (cp.ndarray): Hash grid of occupied voxels.
        explored_voxels (cp.ndarray): Boolean grid of explored voxels.
        fov_deg (float): Field of view in degrees.
        max_range (float): Maximum sensor range.
        resolution (float): Raycasting resolution.
        n_rays (int): Number of rays.

    Returns:
        int: Reward (number of newly visible voxels).
        set: Set of newly visible voxel indices.
    """
    az = cp.linspace(-fov_deg / 2, fov_deg / 2, n_rays)
    el = cp.linspace(-fov_deg / 2, fov_deg / 2, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)  # (R, 3)
    dirs = dirs @ cp.asarray(orientation.T)

    dists = cp.arange(0, max_range, resolution)
    rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None]
    rays += cp.asarray(candidate_pose)

    idxs = cp.floor(rays / voxel_size).astype(cp.int32) - cp.asarray(min_idx)
    valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

    idxs_np = cp.asnumpy(idxs)
    valid_np = cp.asnumpy(valid)
    hash_np = cp.asnumpy(hash_grid)
    explored_np = cp.asnumpy(explored_voxels)

    visible = set()
    reward = 0
    for r in range(idxs_np.shape[0]):
        for s in range(idxs_np.shape[1]):
            if not valid_np[r, s]:
                continue
            i, j, k = idxs_np[r, s]
            if hash_np[i, j, k]:
                voxel_idx = (i + min_idx[0], j + min_idx[1], k + min_idx[2])
                if not explored_np[i, j, k]:
                    visible.add(voxel_idx)
                    reward += 1
                break
    return reward, visible
def get_visible_voxels_first_hit(candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid,
                                 fov_deg_ver=90,fov_deg_hor=90, max_range=4.0, resolution=0.2, n_rays=30):
    az = cp.linspace(-fov_deg_hor / 2, fov_deg_hor / 2, n_rays)
    el = cp.linspace(-45, 5, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)  # (R, 3)
    dirs = dirs @ cp.asarray(orientation.T)

    dists = cp.arange(0, max_range, resolution)
    rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None]
    rays += cp.asarray(candidate_pose)

    idxs = cp.floor(rays / voxel_size).astype(cp.int32) - cp.asarray(min_idx)
    valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

    idxs_np = cp.asnumpy(idxs)
    valid_np = cp.asnumpy(valid)
    hash_np = cp.asnumpy(hash_grid)

    visible = set()
    for r in range(idxs_np.shape[0]):
        for s in range(idxs_np.shape[1]):
            if not valid_np[r, s]:
                continue
            i, j, k = idxs_np[r, s]
            if hash_np[i, j, k]:
                visible.add((i + min_idx[0], j + min_idx[1], k + min_idx[2]))
                break 
    return visible
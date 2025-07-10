import os
import sys
import pickle
import numpy as np
import math
from scipy.stats import mode
from utils import *
from scipy.spatial import cKDTree
import time
import open3d as o3d
import csv


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
        self.sensor_fov = self.cfg.sensor.sensor_fov
        self.layer_modes = None

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
            # self.sensor_range = int(round(self.cfg.sensor.sensor_range / self.resolution))


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
    def re_init_planner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        """ without initializing the trajectory optimizer."""
        self.planner.reinit_map(
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
            sampled_points_idx (np.ndarray): Array of sampled points' grid indices (N x 3), (s, x, y).
        Returns:
            np.ndarray: Adjacency matrix of size N x N with path lengths.
        """
        # Flip x and y: (s, x, y) -> (s, y, x)
        sampled_points_flipped = sampled_points_idx.copy()
        sampled_points_flipped[:, 1], sampled_points_flipped[:, 2] = (
            sampled_points_flipped[:, 2], sampled_points_flipped[:, 1].copy()
        )
        sampled_points_o3d = o3d.utility.Vector3iVector(sampled_points_flipped.astype(np.int32))
        return self.planner.compute_adjacency_matrix(sampled_points_o3d)
        
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
        Sample points using Poisson Disk Sampling to ensure uniform spatial coverage
        with a minimum distance between candidate viewpoints.

        Returns:
            np.ndarray: Array of valid sampled points (s, x, y indices).
            np.ndarray: Array of valid sampled points in map coordinates (x, y, z).
        """
        # Project traversable 3D grid to 2D (top-down view)
        traversable_mask = (self.trav < self.cost_barrier) & (self.elev_g > -90)
        traversable_indices = np.argwhere(traversable_mask)
        xy_coords = np.array([
            [(x - self.offset[0]) * self.resolution + self.center[0],
            (y - self.offset[1]) * self.resolution + self.center[1]]
            for _, x, y in traversable_indices
        ])

        # Poisson disk sampling using Bridson's algorithm
        def bridson_sampling(width, height, r, k=30):
            cell_size = r / np.sqrt(2)
            grid_width = int(np.ceil(width / cell_size))
            grid_height = int(np.ceil(height / cell_size))
            grid = -np.ones((grid_height, grid_width), dtype=int)
            samples = []
            active_list = []

            def get_cell(p):
                return int(p[1] / cell_size), int(p[0] / cell_size)

            def in_bounds(p):
                return 0 <= p[0] < width and 0 <= p[1] < height

            def is_far_enough(p):
                gi, gj = get_cell(p)
                for i in range(max(gi - 2, 0), min(gi + 3, grid_height)):
                    for j in range(max(gj - 2, 0), min(gj + 3, grid_width)):
                        idx = grid[i, j]
                        if idx != -1:
                            if np.linalg.norm(samples[idx] - p) < r:
                                return False
                return True

            # Initial sample
            grid_spacing = r / np.sqrt(2)
            x_coords = np.arange(grid_spacing / 2, width, grid_spacing)
            y_coords = np.arange(grid_spacing / 2, height, grid_spacing)
            for x in x_coords:
                for y in y_coords:
                    sample = np.array([x, y])
                    samples.append(sample)
                    gi, gj = get_cell(sample)
                    grid[gi, gj] = len(samples) - 1
                    active_list.append(len(samples) - 1)

            while active_list:
                idx = np.random.choice(active_list)
                base = samples[idx]
                found = False
                for _ in range(k):
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(r, 2 * r)
                    offset = radius * np.array([np.cos(angle), np.sin(angle)])
                    new_point = base + offset
                    if in_bounds(new_point) and is_far_enough(new_point):
                        samples.append(new_point)
                        gi, gj = get_cell(new_point)
                        grid[gi, gj] = len(samples) - 1
                        active_list.append(len(samples) - 1)
                        found = True
                        break
                if not found:
                    active_list.remove(idx)

            return np.array(samples)

        # Fit bounding box
        xy_min = xy_coords.min(axis=0)
        xy_max = xy_coords.max(axis=0)
        width, height = xy_max - xy_min
        poisson_samples = bridson_sampling(width, height, r=2,k=10)
        poisson_samples += xy_min

        # Use k-d tree to snap Poisson samples to closest traversable locations
        tree = cKDTree(xy_coords)
        _, indices = tree.query(poisson_samples)
        selected_indices = traversable_indices[indices]

        # Remove duplicates
            # Filter out points with the same x, y indices and the same exact height in the elevation map
        unique_points = []
        seen_xy = {}
        for s, x, y in selected_indices:
            xy_key = (x, y)
            height = self.elev_g[s, x, y]
            if xy_key not in seen_xy or np.abs(seen_xy[xy_key] - height) > 0.05:
                unique_points.append([s, x, y])
                seen_xy[xy_key] = height

        unique_indices = np.array(unique_points, dtype=np.int32)

        # Convert to map coordinates
        sampled_xyz = np.empty((len(unique_indices), 3), dtype=np.float32)
        for idx, (s, x, y) in enumerate(unique_indices):
            map_x = (x - self.offset[0]) * self.resolution + self.center[0]
            map_y = (y - self.offset[1]) * self.resolution + self.center[1]
            map_z = self.elev_g[s, x, y]
            sampled_xyz[idx] = [map_x, map_y, map_z]

        return unique_indices, sampled_xyz

    def sampleUniformPointsInSpace_idle(self):
        """
        Sample points that are uniformly distributed in space with a fixed distance equal to the sensor range
        in the x and y directions, and a smaller fixed step in the vertical (slice) direction.

        Note:
            the grid is indexed as (slice, y, x), where slice is the first dimension. The notaion in this function is reversed such that x_indices is actually the y dimension.
    
        Returns:
            np.ndarray: Array of valid sampled points (s, x, y indices).
            np.ndarray: Array of valid sampled points in map coordinates (x, y, z).
        """
        step_x = max(2, int(1.05 / self.resolution))  # Step size in the x dimension
        step_y = max(2, int(1.05 / self.resolution))  # Step size in the y dimension
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
    
    def getExploredGraph(self):
        """
        Get the explored graph.

        Returns:
            np.ndarray: The explored graph.
        """
        return self.explored


    def nextBestView(self):
        """
        Calculate the reward for each sampled point based on the number of unseen cells in its neighborhood.
        Returns:
            np.ndarray: Array of rewards for each sampled point.
        """
        min_reward = 150
        finished = False
        sampled_points_idx, sampled_points_xyz = self.sampleUniformPointsInSpace_idle()
        target_num = np.count_nonzero(~np.isnan(self.explored))
        candidate_points_idx = np.empty((0, 3), dtype=np.int32)
        candidate_points_xyz = np.empty((0, 3), dtype=np.float32)
        candidate_points_angles = np.empty((0,), dtype=np.float32)
        num = 0

        # Prepare CSV file for timing logs
        timing_csv = os.path.join(self.tomo_dir, "cpu_reward_profile.csv")
        if not os.path.exists(timing_csv):
            with open(timing_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "n_candidates", "reward_time_ms", "total_time_ms"])

        while not finished:
            print("Explored cells:", np.nansum(self.explored))
            if np.nansum(self.explored) >= self.cfg.planner.coverage_threshold * target_num:
                break

            # --- Filter out candidates that are already in explored regions ---
            mask = []
            for idx in range(sampled_points_idx.shape[0]):
                s, x, y = sampled_points_idx[idx]
                if np.any(self.explored[:, x, y] == 0):
                    mask.append(True)
                else:
                    mask.append(False)
            mask = np.array(mask)
            sampled_points_idx = sampled_points_idx[mask]
            sampled_points_xyz = sampled_points_xyz[mask]
            if sampled_points_idx.shape[0] == 0:
                break
            # ---------------------------------------------------------------
            TIME_START = time.time()
            rewards, best_angles, explored_cells = calculate_rewards(
                sampled_points_idx,
                self.elev_g,
                self.trav,
                self.explored,
                self.sensor_range,
                self.sensor_fov,
                self.resolution,
                self.cost_barrier
            )
            TIME_END = time.time()
            reward_time_ms = (TIME_END - TIME_START) * 1000

            print(f"Time taken for reward calculation: {reward_time_ms:.4f} milliseconds")
            print(f"Number of candidate points: {sampled_points_idx.shape}")
            best_reward_index = np.argmax(rewards)
            best_reward = rewards[best_reward_index]

            if best_reward < min_reward:
                finished = True
                break

            best_angle = best_angles[best_reward_index]
            best_explored_cells = explored_cells[best_reward_index]

            self.explored = best_explored_cells
            candidate_points_idx = np.vstack((candidate_points_idx, sampled_points_idx[best_reward_index]))
            candidate_points_xyz = np.vstack((candidate_points_xyz, sampled_points_xyz[best_reward_index]))
            candidate_points_angles = np.append(candidate_points_angles, best_angle)
            sampled_points_idx = np.delete(sampled_points_idx, best_reward_index, axis=0)
            sampled_points_xyz = np.delete(sampled_points_xyz, best_reward_index, axis=0)
            print(f"Best angle: {best_angle}, Percent of coverage: {np.nansum(self.explored) / target_num}")
            num += 1
            TIME_END2 = time.time()
            total_time_ms = (TIME_END2 - TIME_START) * 1000
            print(f"Time taken for this iteration: {total_time_ms:.4f} milliseconds")

            try:
                with open(timing_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([num, sampled_points_idx.shape[0], reward_time_ms, total_time_ms])
            except Exception as e:
                print(f"Failed to write timing log: {e}")

        return candidate_points_idx, candidate_points_angles, candidate_points_xyz
                                 

from numba import njit, prange

@njit(parallel=True)
def calculate_rewards(sampled_points_idx, elev_g, trav, explored, sensor_range, sensor_fov, resolution, cost_barrier):
    num_points = sampled_points_idx.shape[0]
    base_angles = [0, 90, 180, 270]
    rewards = np.zeros(num_points, dtype=np.int32)
    best_angles = np.zeros(num_points, dtype=np.float32)
    explored_cells = np.zeros((num_points, *explored.shape), dtype=np.float32)

    for i in prange(num_points):
        point_index = sampled_points_idx[i]
        current_height = elev_g[point_index[0], point_index[1], point_index[2]]
        same_height_layers = np.where(np.abs(elev_g[:, point_index[1], point_index[2]] - current_height) < 0.2)[0]

        max_reward = -1
        best_angle = 0
        best_explored_cells = np.zeros_like(explored)

        for base_angle in base_angles:
            angles = np.deg2rad(np.arange(base_angle - sensor_fov / 2, base_angle + sensor_fov / 2, step=5))
            temp_explored_cells = explored.copy()
            temp_reward = 0

            for angle in angles:
                for r in np.arange(0, sensor_range, resolution):
                    dx = r * math.cos(angle)
                    dy = r * math.sin(angle)
                    i_x = int(round(point_index[1] + dx / resolution))
                    i_y = int(round(point_index[2] + dy / resolution))
                    if 0 <= i_x < elev_g.shape[1] and 0 <= i_y < elev_g.shape[2]:
                        stop = False
                        counted = False  # Track if this (i_x, i_y) has been counted for reward
                        for layer in same_height_layers:
                            if temp_explored_cells[layer, i_x, i_y] == 0:
                                temp_explored_cells[layer, i_x, i_y] = 1
                                if not counted:
                                    temp_reward += 1
                                    counted = True
                            if trav[layer, i_x, i_y] == cost_barrier:
                                stop = True
                                break
                        if stop:
                            break
                    else:
                        break

            if temp_reward > max_reward:
                max_reward = temp_reward
                best_angle = base_angle
                best_explored_cells[:] = temp_explored_cells

        rewards[i] = max_reward
        best_angles[i] = best_angle
        explored_cells[i] = best_explored_cells

    return rewards, best_angles, explored_cells
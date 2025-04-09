import os
import sys
import pickle
import numpy as np
import math
from scipy.stats import mode
from utils import *

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
        trav_gx = tomogram[1]
        trav_gy = tomogram[2]
        elev_g = tomogram[3]
        elev_g = np.nan_to_num(elev_g, nan=-100)
        self.elev_g = elev_g
        elev_c = tomogram[4]
        elev_c = np.nan_to_num(elev_c, nan=1e6)
        self.trav_raw = tomogram[5]         # Adding raw trav cost (no inflation) for reward calculation

        
        self.initPlanner(self.trav, trav_gx, trav_gy, elev_g, elev_c)
        # exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)
        # layers_t : travel cost
        # trav_grad_x : gradient x
        # trav_grad_y : gradient y
        # layers_g : ground height
        # layers_c : ceiling height

        # Initialize the explored graph
        self.explored = self.initExplorationGraph()
        self.layer_modes = self.compute_layer_modes()

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
    # def plan_TSP(self, sampled_points_idx):
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
                # Plan a path between the two points
                print("Planning path between points:", sampled_points_idx[i], sampled_points_idx[j])
                self.planner.plan(sampled_points_idx[i], sampled_points_idx[j], True)
                path_finder: a_star.Astar = self.planner.get_path_finder()
                path = path_finder.get_result_matrix()
    
                if len(path) > 0:  # If a valid path exists
                    path_length = len(path)  # Use the number of steps as the path length
                    adj_matrix[i, j] = path_length
                    adj_matrix[j, i] = path_length  # Symmetry for undirected graph
    
        return adj_matrix
        

    def plan(self, start_pos, end_pos):
        # TODO: calculate slice index. By default the start and end pos are all at slice 0
        # self.start_idx[1:] = self.pos2idx(start_pos)
        # self.end_idx[1:] = self.pos2idx(end_pos)
        # self.start_idx[:] = self.pos2idx_3D(start_pos)
        # self.end_idx[:] = self.pos2idx_3D(end_pos)
        

        self.start_idx = start_pos.astype(np.int32)
        self.end_idx = end_pos.astype(np.int32)
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
    
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32) # Swap x and y for grid indexing
        return idx
    
    def compute_layer_modes(self):
        """
        Precompute the mode of valid heights for each layer in the elevation map.
        Invalid heights (-100) are excluded from the calculation.
    
        Returns:
            np.ndarray: An array of modes for each layer.
        """
        layer_modes = []
        for s in range(self.elev_g.shape[0]):
            # Flatten the layer and exclude invalid heights (-100)
            valid_heights = self.elev_g[s][self.elev_g[s] != -100]
            if len(valid_heights) > 0:
                # Compute the mode of valid heights
                layer_mode = mode(valid_heights, nan_policy='omit').mode[0]
            else:
                # If no valid heights, set mode to NaN
                layer_mode = np.nan
            layer_modes.append(layer_mode)
        return np.array(layer_modes, dtype=np.float32)
    
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
        idx_xy = np.round(pos_xy[:2] / self.resolution).astype(np.int32) + self.offset
        idx_xy = np.array([idx_xy[1], idx_xy[0]], dtype=np.float32)  # Swap x and y for grid indexing
    
        # Search for the z index (layer number) using the precomputed layer modes
        z_height = pos[2]  # Extract the z-coordinate
        z_idx = -1  # Default to -1 if no valid layer is found
        if z_height > self.layer_modes[-1]:
            z_idx = self.layer_modes.shape[0] -1    
        for s, layer_mode in enumerate(self.layer_modes):
            if not np.isnan(layer_mode) and layer_mode >= z_height:
                z_idx = s-1
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
        step_x = max(1, int(self.sensor_range / self.resolution))  # Step size in the x dimension
        step_y = max(1, int(self.sensor_range / self.resolution))  # Step size in the y dimension
        slice_indices = np.arange(0, self.elev_g.shape[0], 1)
        x_indices = np.arange(0, self.elev_g.shape[1], step_x)
        y_indices = np.arange(0, self.elev_g.shape[2], step_y)
        sampled_indices = np.array(np.meshgrid(slice_indices, x_indices, y_indices, indexing="ij"))
        sampled_indices = sampled_indices.reshape(3, -1).T  # Reshape to (N, 3)
    
        # Filter out invalid or untraversable points
        valid_indices = []
        for s, x, y in sampled_indices:
            if self.trav[s, x, y] < 30 and self.elev_g[s, x, y] >= 0:
                valid_indices.append([s, x, y])
    
        valid_indices = np.array(valid_indices)
    
    # Filter out points with the same x, y indices and the same exact height in the elevation map
        unique_points = []
        seen_xy = {}
        for s, x, y in valid_indices:
            xy_key = (x, y)
            height = self.elev_g[s, x, y]
            if xy_key not in seen_xy or seen_xy[xy_key] != height:
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
        # Get the indices of all traversable points (travel cost < max_cost)
        traversable_mask = (self.trav < self.cost_barrier) & (self.elev_g >= 0)  # cost less than barrier and is a valid grid
        traversable_indices = np.argwhere(traversable_mask)
    
        # If there are fewer traversable points than requested samples, return all
        if len(traversable_indices) <= num_samples:
            sampled_xyz = np.empty((len(traversable_indices), 3), dtype=np.float32)
            for idx, (s, x, y) in enumerate(traversable_indices):
                map_x = (x - self.offset[0]) * self.resolution + self.center[0]
                map_y = (y - self.offset[1]) * self.resolution + self.center[1]
                map_z = self.elev_g[s, x, y]
                sampled_xyz[idx] = [map_x, map_y, map_z]
            return traversable_indices, sampled_xyz
    
        # Uniformly sample from the traversable points
        sampled_idx = traversable_indices[
            np.random.choice(len(traversable_indices), num_samples, replace=False)
        ]
    
        # Preallocate the sampled_xyz array
        sampled_xyz = np.empty((num_samples, 3), dtype=np.float32)
    
        # Fill the sampled_xyz array
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
        map_y = (idx[1] - self.offset[1]) * self.resolution + self.center[1]
        map_x = (idx[2] - self.offset[0]) * self.resolution + self.center[0]
        map_z = self.elev_g[idx[0], idx[2], idx[1]]
        return np.array([map_x, map_y, map_z], dtype=np.float32)
    
    def nextBestView(self):
        """
        Calculate the reward for each sampled point based on the number of unseen cells in its neighborhood.
    
        Returns:
            np.ndarray: Array of rewards for each sampled point.
        """
        min_reward = 10
        finished = False
        sampled_points_idx, sampled_points_xyz= self.sampleUniformPointsInSpace()
        best_point = None
        best_angle = None
        best_explored_cells = self.explored.copy()
        candidate_points_idx = np.full(sampled_points_idx.shape, np.nan, dtype=np.float32)
        candidate_points_angle = np.full(sampled_points_idx.shape[0], np.nan, dtype=np.float32)
        candidate_points_xyz = np.full(sampled_points_xyz.shape, np.nan, dtype=np.float32)
        target_num = np.count_nonzero(~np.isnan(self.explored))
        for j in range(candidate_points_idx.shape[0]):
            if finished == True:
                    break
            print("explored cells:", np.nansum(self.explored))
            if np.nansum(self.explored) < self.cfg.planner.coverage_threshold * target_num: 
                best_reward = -1               
                ## Loop to find the next best point
                # print("percent of coverage:", np.sum(self.explored) / target_num)
                for i, point_index in enumerate(sampled_points_idx):
                    angle, reward, explored_cells = self.BestAnglewithReward(point_index)
                    if reward > best_reward:
                        best_reward = reward
                        best_point = point_index
                        best_angle = angle
                        best_explored_cells = explored_cells
                # Update the explored graph with the best angle
                self.explored = best_explored_cells
                candidate_points_idx[j] = best_point
                candidate_points_angle[j] = best_angle
                matching_indices = np.where((sampled_points_idx == best_point).all(axis=1))[0]
                if len(matching_indices) > 0:
                    candidate_points_xyz[j] = sampled_points_xyz[matching_indices[0]]
                    # print("Best reward:", best_reward)
                if best_reward < min_reward:
                    finished = True
                    break
                # Remove the best point from the sampled points
                                # Find the matching row index for best_point
                matching_indices = np.where((sampled_points_idx == best_point).all(axis=1))[0]
                
                # Check if a match is found before attempting to delete
                if len(matching_indices) > 0:
                    sampled_points_idx = np.delete(sampled_points_idx, matching_indices[0], axis=0)
                    sampled_points_xyz = np.delete(sampled_points_xyz, matching_indices[0], axis=0)
                else:
                    print("Warning: Best point not found in sampled_points_idx. Skipping deletion.")
            else: 
                break
            print("percent of coverage:", np.nansum(self.explored) / target_num)
            
    
        # Remove NaN values from candidate points
        assert candidate_points_idx.shape[0] == candidate_points_angle.shape[0] == candidate_points_xyz.shape[0], \
            "Mismatch in the number of rows between candidate arrays."
        
        # Remove rows where any column contains NaN
        valid_mask = ~np.isnan(candidate_points_idx).any(axis=1)
        if np.any(valid_mask):  # Only apply the mask if there are valid rows
            candidate_points_idx = candidate_points_idx[valid_mask]
            candidate_points_angle = candidate_points_angle[valid_mask]
            candidate_points_xyz = candidate_points_xyz[valid_mask]
        else:
            print("All candidate points contain NaN values.")
        return candidate_points_idx, candidate_points_angle, candidate_points_xyz
    
    def getExploredGraph(self):
        """
        Get the explored graph.

        Returns:
            np.ndarray: The explored graph.
        """
        return self.explored




    def BestAnglewithReward(self, point_index): 
        """
        Calculate the best angle for a given point index based on the number of unseen cells in its neighborhood.
        Args:
            point_index (tuple): The index of the point in the grid (slice, x, y).
        Returns:
            best_angle (float): The best angle in degrees
            reward (int): The reward for the best angle
            Explored_cells (np.ndarray): The explored cells for the best angle
        """
        base_angles = [0, 90, 180, 270]
        rewards = np.zeros(len(base_angles), dtype=np.int32)
        Explored_cells = np.zeros((len(base_angles), *self.explored.shape), dtype=np.float32)
    
        # Get the height of the current point
        current_height = self.elev_g[point_index[0], point_index[1], point_index[2]]
    
        # Find all layers with the same height at the same x, y position
        same_height_layers = np.where(self.elev_g[:, point_index[1], point_index[2]] == current_height)[0]
    
        for i, base_angle in enumerate(base_angles):
            # Calculate angles with 2-degree steps
            angles = np.deg2rad(np.arange(base_angle - self.sensor_fov / 2, base_angle + self.sensor_fov / 2, step=10))
            Explored_cells[i] = self.explored.copy()
            for angle in angles:
                # Calculate the coordinates of the sensor range
                x_min = point_index[1]
                x_max = point_index[1] + math.floor(self.sensor_range * np.cos(angle) / self.resolution)
                y_min = point_index[2]
                y_max = point_index[2] + math.floor(self.sensor_range * np.sin(angle) / self.resolution)
                # Determine the step direction for x and y
                x_step = 1 if x_max >= x_min else -1
                y_step = 1 if y_max >= y_min else -1
    
                for i_x in range(x_min, x_max + x_step, x_step): 
                    stop = False
                    for i_y in range(y_min, y_max + y_step, y_step): 
                        if 0 <= i_x < self.map_dim[0] and 0 <= i_y < self.map_dim[1]:
                            for layer in same_height_layers:  # Iterate over layers with the same height
                                if Explored_cells[i, layer, i_x, i_y] == 0:
                                    rewards[i] += 1
                                    Explored_cells[i, layer, i_x, i_y] = 1
                                if self.trav[layer, i_x, i_y] == self.cost_barrier:  # Stop if a barrier is hit
                                    stop = True
                                    break
                        if stop:
                            break
    
        # Determine the best angle
        best_angle_index = np.argmax(rewards)
        best_angle = base_angles[best_angle_index]
        
        return best_angle, rewards[best_angle_index], Explored_cells[best_angle_index]


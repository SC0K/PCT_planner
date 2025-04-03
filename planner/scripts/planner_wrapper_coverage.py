import os
import sys
import pickle
import numpy as np
import math
import cupy as cp

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
        self.sensor_range = None
        self.elev_g = None
        self.trav = None
        self.explored = None
        self.sensor_range = self.cfg.sensor.sensor_range
        self.sensor_fov = self.cfg.sensor.sensor_fov

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

    def initExplorationGraph(self):
        """
        Initialize a graph to track whether cells in the elevation grid (elev_g) are explored.

        Returns:
            np.ndarray: A float array where NaN indicates ignored cells, 0.0 indicates unexplored cells, 
                        and 1.0 indicates explored cells.
        """
        # Initialize the exploration graph with NaN values
        exploration_graph = np.full_like(self.elev_g, np.nan, dtype=np.float32)

        # Set non-NaN cells to 0.0 (unexplored)
        nan_mask = np.isnan(self.elev_g)
        exploration_graph[~nan_mask] = 0.0

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

    def plan(self, start_pos, end_pos):
        # TODO: calculate slice index. By default the start and end pos are all at slice 0
        self.start_idx[1:] = self.pos2idx(start_pos)
        self.end_idx[1:] = self.pos2idx(end_pos)

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
        idx = np.array([idx[1], idx[0]], dtype=np.float32)
        return idx
    
    def sampleTraversablePoints(self, num_samples):
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
            np.ndarray: The map coordinates (x, y).
        """
        # Convert grid indices to map coordinates
        map_x = (idx[1] - self.offset[0]) * self.resolution + self.center[0]
        map_y = (idx[2] - self.offset[1]) * self.resolution + self.center[1]
        map_z = self.elev_g[idx[0], idx[1], idx[2]]
        return np.array([map_x, map_y, map_z], dtype=np.float32)
    
    def nextBestView(self):
        """
        Iteratively select the best points based on rewards, update the explored region, 
        and stop when either all sampled points are selected or 90% of the explorable points are explored.
        
        Returns:
            list: List of selected points (indices).
            list: List of corresponding angles for each selected point.
            list: List of map coordinates for each selected point.
        """
        min_reward = 10
        selected_points = []
        selected_angles = []
        selected_points_xyz = []
        
        # Convert self.explored to a CuPy array
        explored_cp = cp.array(self.explored)
        
        # Calculate the total number of explorable points
        total_explorable_points = cp.sum(~cp.isnan(explored_cp))
        explored_points = cp.sum(explored_cp == 1)
        
        # Sample traversable points once
        sampled_points_idx, sampled_points_xyz = self.sampleTraversablePoints(num_samples=self.cfg.planner.sample_num)
        
        while explored_points / total_explorable_points < 0.9 and len(sampled_points_idx) > 0:
            # Calculate rewards for all sampled points
            angles, rewards, explored_cells = self.BestAnglewithReward(sampled_points_idx)
            
            # Ensure rewards is a CuPy array
            rewards_cp = cp.array(rewards)
            
            # Find the best point and angle
            best_reward_idx = cp.argmax(rewards_cp).get()  # Convert to NumPy scalar
            best_reward = rewards_cp[best_reward_idx]
            
            # Stop if no point meets the minimum reward threshold
            if best_reward < min_reward:
                break
            
            best_point = sampled_points_idx[best_reward_idx]
            best_angle = angles[best_reward_idx]
            best_point_xyz = sampled_points_xyz[best_reward_idx]
            
            # Update the explored graph with the best explored cells
            self.explored = explored_cells[best_reward_idx]
            
            # Add the selected point, angle, and coordinates to the results
            selected_points.append(best_point)
            selected_angles.append(best_angle)
            selected_points_xyz.append(best_point_xyz)
            
            # Update the count of explored points
            explored_cp = cp.array(self.explored)  # Reconvert to CuPy after updating
            explored_points = cp.sum(explored_cp == 1)
            
            # Remove the selected point from the sampled points
            sampled_points_idx = np.delete(sampled_points_idx, best_reward_idx, axis=0)
            sampled_points_xyz = np.delete(sampled_points_xyz, best_reward_idx, axis=0)
            
            # Optionally, remove the selected point from the graph
            self.trav[best_point[0], best_point[1], best_point[2]] = self.cost_barrier
        
        return selected_points, selected_angles, selected_points_xyz



    def BestAnglewithReward(self, points_idx):
        """
        Calculate the best angle for all given points based on the number of unseen cells in their neighborhoods.
        Args:
            points_idx (np.ndarray): Array of point indices in the grid (N, 3), where N is the number of points.
        Returns:
            np.ndarray: Best angles for all points (N,).
            np.ndarray: Rewards for the best angles (N,).
            np.ndarray: Updated explored cells for all points (N, ...).
        """
        # Ensure points_idx is 2D
        if points_idx.ndim == 1:
            points_idx = points_idx.reshape(1, -1)  # Reshape to (1, 3) for a single point
    
        # Convert points_idx to CuPy array
        points_idx = cp.array(points_idx)
    
        # Convert self.trav to CuPy array
        trav_gpu = cp.array(self.trav)
    
        base_angles = cp.array([0, 90, 180, 270])
        num_points = points_idx.shape[0]
        num_angles = len(base_angles)
    
        # Initialize rewards and explored cells
        rewards = cp.zeros((num_points, num_angles), dtype=cp.int32)
        Explored_cells = cp.tile(cp.array(self.explored, dtype=cp.float32)[cp.newaxis, :, :, :], (num_points, 1, 1, 1))
    
        # Precompute angles for all base angles
        angles = cp.deg2rad(cp.arange(-self.sensor_fov / 2, self.sensor_fov / 2, step=2))
        all_angles = angles[cp.newaxis, :] + cp.deg2rad(base_angles[:, cp.newaxis])
        all_angles = all_angles.flatten()
    
        # Precompute sensor ranges
        cos_angles = cp.cos(all_angles)
        sin_angles = cp.sin(all_angles)
    
        # Process each point and angle
        for i, angle in enumerate(base_angles):
            for s in range(points_idx.shape[0]):
                # Extract the slice index
                slice_idx = points_idx[s, 0]
    
                # Compute the sensor range for the current point and angle
                x_min = points_idx[s, 1]
                y_min = points_idx[s, 2]
                x_max = x_min + int(self.sensor_range * cos_angles[i] / self.resolution)
                y_max = y_min + int(self.sensor_range * sin_angles[i] / self.resolution)
    
                # Clip indices to stay within bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(self.trav.shape[1] - 1, x_max)
                y_max = min(self.trav.shape[2] - 1, y_max)
    
                # Create a boolean mask for valid cells
                valid_mask = (trav_gpu[slice_idx, x_min:x_max + 1, y_min:y_max + 1] != self.cost_barrier)
    
                # Count unexplored cells and update rewards
                unexplored_mask = (Explored_cells[s, slice_idx, x_min:x_max + 1, y_min:y_max + 1] == 0) & valid_mask
                rewards[s, i] += cp.sum(unexplored_mask)
    
                # Mark cells as explored
                Explored_cells[s, slice_idx, x_min:x_max + 1, y_min:y_max + 1][unexplored_mask] = 1
    
        # Determine the best angle for each point
        best_angle_indices = cp.argmax(rewards, axis=1)
        best_angles = base_angles[best_angle_indices]
        best_rewards = rewards[cp.arange(num_points), best_angle_indices]
    
        return best_angles.get(), best_rewards.get(), Explored_cells
    

    # def BestAnglewithReward(self, point_index): 
    #     """
    #     Calculate the best angle for a given point index based on the number of unseen cells in its neighborhood.
    #     Args:
    #         point_index (tuple): The index of the point in the grid (slice, x, y).
    #     Returns:
    #         best_angle (float): The best angle in degrees
    #         reward (int): The reward for the best angle
    #         Explored_cells (np.ndarray): The explored cells for the best angle
    #     """
    #     base_angles = [0, 90, 180, 270]
    #     rewards = np.zeros(len(base_angles), dtype=np.int32)
    #     Explored_cells = np.zeros((len(base_angles), *self.explored.shape), dtype=np.float32)
    #     for i, base_angle in enumerate(base_angles):
    #         # Calculate angles with 2-degree steps
    #         angles = np.deg2rad(np.arange(base_angle - self.sensor_fov / 2, base_angle + self.sensor_fov / 2, step=2))
    #         Explored_cells[i] = self.explored.copy()
    #         for angle in angles:
    #             # Calculate the coordinates of the sensor range
    #             x_min = point_index[1]
    #             x_max = point_index[1] + math.floor(self.sensor_range * np.cos(angle) / self.resolution)
    #             y_min = point_index[2]
    #             y_max = point_index[2] + math.floor(self.sensor_range * np.sin(angle) / self.resolution)
    #             # Determine the step direction for x and y
    #             x_step = 1 if x_max >= x_min else -1
    #             y_step = 1 if y_max >= y_min else -1

    #             # for i_x in range(x_min, x_max + x_step, x_step): 
    #             #     stop = False
    #             #     for i_y in range(y_min, y_max + y_step, y_step): 
    #             #         if 0 <= i_x < self.map_dim[0] and 0 <= i_y < self.map_dim[1]:
    #             #             if Explored_cells[i, point_index[0], i_x, i_y] == 0:
    #             #                 rewards[i] += 1
    #             #                 Explored_cells[i, point_index[0], i_x, i_y] = 1
    #             #             if self.trav[point_index[0], i_x, i_y] == self.cost_barrier:    # Stop if a barrier is hit
    #             #                 # rewards[i] += 1     # TODO Optional: reward for hitting a barrier is increased
    #             #                 stop = True
    #             #                 break  
    #             #     if stop:
    #             #         break

    #             x_indices = np.arange(x_min, x_max + x_step, x_step)
    #             y_indices = np.arange(y_min, y_max + y_step, y_step)

    #             # Apply bounds check independently for x_indices and y_indices
    #             x_valid_mask = (0 <= x_indices) & (x_indices < self.map_dim[0])
    #             y_valid_mask = (0 <= y_indices) & (y_indices < self.map_dim[1])

    #             # Filter valid indices
    #             x_indices = x_indices[x_valid_mask]
    #             y_indices = y_indices[y_valid_mask]

    #             # Create a meshgrid of x_indices and y_indices
    #             x_mesh, y_mesh = np.meshgrid(x_indices, y_indices, indexing='ij')

    #             # Iterate through the meshgrid and stop if a barrier is encountered
    #             for i_x, i_y in zip(x_mesh.flatten(), y_mesh.flatten()):
    #                 if self.trav[point_index[0], i_x, i_y] == self.cost_barrier:  # Stop if a barrier is hit
    #                     break  # Exit the loop when a barrier is encountered
    #                 if Explored_cells[i, point_index[0], i_x, i_y] == 0:
    #                     rewards[i] += 1
    #                     Explored_cells[i, point_index[0], i_x, i_y] = 1

               
    #     # Determine the best angle
    #     best_angle_index = np.argmax(rewards)
    #     best_angle = base_angles[best_angle_index]
        
    #     return best_angle, rewards[best_angle_index], Explored_cells[best_angle_index]


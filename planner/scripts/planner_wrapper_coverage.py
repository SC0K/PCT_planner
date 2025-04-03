import os
import sys
import pickle
import numpy as np
import math

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
        Calculate the reward for each sampled point based on the number of unseen cells in its neighborhood.
    
        Returns:
            np.ndarray: Array of rewards for each sampled point.
        """
        min_reward = 10
        finished = False
        sampled_points_idx, sampled_points_xyz= self.sampleTraversablePoints(num_samples=self.cfg.planner.sample_num)
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
            if np.sum(self.explored) < self.cfg.planner.coverage_threshold * target_num: 
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
                    print("Best reward:", best_reward)
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
    
        # Remove NaN values from candidate points
        valid_mask = ~np.isnan(candidate_points_idx).any(axis=1)
        candidate_points_idx = candidate_points_idx[valid_mask]
        candidate_points_angle = candidate_points_angle[valid_mask]
        candidate_points_xyz = candidate_points_xyz[valid_mask]
        return candidate_points_idx, candidate_points_angle, candidate_points_xyz





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
        for i, base_angle in enumerate(base_angles):
            # Calculate angles with 2-degree steps
            angles = np.deg2rad(np.arange(base_angle - self.sensor_fov / 2, base_angle + self.sensor_fov / 2, step=2))
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
                            if Explored_cells[i, point_index[0], i_x, i_y] == 0:
                                rewards[i] += 1
                                Explored_cells[i, point_index[0], i_x, i_y] = 1
                            if self.trav[point_index[0], i_x, i_y] == self.cost_barrier:    # Stop if a barrier is hit
                                # rewards[i] += 1     # TODO Optional: reward for hitting a barrier is increased
                                stop = True
                                break  
                    if stop:
                        break

        # Determine the best angle
        best_angle_index = np.argmax(rewards)
        best_angle = base_angles[best_angle_index]
        
        return best_angle, rewards[best_angle_index], Explored_cells[best_angle_index]
    
    # def nextBestView(self, trav, sensor_range, coverage_threshold=0.95, num_samples=100):
    #     """
    #     Perform Next Best View (NBV) to achieve coverage path planning.
    
    #     Args:
    #         trav (np.ndarray): The travel cost map.
    #         sensor_range (int): The radius of the sensor range.
    #         coverage_threshold (float): The percentage of grid cells to cover (default: 90%).
    #         num_samples (int): The number of candidate points to sample.
    
    #     Returns:
    #         list: List of selected points for coverage.
    #     """
    #     # Initialize the seen cells grid
    #     seen_cells = np.zeros_like(trav, dtype=bool)
    #     total_cells = np.prod(trav.shape)
    #     target_coverage = coverage_threshold * total_cells
    
    #     selected_points = []
    
    #     while np.sum(seen_cells) < target_coverage:
    #         # Sample candidate points
    #         sampled_points = self.sampleTraversablePoints(trav, num_samples)
    
    #         # Calculate rewards for each sampled point
    #         rewards = self.calculateRewards(sampled_points, seen_cells, trav, sensor_range)
    
    #         # Choose the point with the highest reward
    #         best_idx = np.argmax(rewards)
    #         best_point = sampled_points[best_idx]
    #         selected_points.append(best_point)
    
    #         # Update seen cells
    #         x, y = best_point
    #         x_min = max(0, x - sensor_range)
    #         x_max = min(trav.shape[0], x + sensor_range + 1)
    #         y_min = max(0, y - sensor_range)
    #         y_max = min(trav.shape[1], y + sensor_range + 1)
    #         seen_cells[x_min:x_max, y_min:y_max] = True  # Mark cells as seen
    
    #         # Log progress
    #         rospy.loginfo("Selected point: %s, Coverage: %.2f%%", best_point, (np.sum(seen_cells) / total_cells) * 100)
    
    #     return selected_points
    

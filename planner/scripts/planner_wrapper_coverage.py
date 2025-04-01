import os
import sys
import pickle
import numpy as np

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
            self.sensor_range = int(round(self.cfg.planner.sensor_range / self.resolution))


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
            np.ndarray: Array of sampled traversable points (x, y indices).
        """
        # Get the indices of all traversable points (travel cost < max_cost)
        traversable_mask = (self.trav < self.cost_barrier) & (self.elev_g >= 0)  # cost less than barrier and is a valid grid

        traversable_indices = np.argwhere(traversable_mask)

        # If there are fewer traversable points than requested samples, return all
        if len(traversable_indices) <= num_samples:
            return traversable_indices

        # Uniformly sample from the traversable points
        sampled_idx = traversable_indices[
            np.random.choice(len(traversable_indices), num_samples, replace=False)
        ]

        sampled_xyz = []
        # Convert to (x, y, z) format
        for s,x,y in sampled_idx:
            # Convert grid indices to map coordinates
            map_x = (x - self.offset[0]) * self.resolution + self.center[0]
            map_y = (y - self.offset[1]) * self.resolution + self.center[1]
            map_z = self.elev_g[s,x,y]
            sampled_xyz.append([map_x, map_y, map_z])
        sampled_xyz = np.array(sampled_xyz, dtype=np.float32)


        return sampled_idx , sampled_xyz
    
    def calculateRewards(self, sampled_points, seen_cells, trav, sensor_range):
        """
        Calculate the reward for each sampled point based on the number of unseen cells in its neighborhood.
    
        Args:
            sampled_points (np.ndarray): Array of sampled traversable points (x, y indices).
            seen_cells (np.ndarray): Boolean grid indicating which cells have been seen.
            trav (np.ndarray): The travel cost map.
            sensor_range (int): The radius of the sensor range.
    
        Returns:
            np.ndarray: Array of rewards for each sampled point.
        """
        rewards = np.zeros(len(sampled_points), dtype=np.int32)
    
        for i, (x, y) in enumerate(sampled_points):
            # Define the neighborhood bounds
            x_min = max(0, x - sensor_range)
            x_max = min(trav.shape[0], x + sensor_range + 1)
            y_min = max(0, y - sensor_range)
            y_max = min(trav.shape[1], y + sensor_range + 1)
    
            # Count the number of unseen cells in the neighborhood
            neighborhood = seen_cells[x_min:x_max, y_min:y_max]
            rewards[i] = np.sum(~neighborhood)  # Count unseen cells
    
        return rewards
    
    def nextBestView(self, trav, sensor_range, coverage_threshold=0.9, num_samples=100):
        """
        Perform Next Best View (NBV) to achieve coverage path planning.
    
        Args:
            trav (np.ndarray): The travel cost map.
            sensor_range (int): The radius of the sensor range.
            coverage_threshold (float): The percentage of grid cells to cover (default: 90%).
            num_samples (int): The number of candidate points to sample.
    
        Returns:
            list: List of selected points for coverage.
        """
        # Initialize the seen cells grid
        seen_cells = np.zeros_like(trav, dtype=bool)
        total_cells = np.prod(trav.shape)
        target_coverage = coverage_threshold * total_cells
    
        selected_points = []
    
        while np.sum(seen_cells) < target_coverage:
            # Sample candidate points
            sampled_points = self.sampleTraversablePoints(trav, num_samples)
    
            # Calculate rewards for each sampled point
            rewards = self.calculateRewards(sampled_points, seen_cells, trav, sensor_range)
    
            # Choose the point with the highest reward
            best_idx = np.argmax(rewards)
            best_point = sampled_points[best_idx]
            selected_points.append(best_point)
    
            # Update seen cells
            x, y = best_point
            x_min = max(0, x - sensor_range)
            x_max = min(trav.shape[0], x + sensor_range + 1)
            y_min = max(0, y - sensor_range)
            y_max = min(trav.shape[1], y + sensor_range + 1)
            seen_cells[x_min:x_max, y_min:y_max] = True  # Mark cells as seen
    
            # Log progress
            rospy.loginfo("Selected point: %s, Coverage: %.2f%%", best_point, (np.sum(seen_cells) / total_cells) * 100)
    
        return selected_points
    

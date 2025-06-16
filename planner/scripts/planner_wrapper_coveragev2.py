import os
import sys
import pickle
import numpy as np
import math
from scipy.stats import mode
from utils import *
import open3d as o3d
import cupy as cp
import sklearn.cluster
import time
import cupyx.scipy.ndimage as cp_ndimage
import rospy
import reward_cpp


sys.path.append('../')
from lib import a_star, ele_planner, traj_opt

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'
ray_prep_kernel_code = r'''
extern "C" __global__
void ray_prep_kernel(
    const double* poses,         // (n, 3)
    const double* yaws,          // (n,)
    int n_poses,
    int n_rays,
    int n_steps,
    double fov_deg,
    double el_min_deg,
    double el_max_deg,
    double max_range,
    double resolution,
    double voxel_size,
    const int* min_idx,          // (3,)
    const int* grid_shape,       // (3,)
    int* idxs_out,               // (n, n_rays*n_rays, n_steps, 3)
    bool* valid_out              // (n, n_rays*n_rays, n_steps)
) {
    int pose_idx = blockIdx.x;
    int ray_idx = threadIdx.x;
    if (pose_idx >= n_poses || ray_idx >= n_rays * n_rays) return;

    int az_i = ray_idx / n_rays;
    int el_i = ray_idx % n_rays;
    double az = (-fov_deg/2.0) + az_i * (fov_deg/(n_rays-1));
    double el = el_min_deg + el_i * ((el_max_deg-el_min_deg)/(n_rays-1));
    double az_rad = az * 0.017453292519943295;
    double el_rad = el * 0.017453292519943295;

    // Direction in camera frame
    double dx = cos(el_rad) * cos(az_rad);
    double dy = cos(el_rad) * sin(az_rad);
    double dz = sin(el_rad);

    // Apply yaw rotation (Z axis)
    double yaw = yaws[pose_idx] * 0.017453292519943295;
    double c = cos(yaw), s = sin(yaw);
    double dir_x = c*dx - s*dy;
    double dir_y = s*dx + c*dy;
    double dir_z = dz;

    // Camera position (shifted)
    const double* cam = poses + pose_idx*3;

    for (int s = 0; s < n_steps; ++s) {
        double dist = s * resolution;
        double px = cam[0] + dir_x * dist;
        double py = cam[1] + dir_y * dist;
        double pz = cam[2] + dir_z * dist;

        int ix = int(floor(px / voxel_size)) - min_idx[0];
        int iy = int(floor(py / voxel_size)) - min_idx[1];
        int iz = int(floor(pz / voxel_size)) - min_idx[2];

        int out_idx = (((pose_idx * n_rays * n_rays + ray_idx) * n_steps) + s) * 3;
        idxs_out[out_idx + 0] = ix;
        idxs_out[out_idx + 1] = iy;
        idxs_out[out_idx + 2] = iz;

        bool valid = (ix >= 0 && ix < grid_shape[0] &&
                      iy >= 0 && iy < grid_shape[1] &&
                      iz >= 0 && iz < grid_shape[2]);
        valid_out[(pose_idx * n_rays * n_rays + ray_idx) * n_steps + s] = valid;
    }
}
'''
ray_prep_kernel = cp.RawKernel(ray_prep_kernel_code, "ray_prep_kernel")
ray_hit_kernel_code = r'''
extern "C" __global__
void ray_first_hit(const int* idxs, const bool* valid, const bool* hash,
                   int n_rays, int n_steps,
                   int gx, int gy, int gz,
                   int* visible_hits, int* hit_flags) {
    int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ray_id >= n_rays) return;

    for (int s = 0; s < n_steps; ++s) {
        int base_idx = (ray_id * n_steps + s) * 3;
        int valid_idx = ray_id * n_steps + s;

        if (!valid[valid_idx]) continue;

        int i = idxs[base_idx + 0];
        int j = idxs[base_idx + 1];
        int k = idxs[base_idx + 2];

        if (i < 0 || i >= gx || j < 0 || j >= gy || k < 0 || k >= gz) continue;

        int flat_idx = i * (gy * gz) + j * gz + k;

        if (hash[flat_idx]) {
            int out_idx = ray_id * 3;
            visible_hits[out_idx + 0] = i;
            visible_hits[out_idx + 1] = j;
            visible_hits[out_idx + 2] = k;
            hit_flags[ray_id] = 1;
            return;
        }
    }
}
'''
reward_kernel_code = r'''
extern "C" __global__
void reward_count(const int* visible_hits, const int* hit_flags, const bool* explored,
                  const int* min_idx, int n_candidates, int n_hits_per_candidate, int* rewards) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_candidates) return;

    int reward = 0;
    for (int j = 0; j < n_hits_per_candidate; ++j) {
        int idx = i * n_hits_per_candidate + j;
        if (hit_flags[idx]) {
            int x = visible_hits[idx * 3 + 0] - min_idx[0];
            int y = visible_hits[idx * 3 + 1] - min_idx[1];
            int z = visible_hits[idx * 3 + 2] - min_idx[2];
            // Bounds check
            if (x >= 0 && y >= 0 && z >= 0) {
                reward += !explored[x * (gridDim.y * gridDim.z) + y * gridDim.z + z];
            }
        }
    }
    rewards[i] = reward;
}
'''
reward_count_kernel = cp.RawKernel(reward_kernel_code, "reward_count")
ray_first_hit_kernel = cp.RawKernel(ray_hit_kernel_code, "ray_first_hit")


class TomogramCoveragePlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # self.use_quintic = self.cfg.planner.use_quintic
        self.use_quintic = False
        self.max_heading_rate = self.cfg.planner.max_heading_rate

        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir

        self.resolution = None
        self.resolution_raycast = None
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
        self.sensor_range_analysis = 4.5
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
        self.resolution_raycast = voxel_size

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
        """ Initialise the planner without initialising the optimiser."""
        self.planner.reinit_map(
            20,
            15,
            self.resolution,
            self.n_slice,
            0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            self.gateway.reshape(-1, self.gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )
    
    def add_obstacle_points(self, world_points, z_buffer=0.5, xy_buffer=0.2):
        """
        Add new obstacle points to the tomograph by marking all provided points as untraversable.
    
        Args:
            world_points (np.ndarray): Nx3 array of new obstacle points in world coordinates.
            z_buffer (float): Height buffer above the lowest point to mark as obstacle (meters).
            xy_buffer (float): XY buffer around each obstacle (meters).
        """
        if world_points.shape[0] == 0:
            return
    
        for pt in world_points:
            # Convert world point to tomograph grid indices
            idx = self.pos2idx_3D(pt)
            idx = np.round(idx).astype(int)
            s, x, y = idx
    
            # Mark a region around (x, y) in all layers at or below this z as untraversable
            xy_radius = int(np.ceil(xy_buffer / self.resolution))
            for ds in range(self.elev_g.shape[0]):
                # Only mark if the elevation is below or close to the obstacle point
                elev = self.elev_g[ds, y, x]
                if abs(elev - pt[2]) <= z_buffer:
                    x_min = max(0, x - xy_radius)
                    x_max = min(self.elev_g.shape[2], x + xy_radius + 1)
                    y_min = max(0, y - xy_radius)
                    y_max = min(self.elev_g.shape[1], y + xy_radius + 1)
                    self.trav[ds, y_min:y_max, x_min:x_max] = self.cost_barrier  # Mark as untraversable
    
        self.init_planner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)

        
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
                self.planner.plan(start_idx, end_idx, False)
                path_finder: a_star.Astar = self.planner.get_path_finder()
                path = path_finder.get_result_matrix()
    
                if len(path) > 0:  # If a valid path exists
                    path_length = len(path)  # Use the number of steps as the path length
                    adj_matrix[i, j] = path_length
                    adj_matrix[j, i] = path_length  # Symmetry for undirected graph
    
        return adj_matrix
        
    def plan_with_idx(self, start_pos, end_pos):

        self.start_idx = np.array([start_pos[0], start_pos[2], start_pos[1]], dtype=np.int32)   # planner needs s,y,x whereas the grid index is s,x,y
        self.end_idx = np.array([end_pos[0], end_pos[2], end_pos[1]], dtype=np.int32)
        print("start_idx:", self.start_idx)
        print("end_idx:", self.end_idx)

        self.planner.plan(self.start_idx, self.end_idx, False)
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
    def plan_with_idx_online(self, start_pos, end_pos):
        start_pos = np.array([start_pos[0], start_pos[2], start_pos[1]], dtype=np.int32)   # planner needs s,y,x whereas the grid index is s,x,y
        end_pos = np.array([end_pos[0], end_pos[2], end_pos[1]], dtype=np.int32)
        self.planner.plan(start_pos, end_pos, False)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()   
        if len(path) == 0:
            rospy.logerr("No path found between start and end positions.")
            return None
        traj_3d = np.array([self.idx2pos_3D(idx) for idx in path]) + np.array([0,0, 0.5])  # Add a small offset to z for visualization
        return traj_3d
    
    def plan(self, start_pos, end_pos):
        self.start_idx[:] = self.pos2idx_3D_plan(start_pos)
        self.end_idx[:] = self.pos2idx_3D_plan(end_pos)
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
    def plan_online(self, start_pos, end_pos):
        self.start_idx[:] = self.pos2idx_3D_plan(start_pos)
        self.end_idx[:] = self.pos2idx_3D_plan(end_pos)
        print("start_idx:", self.start_idx)
        print("end_idx:", self.end_idx)

        self.planner.plan(self.start_idx, self.end_idx, True)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()

        print("path:", path)    
        if len(path) == 0:
            rospy.logerr("No path found between start and end positions.")
            return None
        traj_3d = np.array([self.idx2pos_3D(idx) for idx in path]) + np.array([0,0, 0.5])  # Add a small offset to z for visualization
        return traj_3d
    
    def online_local_replan(self, start_pos, target_pos, height_tol=1.0):
        """
        Given a target position in world coordinates, find the closest traversable grid cell
        along the line from start_pos to target_pos, and plan a path from start_pos to it.
        """
        # Convert to grid idx (s, x, y)
        start_idx = np.round(self.pos2idx_3D_plan(start_pos)).astype(int)
        goal_idx = np.round(self.pos2idx_3D_plan(target_pos)).astype(int)
        if abs(goal_idx[1] - start_idx[1])+abs(goal_idx[2]-start_idx[2]) < 3:
            rospy.logwarn("Target position is too close to the start position, skipping replan.")
            return np.array([start_pos,target_pos])
        grid_shape = self.trav.shape
        target_height = target_pos[2]
    
        # Generate points along the line from start_idx to goal_idx
        num_points = int(np.linalg.norm(goal_idx - start_idx)) + 1
        line_points = np.linspace( goal_idx,start_idx, num_points).astype(int)
    
        reachable_goal = None
        for idx in line_points:
            s, x, y = idx
            if 0 <= s < grid_shape[0] and 0 <= x < grid_shape[1] and 0 <= y < grid_shape[2]:
                elev = self.elev_g[s, x, y]
                if self.trav[s, x, y] < 20 and elev > -90 and abs(elev - target_height) < height_tol:
                    candidate_idx = np.array([s, y, x])  # Note: swap x/y if needed by your convention
                    reachable_goal = self.idx2pos_3D(candidate_idx)
                    break
    
        if reachable_goal is not None:
            if np.linalg.norm(reachable_goal+np.array([0,0,0.5]) - start_pos) < 0.3:
                rospy.logwarn("Reachable goal position is too close to the start position, skipping replan.")
                return np.array([start_pos, reachable_goal])
            planned_traj = self.plan_online(start_pos, reachable_goal+np.array([0,0,0.5]))
            return planned_traj
        else:
            return None
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32) # Swap x and y for grid indexing
        return idx
    def pos2idx_3D_plan(self, pos):
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
        idx_xy = np.round((np.array(pos[:2]) / self.voxel_size) - self.min_idx[:2]).astype(int)
        # Clip indices to valid range
        idx_xy[0] = np.clip(idx_xy[0], 0, self.elev_g.shape[2] - 1)
        idx_xy[1] = np.clip(idx_xy[1], 0, self.elev_g.shape[1] - 1)
    
        # Search for the z index (layer number) as the maximum s where elev_g[s, idx_xy[1], idx_xy[0]] <= z_height
        z_height = pos[2]  # Extract the z-coordinate
        z_idx = 0  # Default to 0 if no valid layer is found
        for s in range(self.elev_g.shape[0] - 1, -1, -1):  # Search from top down
            elev = self.elev_g[s, idx_xy[1], idx_xy[0]]
            if elev <= z_height:
                z_idx = s
                break
    
        # Combine z_idx with x and y indices
        idx = np.array([z_idx, idx_xy[0], idx_xy[1]], dtype=np.int32)
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
            # print(f"Layer height {s}: {self.elev_g[s, idx_xy[1], idx_xy[0]]}")
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
        step_x = max(1, int(1 / self.resolution))  # Step size in the x dimension
        step_y = max(1, int(1 / self.resolution))  # Step size in the y dimension
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
        idx = np.array(idx, dtype=int)  # Ensure integer indices
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
        explored_voxels = self.explored_voxels.copy()

        explored_np = cp.asnumpy(explored_voxels)
    
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
    def nextBestView(self, k_best=1):
        """
        Select the next best views using batched GPU raycasting and choose the top-k poses by reward.
        Applies masked dilation after each batch to simulate sensor footprint.

        Args:
            k_best (int): Number of top candidate viewpoints to select per iteration.
        """
        import cupyx.scipy.ndimage as cp_ndimage

        min_reward = 10  # Minimum reward threshold to consider a viewpoint valid
        best_idxs = []
        best_angles = []
        best_xyz = []
        finished = False

        sampled_points_idx, sampled_points_xyz = self.sampleUniformPointsInSpace()
        # sampled_points_xyz = sampled_points_xyz[:1]
        # sampled_points_idx = sampled_points_idx[:1]
        sampled_points_xyz = sampled_points_xyz + np.array([0, 0, 0.6])  # camera height offset

        yaw_angles = [0, 90, 180, 270]
        # yaw_angles = [0,45, 90, 135, 180,225, 270]
        grid_shape = cp.asarray(self.hash_grid.shape)
        min_idx_cp = cp.asarray(self.min_idx)

        while not finished and len(sampled_points_xyz) > 0:
            pose_list, angle_list, pose_idx_map = [], [], []
            for i, p in enumerate(sampled_points_xyz):
                for yaw in yaw_angles:
                    pose_list.append(p)
                    angle_list.append(yaw)
                    pose_idx_map.append(i)
            pose_list = np.array(pose_list)

            rewards, visibles = self.batched_ray_reward(pose_list, angle_list)

            top_k_indices = np.argsort(rewards)[-k_best:][::-1]
            top_k_indices = [i for i in top_k_indices if rewards[i] >= min_reward]

            if len(top_k_indices) == 0:
                finished = True
                break

            used_candidate_indices = set()
            new_voxel_array = cp.zeros_like(self.explored_voxels, dtype=cp.bool_)

            for i in top_k_indices:
                best_pose = pose_list[i]
                best_angle = angle_list[i]
                best_index = pose_idx_map[i]
                best_visible = visibles[i]

                if best_index in used_candidate_indices:
                    continue
                used_candidate_indices.add(best_index)

                best_idxs.append(sampled_points_idx[best_index])
                best_angles.append(best_angle)
                best_xyz.append(best_pose)

                # Add visible voxels
                v_np = np.array(list(best_visible)) - self.min_idx
                v_cp = cp.asarray(v_np)
                valid = cp.all((v_cp >= 0) & (v_cp < grid_shape), axis=1)
                v_cp = v_cp[valid]
                new_voxel_array[v_cp[:, 0], v_cp[:, 1], v_cp[:, 2]] = True

            # Dilation to fill FOV gaps
            struct = cp.ones((3, 3, 1), dtype=cp.bool_)  # x×y×z 
            dilated = cp_ndimage.binary_dilation(new_voxel_array, structure=struct)
            dilated = cp_ndimage.binary_dilation(dilated, structure=struct)

            dilated = cp.logical_and(dilated, self.hash_grid)
            self.explored_voxels = cp.logical_or(self.explored_voxels, dilated)

            # self.explored_voxels = cp.logical_or(self.explored_voxels, new_voxel_array)

            # Remove used candidate poses
            keep_mask = np.ones(len(sampled_points_xyz), dtype=bool)
            for idx in used_candidate_indices:
                keep_mask[idx] = False
            sampled_points_xyz = sampled_points_xyz[keep_mask]
            sampled_points_idx = sampled_points_idx[keep_mask]

        return best_idxs, best_angles, np.array(best_xyz) - np.array([0, 0, 0.6])



    def compute_explored_voxels(self, candidate_points_xyz, angles, use_dilation=True):
        """
        Compute the explored voxels based on candidate points and raycasting.
    
        Args:
            candidate_points_xyz (np.ndarray): Candidate points in world coordinates.
            angles (np.ndarray): Angles for each candidate point.
            use_dilation (bool): Whether to apply dilation to fill FOV gaps.
    
        Returns:
            cp.ndarray: Updated explored voxel grid.
        """
        import cupyx.scipy.ndimage as cp_ndimage
    
        explored_voxels_max = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        candidate_points_xyz = candidate_points_xyz + np.array([0, 0, 0.5])  # Adjust for z-axis
        explored_voxels_candidate = cp.zeros((len(candidate_points_xyz),) + self.hash_grid.shape, dtype=cp.bool_)
        for i, candidate_pose in enumerate(candidate_points_xyz):
            # Perform raycasting for the current pose
            print(f"Exploring candidate pose: {candidate_pose}")
            orientation = np.array([
                [np.cos(np.radians(angles[i])), -np.sin(np.radians(angles[i])), 0],
                [np.sin(np.radians(angles[i])),  np.cos(np.radians(angles[i])), 0],
                [0, 0, 1]
            ])
            visible = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape, self.hash_grid,
                self.fov_vert, self.fov_hor, self.sensor_range,
                self.resolution_raycast, n_rays=50)
            # Maximum possible visibility
            visible_max = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape, self.hash_grid,
                self.fov_vert, 360, self.sensor_range_analysis,
                self.resolution_raycast, n_rays=50)
            for v in visible_max:
                local_idx = tuple(v - self.min_idx)
                explored_voxels_max[local_idx] = True
            for v in visible:
                local_idx = tuple(v - self.min_idx)
                explored_voxels_candidate[i][local_idx] = True
    
        # --- Dilation to fill FOV gaps (same as nextBestView) ---
        if use_dilation:
            struct = cp.ones((2, 2, 1), dtype=cp.bool_)  # x×y×z
            explored_voxels_max = cp_ndimage.binary_dilation(explored_voxels_max, structure=struct)
            explored_voxels_max = cp_ndimage.binary_dilation(explored_voxels_max, structure=struct)
            explored_voxels_max = cp.logical_and(explored_voxels_max, self.hash_grid)
    
        return explored_voxels_max, explored_voxels_candidate

    def compute_and_visualise_explored_voxels(self, candidate_points_xyz, angles, use_dilation=False):
        """
        Compute the explored voxels from candidate viewpoints using raycasting and visualize them.
    
        Args:
            candidate_points_xyz (np.ndarray): Candidate camera poses in world coordinates.
            angles (np.ndarray): Yaw angles (degrees) corresponding to each candidate pose.
            use_dilation (bool): Whether to apply dilation to fill FOV gaps.
        """
        import cupyx.scipy.ndimage as cp_ndimage
    
        # Initialize voxel grid to track which voxels have been explored
        candidate_points_xyz = candidate_points_xyz + np.array([0, 0, 0.6])
        explored_voxels = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
    
        # Iterate through each candidate pose and perform raycasting
        for i, candidate_pose in enumerate(candidate_points_xyz):
            orientation = np.array([
                [np.cos(np.radians(angles[i])), -np.sin(np.radians(angles[i])), 0],
                [np.sin(np.radians(angles[i])),  np.cos(np.radians(angles[i])), 0],
                [0, 0, 1]
            ])
    
            # Perform raycasting
            visible = get_visible_voxels_first_hit(
                candidate_pose, orientation, self.voxel_size, self.min_idx, self.grid_shape,
                self.hash_grid, self.fov_vert, self.fov_hor+10, self.sensor_range_analysis,
                self.resolution_raycast, n_rays=50
            )
    
            # Mark visible voxels as explored
            for v in visible:
                local_idx = tuple(np.array(v) - self.min_idx)
                explored_voxels[local_idx] = True
    
        # --- Optionally use the same dilation as in nextBestView ---
        if use_dilation:
            struct = cp.ones((2, 2, 1), dtype=cp.bool_)  # x×y×z (z, y, x)
            dilated = cp_ndimage.binary_dilation(explored_voxels, structure=struct)
            dilated = cp_ndimage.binary_dilation(dilated, structure=struct)
            explored_voxels = cp.logical_and(dilated, self.hash_grid)
    
        # Convert voxel indices and exploration state to NumPy
        explored_np = cp.asnumpy(explored_voxels)
        hash_np = cp.asnumpy(self.hash_grid)
    
        # Get indices of occupied voxels
        voxel_indices = np.argwhere(hash_np)
    
        # Compute voxel centers in world coordinates
        voxel_centers = (voxel_indices + self.min_idx) * self.voxel_size
    
        # Build Open3D point cloud
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
        # Assign colors: red for explored, gray for unexplored
        explored_mask = explored_np[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
        colors = np.where(explored_mask[:, None], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    
        # Visualize
        o3d.visualization.draw_geometries([vis_pcd], window_name="Explored Voxel Map")
    
        return explored_voxels
    
    def batched_ray_reward(self, candidate_poses, orientations_deg):
        """
        Compute the number of new (unexplored) voxels seen from each candidate pose and orientation.
        Uses a custom CUDA kernel for ray prep.
        """
        fov_deg = self.sensor_fov
        max_range = self.sensor_range
        resolution = self.resolution_raycast
        n_rays = 10

        n = len(candidate_poses)
        n_steps = int(max_range / resolution)
        # t0 = time.time()

        # Prepare input arrays
        poses_gpu = cp.asarray(candidate_poses, dtype=cp.float64)
        yaws_gpu = cp.asarray(np.array(orientations_deg, dtype=np.float64))
        min_idx_gpu = cp.asarray(self.min_idx, dtype=cp.int32)
        grid_shape_gpu = cp.asarray(self.grid_shape, dtype=cp.int32)

        # Prepare output arrays
        idxs_out = cp.zeros((n, n_rays * n_rays, n_steps, 3), dtype=cp.int32)
        valid_out = cp.zeros((n, n_rays * n_rays, n_steps), dtype=cp.bool_)

        # Launch the ray prep kernel
        ray_prep_kernel(
            (n,), (n_rays * n_rays,),
            (
                poses_gpu,
                yaws_gpu,
                np.int32(n),
                np.int32(n_rays),
                np.int32(n_steps),
                np.float64(fov_deg),
                np.float64(-45),  # el_min_deg
                np.float64(3),    # el_max_deg
                np.float64(max_range),
                np.float64(resolution),
                np.float64(self.voxel_size),
                min_idx_gpu,
                grid_shape_gpu,
                idxs_out,
                valid_out
            )
        )
        cp.cuda.Device().synchronize()
        # t1 = time.time()

        # Flatten for raycast kernel
        idxs_flat = idxs_out.reshape(-1, 3).ravel()
        valid_flat = valid_out.ravel()
        hash_flat = self.hash_grid.ravel()
        n_total_rays = idxs_out.shape[0] * idxs_out.shape[1]
        n_steps = idxs_out.shape[2]

        visible_hits = cp.full((n_total_rays, 3), -1, dtype=cp.int32)
        hit_flags = cp.zeros((n_total_rays,), dtype=cp.int32)

        # Raycast kernel
        grid = (n_total_rays + 255) // 256
        block = 256
        ray_first_hit_kernel(
            (grid,), (block,),
            (
                idxs_flat,
                valid_flat,
                hash_flat,
                cp.int32(n_total_rays),
                cp.int32(n_steps),
                cp.int32(self.grid_shape[0]), cp.int32(self.grid_shape[1]), cp.int32(self.grid_shape[2]),
                visible_hits.ravel(),
                hit_flags
            )
        )
        cp.cuda.Device().synchronize()
        # t2 = time.time()

        visible_hits_np = visible_hits.get().reshape(n, -1, 3)
        hit_flags_np = hit_flags.get().reshape(n, -1)
        explored = cp.asnumpy(self.explored_voxels)
        min_idx_np = np.array(self.min_idx, dtype=np.int32)

        # Fast C++ post-processing
        # t3 = time.time()
        rewards, visible_voxels_list = reward_cpp.batched_reward(
            visible_hits_np, hit_flags_np, explored, min_idx_np
        )
        # t4 = time.time()

        # try:
        #     with open("batched_ray_reward_profile.csv", "a") as f:
        #         f.write(f"{n},{t1-t0:.6f},{t2-t1:.6f},{t4-t3:.6f},{t4-t0:.6f}\n")
        # except Exception as e:
        #     print(f"Failed to write timing log: {e}")

        # print(f"[batched_ray_reward] n={n} setup={t1-t0:.4f}s raycast={t2-t1:.4f}s post={t4-t3:.4f}s total={t4-t0:.4f}s")
        return rewards, visible_voxels_list

def get_visible_voxels_first_hit(candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid,
                                 fov_deg_ver=90, fov_deg_hor=90, max_range=4.0, resolution=0.2, n_rays=30):
    # Compute azimuth and elevation angles
    az = cp.linspace(-fov_deg_hor / 2, fov_deg_hor / 2, n_rays)
    el = cp.linspace(-45, 3, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    # Compute ray directions in spherical coordinates, then apply orientation
    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)
    dirs = dirs @ cp.asarray(orientation.T)

    # Sample distances and create rays
    dists = cp.arange(0, max_range - 3*resolution, resolution)
    shifted_pose = cp.asarray(candidate_pose) - cp.asarray(min_idx) * voxel_size
    rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None] + shifted_pose

    # Convert to voxel indices
    idxs = cp.floor(rays / voxel_size).astype(cp.int32)
    valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

    n_rays_total, n_steps = idxs.shape[:2]
    idxs_flat = idxs.reshape(-1, 3).ravel()
    valid_flat = valid.ravel()
    hash_flat = hash_grid.ravel()

    visible_hits = cp.full((n_rays_total, 3), -1, dtype=cp.int32)
    hit_flags = cp.zeros(n_rays_total, dtype=cp.int32)

    grid = (n_rays_total + 255) // 256
    block = 256

    ray_first_hit_kernel(
        (grid,), (block,),
        (
            idxs_flat,
            valid_flat,
            hash_flat,
            cp.int32(n_rays_total),
            cp.int32(n_steps),
            cp.int32(grid_shape[0]), cp.int32(grid_shape[1]), cp.int32(grid_shape[2]),
            visible_hits.ravel(),
            hit_flags
        )
    )
    cp.cuda.Device().synchronize()

    visible_np = visible_hits[hit_flags.astype(cp.bool_)].get()
    visible_np += min_idx  # Convert back to global coordinates
    return set(map(tuple, visible_np))


def calculate_rewards_raycast(candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid, explored_voxels,
                              fov_deg=100, max_range=10.0, resolution=0.2, n_rays=30):
    visible = get_visible_voxels_first_hit(
        candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid,
        fov_deg, fov_deg, max_range, resolution, n_rays
    )

    explored_np = cp.asnumpy(explored_voxels)
    reward = sum(1 for v in visible if not explored_np[v[0] - min_idx[0], v[1] - min_idx[1], v[2] - min_idx[2]])

    return reward, visible

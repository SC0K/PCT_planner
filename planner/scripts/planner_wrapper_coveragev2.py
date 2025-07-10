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
        self.sensor_range_analysis = 10
        self.sensor_fov = self.cfg.sensor.sensor_fov
        self.layer_modes = None
        self.fov_vert = 90
        self.fov_hor = 80

    def loadVoxelMap(self, pcd_file, voxel_size=0.2):
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
        self.hash_grid_online = self.hash_grid.copy()

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
        self.trav_raw = tomogram[5]         
        
        self.initPlanner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
        # exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)
        # layers_t : travel cost
        # trav_grad_x : gradient x
        # trav_grad_y : gradient y
        # layers_g : ground height
        # layers_c : ceiling height

        self.explored = self.initExplorationGraph()

    def initExplorationGraph(self):
        exploration_graph = np.full(self.elev_g.shape, np.nan, dtype=np.float32)
        valid_mask = self.elev_g != -100
        exploration_graph[valid_mask] = 0.0
        return exploration_graph
    

    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]       
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])   

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
        gateway[gateway_dn] = -2    
        self.gateway = gateway

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        self.planner.init_map(
        25, 20, self.resolution, self.n_slice, 0.2,
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
        self.planner.init_map(
            25, 20, self.resolution, self.n_slice, 0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            self.gateway.reshape(-1, self.gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )

    def add_obstacle_points(self, world_points, added_voxels, z_buffer=0.5, xy_buffer=0.3):
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
        if len(added_voxels) > 0:
            idxs = cp.array(added_voxels, dtype=cp.int32)
            self.hash_grid_online[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = True

        
    # def compute_adjacency_matrix(self, sampled_points_idx):
    #     """
    #     Compute an adjacency matrix where each entry represents the path length between two sampled points.
    
    #     Args:
    #         sampled_points_idx (np.ndarray): Array of sampled points' grid indices (N x 3).
    
    #     Returns:
    #         np.ndarray: Adjacency matrix of size N x N with path lengths.
    #     """
    #     num_points = sampled_points_idx.shape[0]
    #     adj_matrix = np.full((num_points, num_points), np.inf, dtype=np.float32)  # Initialize with infinity
    
    #     for i in range(num_points):
    #         for j in range(i + 1, num_points):  # Only compute for upper triangle (symmetry)
    #             # self.initPlanner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
    #             # self.init_planner(self.trav, self.trav_gx, self.trav_gy, self.elev_g, self.elev_c)
    #             # Plan a path between the two points
    #             # print("Planning path between points:", sampled_points_idx[i], sampled_points_idx[j])
    #             # self.planner.plan(sampled_points_idx[i], sampled_points_idx[j], True)
    #             # Swap x and y for planning
    #             start_idx = np.array([sampled_points_idx[i][0], sampled_points_idx[i][2], sampled_points_idx[i][1]], dtype=np.int32)
    #             end_idx = np.array([sampled_points_idx[j][0], sampled_points_idx[j][2], sampled_points_idx[j][1]], dtype=np.int32)
    #             self.planner.plan(start_idx, end_idx, False)
    #             path_finder: a_star.Astar = self.planner.get_path_finder()
    #             path = path_finder.get_result_matrix()
    
    #             if len(path) > 0:  # If a valid path exists
    #                 path_length = len(path)  # Use the number of steps as the path length
    #                 adj_matrix[i, j] = path_length
    #                 adj_matrix[j, i] = path_length  # Symmetry for undirected graph
    
    #     return adj_matrix

    def compute_adjacency_matrix(self, sampled_points_idx):
        sampled_points_flipped = sampled_points_idx.copy()
        sampled_points_flipped[:, 1], sampled_points_flipped[:, 2] = (
            sampled_points_flipped[:, 2], sampled_points_flipped[:, 1].copy()
        )
        sampled_points_o3d = o3d.utility.Vector3iVector(sampled_points_flipped.astype(np.int32))
        return self.planner.compute_adjacency_matrix(sampled_points_o3d)
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
        self.start_idx[:] = start_pos
        self.end_idx[:] = end_pos
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
        start_idx = np.round(self.pos2idx_3D_plan(start_pos)).astype(int)
        goal_idx = np.round(self.pos2idx_3D_plan(target_pos)).astype(int)
        if abs(goal_idx[1] - start_idx[1])+abs(goal_idx[2]-start_idx[2]) < 3:
            rospy.logwarn("Target position is too close to the start position, skipping replan.")
            return np.array([start_pos,target_pos])
        # start_idx = np.array([start_idx[0], start_idx[2], start_idx[1]], dtype=np.int32)  # Swap x/y for grid indexing
        # goal_idx = np.array([goal_idx[0], goal_idx[2], goal_idx[1]], dtype=np.int32)
        grid_shape = self.trav.shape
        target_height = target_pos[2]
    
        num_points = int(np.linalg.norm(goal_idx - start_idx)) + 1
        line_points = np.linspace( goal_idx,start_idx, num_points).astype(int)
    
        reachable_goal = None
        for idx in line_points:
            s, x, y = idx
            if 0 <= s < grid_shape[0] and 0 <= x < grid_shape[1] and 0 <= y < grid_shape[2]:
                elev = self.elev_g[s, x, y]
                if self.trav[s, x, y] < 20 and elev > -90 and abs(elev - target_height) < height_tol:
                    reachable_goal = np.array([s, x, y])  # Note: swap x/y if needed by convention
                    break
    
        if reachable_goal is not None:
            planned_traj = self.plan_with_idx_online(start_idx, reachable_goal)
            return planned_traj
        else:
            return None
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32) # Swap x and y for grid indexing
        return idx
    def pos2idx_3D_plan(self, pos):
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy - self.center
    
        idx_xy = np.round(pos_xy / self.resolution).astype(np.int32) + self.offset
        idx_xy = np.array([idx_xy[1], idx_xy[0]], dtype=np.int32)  # Swap x and y for grid indexing
        idx_xy = np.round((np.array(pos[:2]) / self.voxel_size) - self.min_idx[:2]).astype(int)
        idx_xy[0] = np.clip(idx_xy[0], 0, self.elev_g.shape[2] - 1)
        idx_xy[1] = np.clip(idx_xy[1], 0, self.elev_g.shape[1] - 1)
    
        z_height = pos[2]  
        z_idx = 0  
        for s in range(self.elev_g.shape[0] - 1, -1, -1):  
            elev = self.elev_g[s, idx_xy[1], idx_xy[0]]
            if elev <= z_height:
                z_idx = s
                break
    
        idx = np.array([z_idx, idx_xy[0], idx_xy[1]], dtype=np.int32)
        return idx
    def pos2idx_3D(self, pos):
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy - self.center
        
        idx_xy = np.round(pos_xy / self.resolution).astype(np.int32) + self.offset
        idx_xy = np.array([idx_xy[1], idx_xy[0]], dtype=np.int32)  # Swap x and y for grid indexing
        
        z_height = pos[2] 
        z_idx = -1 
        for s in range(self.elev_g.shape[0]):
            # print(f"Layer height {s}: {self.elev_g[s, idx_xy[1], idx_xy[0]]}")
            if abs(z_height - self.elev_g[s, idx_xy[1], idx_xy[0]]) <= self.resolution*2:
                z_idx = s
                break
        
        idx = np.array([z_idx, idx_xy[0], idx_xy[1]], dtype=np.float32)
        return idx
        
    def sampleUniformPointsInSpace(self):
        step_x = max(1, int(1.05 / self.resolution))  # Step size in the x dimension
        step_y = max(1, int(1.05 / self.resolution))  # Step size in the y dimension
        slice_indices = np.arange(0, self.elev_g.shape[0], 1)
        x_indices = np.arange(0, self.elev_g.shape[1], step_x)
        y_indices = np.arange(0, self.elev_g.shape[2], step_y)
        sampled_indices = np.array(np.meshgrid(slice_indices, x_indices, y_indices, indexing="ij"))
        sampled_indices = sampled_indices.reshape(3, -1).T  # Reshape to (N, 3)
    
        valid_indices = []
        for s, x, y in sampled_indices:
            if self.trav[s, x, y] < 15 and self.elev_g[s, x, y] > -90:
                valid_indices.append([s, x, y])
    
        valid_indices = np.array(valid_indices)
    
        unique_points = []
        seen_xy = {}
        for s, x, y in valid_indices:
            xy_key = (x, y)
            height = self.elev_g[s, x, y]
            if xy_key not in seen_xy or np.abs(seen_xy[xy_key] - height) > 0.05:
                unique_points.append([s, x, y])
                seen_xy[xy_key] = height

        unique_indices = np.array(unique_points, dtype=np.int32)

    
        sampled_xyz = np.empty((len(unique_indices), 3), dtype=np.float32)
        for idx, (s, x, y) in enumerate(unique_indices):
            map_x = (x - self.offset[0]) * self.resolution + self.center[0]
            map_y = (y - self.offset[1]) * self.resolution + self.center[1]
            map_z = self.elev_g[s, x, y]
            sampled_xyz[idx] = [map_x, map_y, map_z]
    
        return unique_indices, sampled_xyz
    def sampleUniformPointsInSpaceOnline(self, num_samples, reachable_from, center_idx, radius):
        s_c, x_c, y_c = np.round(center_idx).astype(int)
    
        s_range = range(max(0, s_c - 1), min(self.elev_g.shape[0], s_c + 2))
        x_min = max(0, int(x_c - radius / self.resolution))
        x_max = min(self.elev_g.shape[1], int(x_c + radius / self.resolution) + 1)
        y_min = max(0, int(y_c - radius / self.resolution))
        y_max = min(self.elev_g.shape[2], int(y_c + radius / self.resolution) + 1)
    
        sampled_indices = []
        for s in s_range:
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    dx = (x - x_c) * self.resolution
                    dy = (y - y_c) * self.resolution
                    if np.sqrt(dx**2 + dy**2) > radius:
                        continue
                    if self.trav[s, x, y] < 15 and self.elev_g[s, x, y] > -90:
                        sampled_indices.append([s, x, y])
    
        if not sampled_indices:
            return np.empty((0, 3), dtype=np.int32)
    
        sampled_indices = np.array(sampled_indices, dtype=np.int32)
    
        unique_points = []
        seen_xy = {}
        for s, x, y in sampled_indices:
            xy_key = (x, y)
            height = self.elev_g[s, x, y]
            if xy_key not in seen_xy or np.abs(seen_xy[xy_key] - height) > 0.05:
                unique_points.append([s, x, y])
                seen_xy[xy_key] = height
    
        unique_indices = np.array(unique_points, dtype=np.int32)
    
        if len(unique_indices) > num_samples:
            chosen = np.random.choice(len(unique_indices), num_samples, replace=False)
            unique_indices = unique_indices[chosen]
    
        # filter by reachability from reachable_from
        if reachable_from is not None:
            filtered_indices = []
            for idx in unique_indices:
                traj = self.plan_with_idx_online(reachable_from, idx)
                if traj is not None and len(traj) > 1:
                    filtered_indices.append(idx)
            unique_indices = np.array(filtered_indices, dtype=np.int32)
    
        return unique_indices
    def filter_reachable_candidates(self, candidate_indices, candidate_xyz, reference_idx=None):
        if reference_idx is None:
            reference_idx = 0  
        start_idx = candidate_indices[reference_idx]
        start_xyz = candidate_xyz[reference_idx]
    
        reachable_mask = []
        for i, (idx, xyz) in enumerate(zip(candidate_indices, candidate_xyz)):
            if i == reference_idx:
                reachable_mask.append(True)
                continue
            traj = self.plan_with_idx_online(start_idx, idx)
            if traj is not None and len(traj) > 1:
                reachable_mask.append(True)
            else:
                reachable_mask.append(False)
        reachable_mask = np.array(reachable_mask, dtype=bool)
        filtered_indices = candidate_indices[reachable_mask]
        filtered_xyz = candidate_xyz[reachable_mask]
        return filtered_indices, filtered_xyz
    def sampleTraversablePoints_rad(self, num_samples):
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
        idx = np.array(idx, dtype=int)
        map_x = (idx[1] - self.offset[0]) * self.resolution + self.center[0]
        map_y = (idx[2] - self.offset[1]) * self.resolution + self.center[1]
        map_z = self.elev_g[idx[0], idx[1], idx[2]]
        return np.array([map_x, map_y, map_z], dtype=np.float32)
    
    def visualizeExploredGrid(self):
        explored_voxels = self.explored_voxels.copy()

        explored_np = cp.asnumpy(explored_voxels)
    
        voxel_indices = np.argwhere(explored_np)
    
        voxel_centers = (voxel_indices + self.min_idx) * self.voxel_size
    
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
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
        import cupyx.scipy.ndimage as cp_ndimage

        min_reward = 10  # Minimum reward threshold to consider a viewpoint valid
        best_idxs = []
        best_angles = []
        best_xyz = []
        finished = False

        sampled_points_idx, sampled_points_xyz = self.sampleUniformPointsInSpace()
        # sampled_points_idx, sampled_points_xyz = self.filter_reachable_candidates(sampled_points_xyz_raw, sampled_points_idx_raw)
        # sampled_points_xyz = sampled_points_xyz[:5]
        # sampled_points_idx = sampled_points_idx[:5]
        sampled_points_xyz = sampled_points_xyz + np.array([0, 0, 0.5])  # camera height offset

        yaw_angles = [0, 90, 180, 270]
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

                v_np = np.array(list(best_visible)) - self.min_idx
                v_cp = cp.asarray(v_np)
                valid = cp.all((v_cp >= 0) & (v_cp < grid_shape), axis=1)
                v_cp = v_cp[valid]
                new_voxel_array[v_cp[:, 0], v_cp[:, 1], v_cp[:, 2]] = True

            # Dilation to fill FOV gaps
            struct = cp.ones((2, 2, 1), dtype=cp.bool_)  # x×y×z 
            dilated = cp_ndimage.binary_dilation(new_voxel_array, structure=struct)
            dilated = cp_ndimage.binary_dilation(dilated, structure=struct)

            dilated = cp.logical_and(dilated, self.hash_grid)
            self.explored_voxels = cp.logical_or(self.explored_voxels, dilated)

            # self.explored_voxels = cp.logical_or(self.explored_voxels, new_voxel_array)

            keep_mask = np.ones(len(sampled_points_xyz), dtype=bool)
            for idx in used_candidate_indices:
                keep_mask[idx] = False
            sampled_points_xyz = sampled_points_xyz[keep_mask]
            sampled_points_idx = sampled_points_idx[keep_mask]

        return best_idxs, best_angles, np.array(best_xyz) - np.array([0, 0, 0.6])

    def recompute_visible_voxels_online(self, candidate_points_xyz, angles_deg):
        candidate_points_xyz = np.asarray(candidate_points_xyz)
        angles_deg = np.asarray(angles_deg)
        assert candidate_points_xyz.shape[0] == angles_deg.shape[0], "Number of poses and angles must match"
    
        candidate_points_xyz = candidate_points_xyz
    
        _, visible_voxels_list = self.batched_ray_reward(candidate_points_xyz, angles_deg)
        return visible_voxels_list

    def compute_explored_voxels(self, candidate_points_xyz, angles, use_dilation=True):
        import cupyx.scipy.ndimage as cp_ndimage
    
        explored_voxels_max = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        candidate_points_xyz = candidate_points_xyz + np.array([0, 0, 0.5])  # Adjust for z-axis
        explored_voxels_candidate = cp.zeros((len(candidate_points_xyz),) + self.hash_grid.shape, dtype=cp.bool_)
        for i, candidate_pose in enumerate(candidate_points_xyz):
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
    
        if use_dilation:
            struct = cp.ones((2, 2, 1), dtype=cp.bool_)  # x×y×z
            explored_voxels_max = cp_ndimage.binary_dilation(explored_voxels_max, structure=struct)
            explored_voxels_max = cp_ndimage.binary_dilation(explored_voxels_max, structure=struct)
            explored_voxels_max = cp.logical_and(explored_voxels_max, self.hash_grid)
    
        return explored_voxels_max, explored_voxels_candidate
    def simulate_visibility(self, idx, angle, fov_vert=None, fov_hor=None, sensor_range=None, n_rays=30):
        if fov_vert is None:
            fov_vert = self.fov_vert
        if fov_hor is None:
            fov_hor = self.fov_hor
        if sensor_range is None:
            sensor_range = self.sensor_range
    
        candidate_pose = self.idx2pos_3D(idx)
        poses = np.array([candidate_pose])
        angles = np.array([angle])
    
        _, visibles = self.batched_ray_reward(poses, angles)
        visible = visibles[0] 
    
        if len(visible) == 0:
            return cp.zeros(self.hash_grid.shape, dtype=cp.bool_)
    
        visible_arr = cp.array(list(visible))  
        local_idx = visible_arr - cp.asarray(self.min_idx) 
    
        valid = cp.all((local_idx >= 0) & (local_idx < cp.asarray(self.hash_grid.shape)), axis=1)
        local_idx = local_idx[valid]
    
        mask = cp.zeros(self.hash_grid.shape, dtype=cp.bool_)
        if local_idx.shape[0] > 0:
            mask[local_idx[:, 0], local_idx[:, 1], local_idx[:, 2]] = True
        return mask
    def compute_and_visualise_explored_voxels(self, candidate_points_xyz, angles, use_dilation=True):
        import cupyx.scipy.ndimage as cp_ndimage
    
        candidate_points_xyz = candidate_points_xyz + np.array([0, 0, 0.5])
        angles = np.asarray(angles)
        # Use batched_ray_reward for all candidate poses and angles
        rewards, visibles = self.batched_ray_reward(candidate_points_xyz, angles)
    
        explored_voxels = cp.zeros_like(self.hash_grid, dtype=cp.bool_)
        for visible in visibles:
            for v in visible:
                local_idx = tuple(np.array(v) - self.min_idx)
                explored_voxels[local_idx] = True
    
        if use_dilation:
            struct = cp.ones((2, 2, 1), dtype=cp.bool_)  # x×y×z (z, y, x)
            dilated = cp_ndimage.binary_dilation(explored_voxels, structure=struct)
            dilated = cp_ndimage.binary_dilation(dilated, structure=struct)
            explored_voxels = cp.logical_and(dilated, self.hash_grid)
    
        explored_np = cp.asnumpy(explored_voxels)
        hash_np = cp.asnumpy(self.hash_grid)
    
        voxel_indices = np.argwhere(hash_np)
        voxel_centers = (voxel_indices + self.min_idx) * self.voxel_size
    
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    
        explored_mask = explored_np[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
        colors = np.where(explored_mask[:, None], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    
        o3d.visualization.draw_geometries([vis_pcd], window_name="Explored Voxel Map")
    
    def batched_ray_reward(self, candidate_poses, orientations_deg):
        fov_deg = self.sensor_fov
        max_range = self.sensor_range
        resolution = self.resolution_raycast
        n_rays = 30

        n = len(candidate_poses)
        n_steps = int(max_range / resolution)
        # t0 = time.time()

        poses_gpu = cp.asarray(candidate_poses, dtype=cp.float64)
        yaws_gpu = cp.asarray(np.array(orientations_deg, dtype=np.float64))
        min_idx_gpu = cp.asarray(self.min_idx, dtype=cp.int32)
        grid_shape_gpu = cp.asarray(self.grid_shape, dtype=cp.int32)

        idxs_out = cp.zeros((n, n_rays * n_rays, n_steps, 3), dtype=cp.int32)
        valid_out = cp.zeros((n, n_rays * n_rays, n_steps), dtype=cp.bool_)

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
                np.float64(4),    # el_max_deg
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

        idxs_flat = idxs_out.reshape(-1, 3).ravel()
        valid_flat = valid_out.ravel()
        hash_flat = self.hash_grid.ravel()
        n_total_rays = idxs_out.shape[0] * idxs_out.shape[1]
        n_steps = idxs_out.shape[2]

        visible_hits = cp.full((n_total_rays, 3), -1, dtype=cp.int32)
        hit_flags = cp.zeros((n_total_rays,), dtype=cp.int32)

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
    def batched_ray_reward_online(self, candidate_poses, orientations_deg):
        fov_deg = self.sensor_fov
        max_range = self.sensor_range
        resolution = self.resolution_raycast
        n_rays = 30

        n = len(candidate_poses)
        n_steps = int(max_range / resolution)
        # t0 = time.time()

        poses_gpu = cp.asarray(candidate_poses, dtype=cp.float64)
        yaws_gpu = cp.asarray(np.array(orientations_deg, dtype=np.float64))
        min_idx_gpu = cp.asarray(self.min_idx, dtype=cp.int32)
        grid_shape_gpu = cp.asarray(self.grid_shape, dtype=cp.int32)

        idxs_out = cp.zeros((n, n_rays * n_rays, n_steps, 3), dtype=cp.int32)
        valid_out = cp.zeros((n, n_rays * n_rays, n_steps), dtype=cp.bool_)

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
                np.float64(4),    # el_max_deg
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

        idxs_flat = idxs_out.reshape(-1, 3).ravel()
        valid_flat = valid_out.ravel()
        hash_flat = self.hash_grid_online.ravel()
        n_total_rays = idxs_out.shape[0] * idxs_out.shape[1]
        n_steps = idxs_out.shape[2]

        visible_hits = cp.full((n_total_rays, 3), -1, dtype=cp.int32)
        hit_flags = cp.zeros((n_total_rays,), dtype=cp.int32)

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
    az = cp.linspace(-fov_deg_hor / 2, fov_deg_hor / 2, n_rays)
    el = cp.linspace(-60, 60, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)
    dirs = dirs @ cp.asarray(orientation.T)

    dists = cp.arange(0, max_range - 3*resolution, resolution)
    shifted_pose = cp.asarray(candidate_pose) - cp.asarray(min_idx) * voxel_size
    rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None] + shifted_pose

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

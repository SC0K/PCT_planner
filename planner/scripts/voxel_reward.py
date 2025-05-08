import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange
import time

@njit(parallel=True)
def calculate_voxel_rewards_numba(sampled_points_xyz, gt_centers, scanned_mask, fov_deg, max_range, resolution, tolerance):
    num_candidates = sampled_points_xyz.shape[0]
    num_voxels = gt_centers.shape[0]
    rewards = np.zeros(num_candidates, dtype=np.int32)
    new_voxel_flags = np.zeros((num_candidates, num_voxels), dtype=np.uint8)

    azimuths = np.linspace(-fov_deg / 2, fov_deg / 2, 30)
    elevations = np.linspace(-fov_deg / 2, fov_deg / 2, 30)

    for i in prange(num_candidates):
        origin = sampled_points_xyz[i]
        for az in azimuths:
            for el in elevations:
                az_rad = np.radians(az)
                el_rad = np.radians(el)
                dir_vec = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad),
                    np.sin(el_rad)
                ])
                for d in np.arange(0.0, max_range, resolution):
                    point = origin + d * dir_vec

                    # Brute-force nearest neighbor
                    min_dist = 1e6
                    min_idx = -1
                    for j in range(num_voxels):
                        dist = np.linalg.norm(point - gt_centers[j])
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = j

                    if min_dist < tolerance and scanned_mask[min_idx] == 0:
                        new_voxel_flags[i, min_idx] = 1
                        break

        rewards[i] = np.sum(new_voxel_flags[i])

    return rewards, new_voxel_flags


class VoxelPlanner:
    def __init__(self, pcd_path, voxel_size=0.2, resolution=0.2, sensor_range=5.0, sensor_fov=120, coverage_threshold=0.95):
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.sensor_range = sensor_range
        self.sensor_fov = sensor_fov
        self.coverage_threshold = coverage_threshold

        # Load point cloud and voxelize
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)

        self.gt_centers = np.array([
            self.voxel_grid.get_voxel_center_coordinate(v.grid_index)
            for v in self.voxel_grid.get_voxels()
        ]).astype(np.float32)

        self.scanned_mask = np.zeros(len(self.gt_centers), dtype=np.uint8)

    def nextBestView(self, N_samples=100, min_reward=50):
        finished = False
        sampled_points_xyz = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/sampled_points.npy")
        target_voxel_count = len(self.gt_centers)

        while not finished:
            coverage = np.sum(self.scanned_mask) / target_voxel_count
            print(f"Coverage: {np.sum(self.scanned_mask)} / {target_voxel_count} ({coverage:.2%})")

            if coverage >= self.coverage_threshold:
                print("Coverage threshold reached.")
                break
            start_time = time.time()
            rewards, new_voxel_flags = calculate_voxel_rewards_numba(
                sampled_points_xyz,
                self.gt_centers,
                self.scanned_mask,
                self.sensor_fov,
                self.sensor_range,
                self.resolution,
                self.voxel_size * 1.5
            )

            best_reward_index = np.argmax(rewards)
            best_reward = rewards[best_reward_index]

            if best_reward < min_reward:
                print("No sufficiently informative viewpoint found.")
                break
            end_time = time.time()
            print(f"Selected pose: {sampled_points_xyz[best_reward_index]}, reward: {best_reward}")

            # Update scanned mask
            self.scanned_mask |= new_voxel_flags[best_reward_index]

            # Remove selected point
            sampled_points_xyz = np.delete(sampled_points_xyz, best_reward_index, axis=0)

        return self.scanned_mask
if __name__ == "__main__":
        # Example usage
    planner = VoxelPlanner(
        pcd_path="/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_LEE.pcd",
        voxel_size=0.2,
        resolution=0.2,
        sensor_range=5.0,
        sensor_fov=120,
        coverage_threshold=0.95
    )

    scanned_mask = planner.nextBestView(N_samples=150)

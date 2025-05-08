import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import time

# Parameters
voxel_size = 0.2
tolerance = 0.15
fov_deg = 120
max_range = 5.0

# Load and downsample map
pcd = o3d.io.read_point_cloud("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
voxel_grid_gt = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
gt_centers = np.array([
    voxel_grid_gt.get_voxel_center_coordinate(v.grid_index)
    for v in voxel_grid_gt.get_voxels()
])
tree = cKDTree(gt_centers)
scanned_mask = set()

# -- raycasting --
def get_visible_voxels(candidate_position, orientation, gt_centers, scanned_mask, fov_deg=120, max_range=5.0, resolution=0.2, tolerance=0.1):
    tree = cKDTree(gt_centers)
    visible_voxels = set()

    start_time = time.time()
    for az in np.linspace(-fov_deg / 2, fov_deg / 2, 50):
        for el in np.linspace(-fov_deg / 2, fov_deg / 2, 50):
            az_rad = np.radians(az)
            el_rad = np.radians(el)
            dir_vec = np.array([
                np.cos(el_rad) * np.cos(az_rad),
                np.cos(el_rad) * np.sin(az_rad),
                np.sin(el_rad)
            ])
            dir_vec = orientation @ dir_vec

            for d in np.arange(0.0, max_range, resolution):
                point = candidate_position + d * dir_vec
                dist, idx = tree.query(point)
                if dist < tolerance and idx not in scanned_mask:
                    visible_voxels.add(idx)
                    break 
    end_time = time.time()
    return visible_voxels

def compute_reward(candidate_position, orientation, gt_centers, scanned_mask, **kwargs):
    visible = get_visible_voxels(candidate_position, orientation, gt_centers, scanned_mask, **kwargs)
    return len(visible)

def visualize_visible_voxels(sensor_position, orientation, gt_centers, scanned_mask, **kwargs):
    start_time = time.time()
    visible_indices = get_visible_voxels(sensor_position, orientation, gt_centers, scanned_mask, **kwargs)

    vis_pcd = o3d.geometry.PointCloud()
    points = []
    colors = []
    for i, pt in enumerate(gt_centers):
        points.append(pt)
        if i in visible_indices:
            colors.append([1.0, 0.0, 0.0])  # red
        elif i in scanned_mask:
            colors.append([0, 1, 0])  # green
        else:
            colors.append([0.5, 0.5, 0.5])  # gray
    end_time = time.time()
    print(f"Time taken to visualize: {end_time - start_time:.4f} seconds")

    vis_pcd.points = o3d.utility.Vector3dVector(points)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_pcd])

candidate_pose = np.array([4.0, -1.0, 1.5])

# Define orientation as a rotation matrix (e.g., 45 degrees yaw)
yaw_angle = np.radians(45)  
orientation = np.array([
    [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
    [np.sin(yaw_angle),  np.cos(yaw_angle), 0],
    [0,                 0,                 1]
])

reward = compute_reward(candidate_pose, orientation, gt_centers, scanned_mask, fov_deg=fov_deg, max_range=max_range, tolerance=tolerance)
print(f"Reward at pose {candidate_pose} with orientation:\n{orientation}\nReward: {reward}")

# Visualize
visualize_visible_voxels(candidate_pose, orientation, gt_centers, scanned_mask, fov_deg=fov_deg, max_range=max_range, tolerance=tolerance)
import open3d as o3d
import numpy as np
import cupy as cp
import time

# === Parameters ===
voxel_size = 0.2
fov_deg = 90
max_range = 5.0
resolution = 0.1
n_rays = 50

# === Load and voxelize point cloud ===
pcd = o3d.io.read_point_cloud("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

voxel_centers = np.array([
    voxel_grid.get_voxel_center_coordinate(v.grid_index)
    for v in voxel_grid.get_voxels()
])
voxel_indices = np.floor(voxel_centers / voxel_size).astype(np.int32)
min_idx = voxel_indices.min(axis=0)
max_idx = voxel_indices.max(axis=0)
grid_shape = max_idx - min_idx + 1

# Hash grid
shifted_indices = voxel_indices - min_idx
hash_grid = cp.zeros(tuple(grid_shape), dtype=cp.bool_)
for idx in shifted_indices:
    hash_grid[tuple(idx)] = True

# === Raycasting ===
def get_visible_voxels_first_hit(candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid,
                                 fov_deg=90, max_range=5.0, resolution=0.2, n_rays=30):
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

# === Pose & Orientation ===
candidate_pose = np.array([2.0, -3.0, 1.0])
yaw = np.radians(45)
orientation = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0, 0, 1]
])


start = time.time()
visible_voxels = get_visible_voxels_first_hit(
    candidate_pose, orientation, voxel_size, min_idx, grid_shape, hash_grid,
    fov_deg=fov_deg, max_range=max_range, resolution=resolution, n_rays=n_rays
)
end = time.time()

print(f"\nVisible voxels (first hit only): {len(visible_voxels)}")
print(f"Raycasting time: {end - start:.3f} s")

# === Visualize ===
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
colors = []
vox_set = set(map(tuple, voxel_indices))
for v in voxel_indices:
    if tuple(v) in visible_voxels:
        colors.append([1.0, 0.0, 0.0])  # red
    else:
        colors.append([0.5, 0.5, 0.5])  # gray
vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
o3d.visualization.draw_geometries([vis_pcd])

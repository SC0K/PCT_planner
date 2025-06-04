import open3d as o3d
import numpy as np
import cupy as cp
import time

# === Parameters ===
voxel_size = 0.2
fov_deg = 30
max_range = 4
resolution = 0.2
n_rays = 20
pcd_path = "/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd"

# === Load and shift point cloud ===
pcd = o3d.io.read_point_cloud(pcd_path)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
points = np.asarray(pcd.points)
min_bound = points.min(axis=0)
points -= min_bound
pcd.points = o3d.utility.Vector3dVector(points)

# === Voxelization ===
voxel_indices = np.floor(points / voxel_size).astype(np.int32)
voxel_indices = np.unique(voxel_indices, axis=0)
grid_shape = voxel_indices.max(axis=0) + 1

# === Create occupancy grid ===
hash_grid = cp.zeros(tuple(grid_shape), dtype=cp.bool_)
voxel_cp = cp.asarray(voxel_indices)
hash_grid[voxel_cp[:, 0], voxel_cp[:, 1], voxel_cp[:, 2]] = True

# === RawKernel code (fixed flattening order) ===
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
ray_first_hit_kernel = cp.RawKernel(ray_hit_kernel_code, "ray_first_hit")

# === Raycasting function ===
def raycast_first_hits(camera_pose, orientation):
    az = cp.linspace(-fov_deg / 2, fov_deg / 2, n_rays)
    el = cp.linspace(-30, 30, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)
    dirs = dirs @ cp.asarray(orientation.T)

    dists = cp.arange(0, max_range, resolution)
    rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None] + cp.asarray(camera_pose)

    idxs = cp.floor(rays / voxel_size).astype(cp.int32)
    valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

    n_rays_total, n_steps = idxs.shape[:2]
    idxs_flat = idxs.reshape(-1, 3).ravel()
    valid_flat = valid.ravel()
    hash_flat = hash_grid.ravel()

    visible_hits = cp.full((n_rays_total, 3), -1, dtype=cp.int32)
    hit_flags = cp.zeros((n_rays_total,), dtype=cp.int32)

    grid = (n_rays_total + 255) // 256
    block = 256

    start = time.time()
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
    end = time.time()
    print(f"Raycasting time: {end - start:.4f} s")

    visible_cpu = visible_hits[hit_flags.astype(cp.bool_)].get()
    return set(map(tuple, visible_cpu)), rays.get(), hit_flags.get()

# === Camera pose and orientation ===
camera_pose = np.array([1.0, -2.0, 0.5]) + (-min_bound)
yaw = np.radians(0)
orientation = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0, 0, 1]
])

# === Run raycasting ===
visible_voxels, rays_all, hits = raycast_first_hits(camera_pose, orientation)
print(f"Visible voxel count: {len(visible_voxels)}")

# === Visualization ===
# Occupied voxel centers
voxel_centers = (voxel_indices * voxel_size).astype(np.float32)
colors = []
for v in voxel_indices:
    if tuple(v) in visible_voxels:
        colors.append([1.0, 0.0, 0.0])  # red = visible
    else:
        colors.append([0.5, 0.5, 0.5])  # gray = not hit

vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# === Ray visualization (safe) ===
rays_vis = []
visible_voxels_list = list(visible_voxels)
visible_idx = 0
for i in range(hits.shape[0]):
    ray_pts = rays_all[i]
    if hits[i] == 1 and visible_idx < len(visible_voxels_list):
        hit_voxel = visible_voxels_list[visible_idx]
        hit_coords = np.floor(ray_pts / voxel_size).astype(int)
        match = np.all(hit_coords == hit_voxel, axis=1)
        if np.any(match):
            end_idx = np.argmax(match)
            ray_pts = ray_pts[:end_idx + 1]
        visible_idx += 1
    ray_line = o3d.geometry.LineSet()
    ray_line.points = o3d.utility.Vector3dVector(ray_pts)
    ray_line.lines = o3d.utility.Vector2iVector([[j, j + 1] for j in range(len(ray_pts) - 1)])
    ray_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * (len(ray_pts) - 1))  # green
    rays_vis.append(ray_line)

# === Show scene ===
o3d.visualization.draw_geometries([vis_pcd])

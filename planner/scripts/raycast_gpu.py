# import open3d as o3d
# import numpy as np
# import cupy as cp
# import time

# # === Parameters ===
# voxel_size = 0.2
# fov_deg = 100
# max_range = 4
# resolution = 0.2
# n_rays = 10
# pcd_path = "/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd"

# # === Load and voxelize point cloud ===
# pcd = o3d.io.read_point_cloud(pcd_path)
# pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# voxel_centers = np.array([
#     voxel_grid.get_voxel_center_coordinate(v.grid_index)
#     for v in voxel_grid.get_voxels()
# ])
# voxel_indices = np.floor(voxel_centers / voxel_size).astype(np.int32)
# min_idx = voxel_indices.min(axis=0)
# max_idx = voxel_indices.max(axis=0)
# grid_shape = max_idx - min_idx + 1

# # === Create occupancy grid ===
# shifted_indices = voxel_indices - min_idx
# hash_grid = cp.zeros(tuple(grid_shape), dtype=cp.bool_)
# for idx in shifted_indices:
#     hash_grid[tuple(idx)] = True

# # === RawKernel code ===
# ray_hit_kernel_code = r'''
# extern "C" __global__
# void ray_first_hit(const int* idxs, const bool* valid, const bool* hash,
#                    int n_rays, int n_steps,
#                    int gx, int gy, int gz,
#                    int* visible_hits, int* hit_flags) {
#     int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
#     if (ray_id >= n_rays) return;

#     for (int s = 0; s < n_steps; ++s) {
#         int base_idx = (ray_id * n_steps + s) * 3;
#         int valid_idx = ray_id * n_steps + s;

#         if (!valid[valid_idx]) continue;

#         int i = idxs[base_idx + 0];
#         int j = idxs[base_idx + 1];
#         int k = idxs[base_idx + 2];

#         if (i < 0 || i >= gx || j < 0 || j >= gy || k < 0 || k >= gz) continue;

#         int flat_idx = i * (gy * gz) + j * gz + k;

#         if (hash[flat_idx]) {
#             int out_idx = ray_id * 3;
#             visible_hits[out_idx + 0] = i;
#             visible_hits[out_idx + 1] = j;
#             visible_hits[out_idx + 2] = k;
#             hit_flags[ray_id] = 1;
#             return;
#         }
#     }
# }
# '''
# ray_first_hit_kernel = cp.RawKernel(ray_hit_kernel_code, "ray_first_hit")

# # === Raycasting function ===
# def raycast_first_hits(camera_pose, orientation):
#     az = cp.linspace(-fov_deg / 2, fov_deg / 2, n_rays)
#     el = cp.linspace(-45, 5, n_rays)
#     az_grid, el_grid = cp.meshgrid(az, el)
#     az_flat = cp.radians(az_grid.flatten())
#     el_flat = cp.radians(el_grid.flatten())

#     dirs = cp.stack([
#         cp.cos(el_flat) * cp.cos(az_flat),
#         cp.cos(el_flat) * cp.sin(az_flat),
#         cp.sin(el_flat)
#     ], axis=1)
#     dirs = dirs @ cp.asarray(orientation.T)

#     dists = cp.arange(0, max_range, resolution)
    
#     # Shift camera pose from world to local (voxel grid) coordinates
#     cam_shifted = cp.asarray(camera_pose) - cp.asarray(min_idx) * voxel_size
#     rays = dirs[:, cp.newaxis, :] * dists[cp.newaxis, :, None] + cam_shifted

#     idxs = cp.floor(rays / voxel_size).astype(cp.int32)
#     valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

#     n_rays_total, n_steps = idxs.shape[:2]
#     idxs_flat = idxs.reshape(-1, 3).ravel()
#     valid_flat = valid.ravel()
#     hash_flat = hash_grid.ravel()

#     visible_hits = cp.full((n_rays_total, 3), -1, dtype=cp.int32)
#     hit_flags = cp.zeros((n_rays_total,), dtype=cp.int32)

#     grid = (n_rays_total + 255) // 256
#     block = 256

#     start = time.time()
#     ray_first_hit_kernel(
#         (grid,), (block,),
#         (
#             idxs_flat,
#             valid_flat,
#             hash_flat,
#             cp.int32(n_rays_total),
#             cp.int32(n_steps),
#             cp.int32(grid_shape[0]), cp.int32(grid_shape[1]), cp.int32(grid_shape[2]),
#             visible_hits.ravel(),
#             hit_flags
#         )
#     )
#     cp.cuda.Device().synchronize()
#     end = time.time()
#     print(f"Raycasting time: {end - start:.4f} s")

#     visible_np = visible_hits[hit_flags.astype(cp.bool_)].get()
#     visible_np += min_idx  # convert local to global indices
#     return set(map(tuple, visible_np)), rays.get(), hit_flags.get()

# # === Camera pose and orientation ===
# camera_pose = np.array([1, 1.0         ,3.30000029])  # world coordinates
# yaw = np.radians(0)
# orientation = np.array([
#     [np.cos(yaw), -np.sin(yaw), 0],
#     [np.sin(yaw),  np.cos(yaw), 0],
#     [0, 0, 1]
# ])

# # === Run raycasting ===
# visible_voxels, rays_all, hits = raycast_first_hits(camera_pose, orientation)
# print(f"Visible voxel count: {len(visible_voxels)}")

# # === Visualization ===
# vis_pcd = o3d.geometry.PointCloud()
# vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
# colors = []
# for v in voxel_indices:
#     if tuple(v) in visible_voxels:
#         colors.append([1.0, 0.0, 0.0])  # red = visible
#     else:
#         colors.append([0.5, 0.5, 0.5])  # gray = not hit
# vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# # Optional ray visualization
# rays_vis = []
# visible_voxels_list = list(visible_voxels)
# visible_idx = 0
# for i in range(hits.shape[0]):
#     ray_pts = rays_all[i]
#     if hits[i] == 1 and visible_idx < len(visible_voxels_list):
#         hit_voxel = visible_voxels_list[visible_idx]
#         hit_coords = np.floor(ray_pts / voxel_size).astype(int) + min_idx
#         match = np.all(hit_coords == hit_voxel, axis=1)
#         if np.any(match):
#             end_idx = np.argmax(match)
#             ray_pts = ray_pts[:end_idx + 1]
#         visible_idx += 1
#     ray_line = o3d.geometry.LineSet()
#     ray_line.points = o3d.utility.Vector3dVector(ray_pts)
#     ray_line.lines = o3d.utility.Vector2iVector([[j, j + 1] for j in range(len(ray_pts) - 1)])
#     ray_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * (len(ray_pts) - 1))  # green
#     rays_vis.append(ray_line)
# import cupyx.scipy.ndimage as cp_ndimage

# # === Initialize explored_voxels grid ===
# explored_voxels = cp.zeros_like(hash_grid, dtype=cp.bool_)

# # === Mark newly visible voxels ===
# for v in visible_voxels:
#     local_idx = tuple(np.array(v) - min_idx)
#     explored_voxels[local_idx] = True

# # === Dilate only inside occupied domain ===
# dilated = cp_ndimage.binary_dilation(explored_voxels, iterations=1, structure=cp.ones((3, 3, 3), dtype=cp.bool_))
# dilated = cp.logical_and(dilated, hash_grid)  # Mask out invalid space

# # === Update explored_voxels state ===
# explored_voxels = cp.logical_or(explored_voxels, dilated)

# # === Optional: color based on final explored_voxels state ===
# explored_np = cp.asnumpy(explored_voxels)
# colors = []
# for idx in voxel_indices:
#     local_idx = tuple(idx - min_idx)
#     if explored_np[local_idx]:
#         colors.append([1.0, 0.0, 0.0])  # Red = explored
#     else:
#         colors.append([0.5, 0.5, 0.5])  # Gray = unexplored
# vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
# # === Show scene ===
# o3d.visualization.draw_geometries([vis_pcd] + rays_vis)

######################## batched raycasting example ##########
import open3d as o3d
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import time

# === Parameters ===
voxel_size = 0.2
fov_deg = 100
max_range = 4
resolution = 0.2
n_rays = 20
pcd_path = "/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd"

# === Load and voxelize point cloud ===
pcd = o3d.io.read_point_cloud(pcd_path)
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

# === Create occupancy grid ===
shifted_indices = voxel_indices - min_idx
hash_grid = cp.zeros(tuple(grid_shape), dtype=cp.bool_)
for idx in shifted_indices:
    hash_grid[tuple(idx)] = True

# Initialize explored voxel map
explored_voxels = cp.zeros_like(hash_grid)

# === RawKernel code ===
ray_hit_kernel_code = r'''
extern "C" __global__
void ray_first_hit(const int* idxs, const bool* valid, const bool* hash,
                   int total_rays, int n_steps,
                   int gx, int gy, int gz,
                   int* visible_hits, int* hit_flags) {
    int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ray_id >= total_rays) return;

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

# === Raycasting Function with reward tracking ===
def raycast_multiple_poses_with_rewards(camera_poses, orientations):
    n_poses = len(camera_poses)
    az = cp.linspace(-fov_deg / 2, fov_deg / 2, n_rays)
    el = cp.linspace(-45, 5, n_rays)
    az_grid, el_grid = cp.meshgrid(az, el)
    az_flat = cp.radians(az_grid.flatten())
    el_flat = cp.radians(el_grid.flatten())

    dirs = cp.stack([
        cp.cos(el_flat) * cp.cos(az_flat),
        cp.cos(el_flat) * cp.sin(az_flat),
        cp.sin(el_flat)
    ], axis=1)

    dirs_all = cp.stack([dirs @ cp.asarray(ori.T) for ori in orientations])
    dists = cp.arange(0, max_range, resolution)
    cam_shifted = cp.asarray(camera_poses) - cp.asarray(min_idx) * voxel_size

    rays = dirs_all[:, :, cp.newaxis, :] * dists[None, :, None] + cam_shifted[:, None, None, :]
    idxs = cp.floor(rays / voxel_size).astype(cp.int32)
    valid = cp.all((idxs >= 0) & (idxs < cp.asarray(grid_shape)), axis=-1)

    n_total_rays = idxs.shape[0] * idxs.shape[1]
    n_steps = idxs.shape[2]

    idxs_flat = idxs.reshape(-1, 3).ravel()
    valid_flat = valid.ravel()
    hash_flat = hash_grid.ravel()

    visible_hits = cp.full((n_total_rays, 3), -1, dtype=cp.int32)
    hit_flags = cp.zeros((n_total_rays,), dtype=cp.int32)

    grid = (n_total_rays + 255) // 256
    block = 256

    start = time.time()
    ray_first_hit_kernel(
        (grid,), (block,),
        (
            idxs_flat,
            valid_flat,
            hash_flat,
            cp.int32(n_total_rays),
            cp.int32(n_steps),
            cp.int32(grid_shape[0]), cp.int32(grid_shape[1]), cp.int32(grid_shape[2]),
            visible_hits.ravel(),
            hit_flags
        )
    )
    cp.cuda.Device().synchronize()
    end = time.time()
    print(f"Batched raycasting for {n_poses} poses took: {end - start:.4f} s")

    # Reshape and collect rewards
    visible_hits_np = visible_hits.get().reshape(n_poses, -1, 3)
    hit_flags_np = hit_flags.get().reshape(n_poses, -1)

    rewards = []
    per_view_visible_voxels = []

    for i in range(n_poses):
        hits_i = visible_hits_np[i][hit_flags_np[i] == 1] + min_idx
        voxel_set_i = set(map(tuple, hits_i))
        per_view_visible_voxels.append(voxel_set_i)
        rewards.append(len(voxel_set_i))

        # Mark explored (on GPU)
        for v in voxel_set_i:
            local_idx = tuple(np.array(v) - min_idx)
            explored_voxels[local_idx] = True

    # === Perform dilation on explored region ===
    dilated = cp_ndimage.binary_dilation(explored_voxels, iterations=1, structure=cp.ones((3, 3, 3), dtype=cp.bool_))
    dilated = cp.logical_and(dilated, hash_grid)
    explored_voxels[:] = cp.logical_or(explored_voxels, dilated)

    return per_view_visible_voxels, rewards

# === Example usage ===
camera_poses = np.array([
    [5, 2.0, 0.3],
    [0, 0.0, 0.3]
])
orientations = [np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0, 0, 1]
]) for yaw in [0, np.pi / 4, np.pi / 2]]

voxels_per_pose, rewards = raycast_multiple_poses_with_rewards(camera_poses, orientations)

# === Print reward per viewpoint ===
for i, r in enumerate(rewards):
    print(f"Pose {i}: {r} visible voxels")

# === Optional visualization ===
all_visible_voxels = set().union(*voxels_per_pose)
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
colors = []
for v in voxel_indices:
    if tuple(v) in all_visible_voxels:
        colors.append([1.0, 0.0, 0.0])  # red
    else:
        colors.append([0.5, 0.5, 0.5])  # gray
vis_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
o3d.visualization.draw_geometries([vis_pcd])


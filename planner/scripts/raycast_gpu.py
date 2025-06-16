import open3d as o3d
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import time

# === CUDA kernel for ray prep as a string ===
ray_prep_kernel_code = r'''
extern "C" __global__
void ray_prep_kernel(
    const double* poses,         // (n_poses, 3)
    const double* orientations,  // (n_poses, 3, 3)
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
    int* idxs_out,               // (n_poses, n_rays*n_rays, n_steps, 3)
    bool* valid_out              // (n_poses, n_rays*n_rays, n_steps)
) {
    int pose_idx = blockIdx.x;
    int ray_idx = threadIdx.x;

    if (pose_idx >= n_poses || ray_idx >= n_rays * n_rays) return;

    // Compute azimuth and elevation for this ray
    int az_i = ray_idx / n_rays;
    int el_i = ray_idx % n_rays;
    double az = (-fov_deg/2.0) + az_i * (fov_deg/(n_rays-1));
    double el = el_min_deg + el_i * ((el_max_deg-el_min_deg)/(n_rays-1));
    double az_rad = az * 0.017453292519943295; // deg2rad
    double el_rad = el * 0.017453292519943295;

    // Direction in camera frame
    double dx = cos(el_rad) * cos(az_rad);
    double dy = cos(el_rad) * sin(az_rad);
    double dz = sin(el_rad);

    // Apply orientation (3x3 rotation)
    const double* R = orientations + pose_idx*9;
    double dir[3];
    dir[0] = R[0]*dx + R[1]*dy + R[2]*dz;
    dir[1] = R[3]*dx + R[4]*dy + R[5]*dz;
    dir[2] = R[6]*dx + R[7]*dy + R[8]*dz;

    // Camera position (shifted)
    const double* cam = poses + pose_idx*3;

    for (int s = 0; s < n_steps; ++s) {
        double dist = s * resolution;
        double px = cam[0] + dir[0] * dist;
        double py = cam[1] + dir[1] * dist;
        double pz = cam[2] + dir[2] * dist;

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

# === Raycast kernel as before ===
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
ray_prep_kernel = cp.RawKernel(ray_prep_kernel_code, "ray_prep_kernel")

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

def visualize_rays_for_multiple_poses(camera_poses, orientations, rays_all, hits_all):
    ray_lines = []
    n_poses = len(camera_poses)
    n_rays = rays_all.shape[1]
    for i in range(n_poses):
        for j in range(n_rays):
            # Convert voxel indices to world coordinates
            ray_pts = (rays_all[i, j] + min_idx) * voxel_size
            if hits_all[i, j] == 1:
                valid_mask = ~np.all(rays_all[i, j] == 0, axis=1)
                ray_pts = ray_pts[valid_mask]
            ray_line = o3d.geometry.LineSet()
            ray_line.points = o3d.utility.Vector3dVector(ray_pts)
            ray_line.lines = o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(ray_pts) - 1)])
            color = [0, 1, 0] if i == 0 else [0, 0, 1] if i == 1 else [1, 0, 1]
            ray_line.colors = o3d.utility.Vector3dVector([color] * (len(ray_pts) - 1))
            ray_lines.append(ray_line)
    return ray_lines

def raycast_multiple_poses_with_rewards(camera_poses, orientations):
    """
    Uses a custom CUDA kernel to prepare rays (directions, rotations, indices, validity) on GPU,
    then runs the raycast kernel and reward calculation as before.
    """
    global voxel_size, fov_deg, max_range, resolution, n_rays, min_idx, grid_shape, hash_grid
    n_poses = len(camera_poses)
    n_steps = int(max_range / resolution)

    # --- Prepare output arrays ---
    idxs_out = cp.zeros((n_poses, n_rays * n_rays, n_steps, 3), dtype=cp.int32)
    valid_out = cp.zeros((n_poses, n_rays * n_rays, n_steps), dtype=cp.bool_)

    # --- Prepare input arrays ---
    poses_gpu = cp.asarray(camera_poses, dtype=cp.float64)
    orientations_gpu = cp.asarray(np.stack(orientations), dtype=cp.float64)
    min_idx_gpu = cp.asarray(min_idx, dtype=cp.int32)
    grid_shape_gpu = cp.asarray(grid_shape, dtype=cp.int32)

    t0 = time.time()
    # --- Launch the ray prep kernel ---
    ray_prep_kernel(
        (n_poses,), (n_rays * n_rays,),
        (
            poses_gpu,
            orientations_gpu,
            np.int32(n_poses),
            np.int32(n_rays),
            np.int32(n_steps),
            np.float64(fov_deg),
            np.float64(-45),  # el_min_deg
            np.float64(5),    # el_max_deg
            np.float64(max_range),
            np.float64(resolution),
            np.float64(voxel_size),
            min_idx_gpu,
            grid_shape_gpu,
            idxs_out,
            valid_out
        )
    )
    cp.cuda.Device().synchronize()
    t1 = time.time()
    print(f"Ray prep (CUDA) for {n_poses} poses took: {t1 - t0:.4f} s")

    # --- Flatten for raycast kernel ---
    idxs_flat = idxs_out.reshape(-1, 3).ravel()
    valid_flat = valid_out.ravel()
    hash_flat = hash_grid.ravel()
    n_total_rays = idxs_out.shape[0] * idxs_out.shape[1]
    n_steps = idxs_out.shape[2]

    visible_hits = cp.full((n_total_rays, 3), -1, dtype=cp.int32)
    hit_flags = cp.zeros((n_total_rays,), dtype=cp.int32)

    # --- Raycast kernel ---
    grid = (n_total_rays + 255) // 256
    block = 256
    t2 = time.time()
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
    t3 = time.time()
    print(f"Batched raycasting for {n_poses} poses took: {t3 - t2:.4f} s")

    # --- Reward calculation: CuPy unique (GPU) ---
    t4 = time.time()
    visible_hits = visible_hits.reshape(n_poses, -1, 3)
    hit_flags = hit_flags.reshape(n_poses, -1)
    flat_hits = (
        visible_hits[:, :, 0] * (grid_shape[1] * grid_shape[2]) +
        visible_hits[:, :, 1] * grid_shape[2] +
        visible_hits[:, :, 2]
    )
    flat_hits = cp.where(hit_flags == 1, flat_hits, -1)
    rewards = []
    per_view_visible_voxels = []
    for i in range(n_poses):
        valid_hits = flat_hits[i][flat_hits[i] != -1]
        unique_hits = cp.unique(valid_hits)
        rewards.append(int(unique_hits.size))
        voxels_3d = cp.stack([
            unique_hits // (grid_shape[1] * grid_shape[2]),
            (unique_hits % (grid_shape[1] * grid_shape[2])) // grid_shape[2],
            unique_hits % grid_shape[2]
        ], axis=1).get() + min_idx
        per_view_visible_voxels.append(set(map(tuple, voxels_3d)))
    t5 = time.time()
    print(f"Reward calculation (CuPy unique) took: {t5 - t4:.4f} s")

    # For visualization
    # (Optional: If you want to visualize rays, you can reconstruct them from idxs_out and min_idx)
    rays_all = idxs_out.get()  # Now you can use this for visualization
    hits_all = hit_flags.get()

    return per_view_visible_voxels, rewards, rays_all, hits_all

# === Example usage ===
camera_poses = np.array([
    [5, 2.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6],[0, 0.0, 0.6]
])
orientations = [np.eye(3) for _ in range(len(camera_poses))]

voxels_per_pose, rewards, rays_all, hits_all = raycast_multiple_poses_with_rewards(camera_poses, orientations)

# === Print reward per viewpoint ===
for i, r in enumerate(rewards):
    print(f"Pose {i}: {r} visible voxels")

# === Visualization of all rays for all poses in one window ===
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

ray_lines = visualize_rays_for_multiple_poses(camera_poses, orientations, rays_all, hits_all)
o3d.visualization.draw_geometries([vis_pcd] + ray_lines)
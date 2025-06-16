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
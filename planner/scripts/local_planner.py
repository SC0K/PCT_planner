import rospy
import numpy as np
import cupy as cp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from planner_wrapper_coveragev2 import TomogramCoveragePlanner
from config import Config
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import std_msgs.msg
import math
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import open3d as o3d
import cupyx.scipy.ndimage
import time
from std_msgs.msg import Header, Empty
from sensor_msgs.msg import PointField
POINT_FIELDS_XYZI = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1)
]
def GRID_POINTS_XYZI(resolution, dim_x, dim_y):
    index_proto = np.zeros((dim_x * dim_y, 2), dtype=int)
    lx = np.linspace(0, dim_x - 1, dim_x, dtype=int)
    ly = np.linspace(0, dim_y - 1, dim_y, dtype=int)
    ix, iy = np.meshgrid(lx, ly)
    index_proto[:, 0] = ix.flatten()
    index_proto[:, 1] = iy.flatten()

    point_proto = np.zeros((dim_x * dim_y, 4), dtype=np.float32)
    point_proto[:, :2] = index_proto[:, :2].astype(np.float32, copy=True)
    point_proto[:, 0] -= 0.5 * dim_x
    point_proto[:, 1] -= 0.5 * dim_y
    point_proto[:, :2] *= resolution
    point_proto[:, 3] = 1.0

    return index_proto, point_proto

class LidarMappingNode:
    STATE_FOLLOW_PATH = 0
    STATE_COVERAGE = 1
    STATE_RECOVERY = 2

    def __init__(self, planner):
        self.planner = planner
        self.tf_listener = tf.TransformListener()  
        self.current_waypoint_idx = 0
        self.robot_position = None  
        self.persistence_counter = cp.zeros(self.planner.grid_shape, dtype=cp.uint16)
        self.persistence_threshold = 10

        candidate_points_xyz = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points.npy")
        candidate_points_angles = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points_angles.npy")
        self.candidate_path_idx = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/shortest_path_idx.npy")
        self.global_path = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/full_trajectory.npy")
        self.candidate_points_xyz = np.zeros_like(candidate_points_xyz)
        self.candidate_points_angles = np.zeros_like(candidate_points_angles)
        self.hash_grid = self.planner.hash_grid
        for i, idx in enumerate(self.candidate_path_idx):
            self.candidate_points_xyz[i] = candidate_points_xyz[idx] + np.array([0,0,0.6])
            self.candidate_points_angles[i] = candidate_points_angles[idx]    
        
        self.next_candidate_xyz_idx = 0
        self.target_voxels, self.target_voxels_candidates = planner.compute_explored_voxels(self.candidate_points_xyz, self.candidate_points_angles)
        self.scanned_voxels = cp.zeros_like(self.target_voxels, dtype=cp.bool_)

        self.target_voxels_pc_pub = rospy.Publisher("target_voxels_pc", PointCloud2, queue_size=10)
        self.scanned_voxels_pc_pub = rospy.Publisher("scanned_voxels_pc", PointCloud2, queue_size=10)
        self.current_target_voxels_pc_pub = rospy.Publisher("current_target_voxels_pc", PointCloud2, queue_size=10)
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)
        self.added_voxels_pc_pub = rospy.Publisher("added_voxels_pc", PointCloud2, queue_size=10)
        self.tomogram_pub = rospy.Publisher("tomograph", PointCloud2, latch=True, queue_size=1)
        self.remaining_target_voxels_pc_pub = rospy.Publisher("remaining_target_voxels_pc", PointCloud2, queue_size=10)

        rospy.Subscriber("/anymal/pose_in_sim_world", Odometry, self.robot_pose_callback)
        rospy.Subscriber("/current_lidar_voxel_grid", PointCloud2, self.lidar_grid_callback)

        self.publish_target_voxels()
        self.start_time = None
        self.total_time = None

        self.last_robot_pose = None
        self.stuck_counter = 0
        self.stuck_threshold = 10
        self.use_dilation = False
        self.use_dilation_new_obstacles = True
        self.dilation_size = 1   

        # State machine
        self.state = self.STATE_FOLLOW_PATH
        self.coverage_target = None
        self.coverage_orientation = None
        self.coverage_angle = None
        self.coverage_shift_distance = 1.5
        self.coverage_threshold = 10.0  # percent
        self.recovery_start_time = None
        self.recovery_timeout = 10  # seconds
        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(planner.resolution, planner.map_dim[0], planner.map_dim[1])

    def robot_pose_callback(self, odom_msg):
        self.robot_position = (
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        )
        self.robot_orientation = (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        )

    def is_robot_stuck(self):
        current_pose = (self.robot_position, self.robot_orientation)
        if self.last_robot_pose is not None:
            pos_diff = np.linalg.norm(np.array(self.robot_position) - np.array(self.last_robot_pose[0]))
            ori_diff = np.linalg.norm(np.array(self.robot_orientation) - np.array(self.last_robot_pose[1]))
            if pos_diff < 1e-3 and ori_diff < 1e-3:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_robot_pose = current_pose
        return self.stuck_counter >= self.stuck_threshold

    def quaternion_distance(self, q1, q2):
        """Return the angle (in radians) between two quaternions."""
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(dot)

    def calculate_orientation(self, current_waypoint, next_waypoint):
        dx = next_waypoint[0] - current_waypoint[0]
        dy = next_waypoint[1] - current_waypoint[1]
        yaw = math.atan2(dy, dx)
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw)
        return Quaternion(*quaternion)

    def is_pose_reachable(self, position, min_distance=0.8):
        """
        Check if the given position is at least min_distance away from any occupied voxel (obstacle).
        Only checks xy-plane for simplicity.
        """
        idx = np.round((np.array(position[:3]) / self.planner.voxel_size) - self.planner.min_idx).astype(int)
        x, y, z = idx
        if (x < 0 or y < 0 or z < 0 or
            x >= self.planner.grid_shape[0] or
            y >= self.planner.grid_shape[1] or
            z >= self.planner.grid_shape[2]):
            return False

        radius_vox = int(np.ceil(min_distance / self.planner.voxel_size))
        x_min = max(0, x - radius_vox)
        x_max = min(self.planner.grid_shape[0], x + radius_vox + 1)
        y_min = max(0, y - radius_vox)
        y_max = min(self.planner.grid_shape[1], y + radius_vox + 1)
        z_min = max(0, z - 1)
        z_max = min(self.planner.grid_shape[2], z + 2)

        region = self.hash_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        if cp.any(region):
            return False
        return True

    def step(self):
        if self.state == self.STATE_FOLLOW_PATH:
            self.handle_follow_path()
        elif self.state == self.STATE_COVERAGE:
            self.handle_coverage()
        elif self.state == self.STATE_RECOVERY:
            self.handle_recovery()

    def handle_follow_path(self):
        if self.current_waypoint_idx >= len(self.global_path):
            rospy.loginfo("Path completed.")
            if self.start_time is not None and self.total_time is None:
                self.total_time = time.time() - self.start_time
                rospy.loginfo(f"Total time taken to traverse the path: {self.total_time:.2f} seconds")
            percentage_scanned = cp.sum(self.scanned_voxels & self.target_voxels).get() / cp.sum(self.target_voxels).get() * 100
            rospy.loginfo(f"Percentage of target voxels scanned: {percentage_scanned:.2f}%")
            return

        if self.robot_position is None:
            rospy.logwarn("Robot position not available. Skipping waypoint publishing.")
            return

        current_waypoint = self.global_path[self.current_waypoint_idx]
        robot_pos = np.array(self.robot_position)
        waypoint_pos = np.array(current_waypoint)
        dist_to_waypoint = np.linalg.norm(robot_pos - waypoint_pos)

        distance_threshold = 1.0  # meters
        if dist_to_waypoint > distance_threshold:
            if self.current_waypoint_idx + 1 < len(self.global_path):
                next_waypoint = self.global_path[self.current_waypoint_idx + 1]
                orientation = self.calculate_orientation(current_waypoint, next_waypoint)
            else:
                orientation = Quaternion(0, 0, 0, 1)
            goal_msg = PoseStamped()
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.header.frame_id = "map"
            goal_msg.pose.position.x = current_waypoint[0]
            goal_msg.pose.position.y = current_waypoint[1]
            goal_msg.pose.position.z = current_waypoint[2]
            goal_msg.pose.orientation = orientation
            self.goal_pub.publish(goal_msg)
            return

        # Increment candidate index if this waypoint matches candidate position
        if (self.next_candidate_xyz_idx < len(self.candidate_points_xyz) and
            np.allclose(current_waypoint[:3], self.candidate_points_xyz[self.next_candidate_xyz_idx], atol=0.5)):
            self.next_candidate_xyz_idx += 1

        self.current_waypoint_idx += 1

        if self.next_candidate_xyz_idx < len(self.candidate_points_xyz)+1 and self.current_waypoint_idx < len(self.global_path):
            candidate_point = self.candidate_points_xyz[self.next_candidate_xyz_idx-1]
            next_waypoint = self.global_path[self.current_waypoint_idx]
            distance_to_next_candidate = np.linalg.norm(np.array(next_waypoint[:2]) - np.array(candidate_point[:2]))
            self.publish_current_target_voxels()
            if distance_to_next_candidate < 0.5:
                scanned_target_voxels_local = self.scanned_voxels & self.target_voxels_candidates[self.next_candidate_xyz_idx-1]
                unscanned_voxels_local = self.target_voxels_candidates[self.next_candidate_xyz_idx-1] & ~scanned_target_voxels_local
                if cp.any(unscanned_voxels_local):
                    total_voxels_local = cp.sum(self.target_voxels_candidates[self.next_candidate_xyz_idx-1])
                    unscanned_voxels_count = cp.sum(unscanned_voxels_local)
                    unscanned_percentage = (unscanned_voxels_count / total_voxels_local) * 100
                    if unscanned_percentage > 5.0:
                        unscanned_indices = cp.argwhere(unscanned_voxels_local).get()
                        unscanned_patch_size = self.find_largest_continuous_patch(unscanned_indices)
                        if unscanned_patch_size > 10:
                            unscanned_center = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
                            shift_distance = self.coverage_shift_distance
                            rotation_matrix = tf_trans.quaternion_matrix(tf_trans.quaternion_from_euler(0, 0, math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])))[:3, :3]
                            shift_vector = rotation_matrix @ np.array([-shift_distance, 0, 0])
                            unscanned_center += shift_vector
                            angle = math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])
                            orientation = tf_trans.quaternion_from_euler(0, 0, angle)
                            orientation = Quaternion(*orientation)
                            robot_pos = np.array(self.robot_position)
                            unscanned_center[2] = robot_pos[2]
                            self.coverage_target = unscanned_center
                            self.coverage_orientation = orientation
                            self.coverage_angle = angle
                            self.state = self.STATE_COVERAGE
                            return
                        
    def handle_coverage(self):
    
        # Always recompute the unscanned region and plan to its center
        scanned_target_voxels_local = self.scanned_voxels & self.target_voxels_candidates[self.next_candidate_xyz_idx-1]
        unscanned_voxels_local = self.target_voxels_candidates[self.next_candidate_xyz_idx-1] & ~scanned_target_voxels_local
        if not cp.any(unscanned_voxels_local):
            rospy.loginfo("Unscanned area has been fully scanned.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        unscanned_indices = cp.argwhere(unscanned_voxels_local).get()
        if unscanned_indices.shape[0] == 0:
            rospy.loginfo("No unscanned voxels found in candidate region.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        # Compute the center of the largest unscanned patch (x, y), but set z to the lowest point in the patch
        unscanned_center = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
        min_z_idx = np.min(unscanned_indices[:, 2])
        unscanned_center[2] = (min_z_idx + self.planner.min_idx[2] + 0.5) * self.planner.voxel_size  # project to ground
    
        # Replan path at every step using the latest tomograph
        robot_pos = np.array(self.robot_position)
        # planned_traj = self.planner.plan(robot_pos, unscanned_center)
        planned_traj = None
        if planned_traj is None or len(planned_traj) < 2:
            rospy.logwarn("No valid path found to unscanned region. Skipping to next candidate.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        # Send the next waypoint as the goal
        next_goal = planned_traj[1]
        if len(planned_traj) > 2:
            next_next_goal = planned_traj[2]
            orientation = self.calculate_orientation(next_goal, next_next_goal)
        else:
            orientation = Quaternion(0, 0, 0, 1)
    
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = next_goal[0]
        goal_msg.pose.position.y = next_goal[1]
        goal_msg.pose.position.z = next_goal[2]
        goal_msg.pose.orientation = orientation
        self.goal_pub.publish(goal_msg)
        
    def add_new_obstacles_to_tomograph(self, voxel_list):
        """
        Add newly detected obstacle voxels to the tomograph and update the planner.
        """
        if not voxel_list:
            return False
    
        # Convert voxel indices to world coordinates
        indices = np.array(voxel_list)
        world_points = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
    
        # Add these points to the tomograph (assuming your planner has such a method)
        self.planner.add_obstacle_points(world_points)
    
        rospy.loginfo(f"Added {len(world_points)} new obstacle points to tomograph.")
        self.publishTomogram(self.planner.elev_g, self.planner.trav)
        return True
        
    def publishTomogram(self, elev_g, trav):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        n_slice = elev_g.shape[0]
        vis_g = elev_g.copy()
        vis_t = trav.copy() 
        print("vis_g shape: ", vis_g.shape)
        print("vis_t shape: ", vis_t.shape)
        
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.planner.center

        global_points = None
        for i in range(n_slice - 1):
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.planner.slice_dh
            vis_g[i, mask_h] = np.nan
            vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)

    def handle_recovery(self):
        if self.recovery_start_time is not None and (time.time() - self.recovery_start_time) > self.recovery_timeout:
            rospy.logwarn("Recovery timeout. Skipping to next candidate/waypoint.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.recovery_start_time = None
            return

    def find_largest_continuous_patch(self, unscanned_indices):
        from scipy.ndimage import label
        unscanned_grid = np.zeros(self.planner.grid_shape, dtype=bool)
        for idx in unscanned_indices:
            unscanned_grid[tuple(idx)] = True
        labeled_grid, num_features = label(unscanned_grid)
        patch_sizes = np.bincount(labeled_grid.ravel())[1:]
        return np.max(patch_sizes) if len(patch_sizes) > 0 else 0

    def publish_target_voxels(self):
        indices = cp.argwhere(self.target_voxels).get()
        if indices.shape[0] == 0:
            return
        coords = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
        points = coords.astype(np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.target_voxels_pc_pub.publish(pc2_msg)
    
    def publish_current_target_voxels(self):
        indices = cp.argwhere(self.target_voxels_candidates[self.next_candidate_xyz_idx-1]).get()
        if indices.shape[0] == 0:
            return
        coords = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
        points = coords.astype(np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.current_target_voxels_pc_pub.publish(pc2_msg)
    
    def publish_added_voxels(self):
        if not hasattr(self, 'added_voxels') or len(self.added_voxels) == 0:
            return
        indices = np.array(self.added_voxels)
        coords = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
        points = coords.astype(np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.added_voxels_pc_pub.publish(pc2_msg)

    def publish_scanned_voxels(self):
        tolerance_voxels = 0
        structure = cp.ones((2 * tolerance_voxels + 1,) * 3, dtype=cp.bool_)
        dilated_scanned_voxels = cupyx.scipy.ndimage.binary_dilation(self.scanned_voxels, structure=structure)
        indices = cp.argwhere(dilated_scanned_voxels & self.target_voxels).get()
        if indices.shape[0] == 0:
            return
        coords = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
        points = coords.astype(np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.scanned_voxels_pc_pub.publish(pc2_msg)

    def publish_remaining_target_voxels(self):
        """Publish the remaining target voxels that have not been scanned yet."""
        remaining_target_voxels = self.target_voxels & ~self.scanned_voxels
        indices = cp.argwhere(remaining_target_voxels).get()
        if indices.shape[0] == 0:
            return
        coords = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
        points = coords.astype(np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.remaining_target_voxels_pc_pub.publish(pc2_msg)
        
    
    def lidar_grid_callback(self, msg):
        points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
        if points.shape[0] == 0:
            return
        voxel_indices = np.floor(points / self.planner.voxel_size).astype(np.int32) - np.array(self.planner.min_idx)
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < np.array(self.planner.grid_shape)), axis=1)
        valid_voxel_indices = voxel_indices[valid_mask]
        if valid_voxel_indices.shape[0] == 0:
            return
    
        x_idx = valid_voxel_indices[:, 0]
        y_idx = valid_voxel_indices[:, 1]
        z_idx = valid_voxel_indices[:, 2]
        self.scanned_voxels[x_idx, y_idx, z_idx] = True
    
        self.persistence_counter[x_idx, y_idx, z_idx] += 1
        just_added_mask = (self.persistence_counter[x_idx, y_idx, z_idx] == self.persistence_threshold)
        just_added_mask = cp.asnumpy(just_added_mask)
        just_added_indices = valid_voxel_indices[just_added_mask]
    
        if not hasattr(self, 'added_voxels'):
            self.added_voxels = []
    
        # Track only new voxels for this callback
        new_voxels = []
    
        # --- Dilation control for target voxels ---
        if self.use_dilation_new_obstacles:
            tolerance_voxels = self.dilation_size
            structure = cp.ones((2 * tolerance_voxels + 1,) * 3, dtype=cp.bool_)
            dilated_target_voxels = cupyx.scipy.ndimage.binary_dilation(self.hash_grid, structure=structure)
        else:
            dilated_target_voxels = self.hash_grid
    
        if just_added_indices.shape[0] > 0:
            just_added_indices_cp = cp.array(just_added_indices)
            tx = just_added_indices_cp[:, 0]
            ty = just_added_indices_cp[:, 1]
            tz = just_added_indices_cp[:, 2]
            not_in_dilated_target_mask = ~(dilated_target_voxels[tx, ty, tz])
            filtered_indices = just_added_indices[cp.asnumpy(not_in_dilated_target_mask)]
            for idx in filtered_indices:
                idx_tuple = tuple(idx)
                if idx_tuple not in self.added_voxels:
                    self.added_voxels.append(idx_tuple)
                    new_voxels.append(idx_tuple)  # Only new in this callback
    
        # Only update tomograph if there are new voxels
        if len(new_voxels) > 0:
            self.add_new_obstacles_to_tomograph(new_voxels)
    
        # --- Dilation control for scanned voxels ---
        if self.use_dilation:
            tolerance_voxels = self.dilation_size
            structure = cp.ones((2 * tolerance_voxels + 1,) * 3, dtype=cp.bool_)
            dilated_scanned_voxels = cupyx.scipy.ndimage.binary_dilation(self.scanned_voxels, structure=structure)
        else:
            dilated_scanned_voxels = self.scanned_voxels
    
        scanned_target_voxels = dilated_scanned_voxels & self.target_voxels
        remaining_target_voxels = self.target_voxels & ~scanned_target_voxels
    
        rospy.loginfo(f"Scanned target voxels: {cp.sum(scanned_target_voxels).get()}")
        rospy.loginfo(f"Remaining target voxels: {cp.sum(remaining_target_voxels).get()}")
    
        self.publish_scanned_voxels()
        self.publish_added_voxels()
        self.publish_remaining_target_voxels()

    def follow_path(self):
        self.start_time = time.time()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("local_planner_node", anonymous=True)
    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)
    planner.loadTomogram("experiments/2F_2*1")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd", 0.2)
    node = LidarMappingNode(planner)
    node.follow_path()
    rospy.spin()
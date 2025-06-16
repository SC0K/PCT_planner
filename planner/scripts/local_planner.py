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
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped
import std_msgs.msg
import math
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import open3d as o3d
import cupyx.scipy.ndimage
import time
from std_msgs.msg import Header, Empty
from sensor_msgs.msg import PointField
from std_msgs.msg import Int32MultiArray
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
    STATE_CRITICAL_POINT = 3

    def __init__(self, planner):
        self.planner = planner
        self.tf_listener = tf.TransformListener()  
        self.current_waypoint_idx = 0
        self.robot_position = None  
        self.persistence_counter = cp.zeros(self.planner.grid_shape, dtype=cp.uint16)
        self.persistence_threshold = 10
        self.added_voxels = []
        self.new_voxel_buffer = []
        self.critical_points = []
        self.added_voxels_projected_set = set()
        self.use_dilation = False
        self.tomo_update_flag = False
        self.replanning = False

        self.process_voxel_timer = rospy.Timer(rospy.Duration(1), self.find_critical_points)
        self.update_global_path_timer = rospy.Timer(rospy.Duration(3), self.replan_global_path)

        candidate_points_xyz = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points.npy")
        candidate_points_angles = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points_angles.npy")
        candidate_points_idx = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points_idx.npy")
        self.candidate_path_idx = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/shortest_path_idx.npy")
        self.global_path = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/full_trajectory.npy")
        self.segment_path = np.load(
            "/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/segment_trajectory.npy",
            allow_pickle=True
        )
        if isinstance(self.segment_path, np.ndarray) and self.segment_path.dtype == object and self.segment_path.shape == ():
            self.segment_path = self.segment_path.item()
        self.candidate_points_xyz = np.zeros_like(candidate_points_xyz)
        self.candidate_points_angles = np.zeros_like(candidate_points_angles)
        self.candidate_points_idx = np.zeros_like(candidate_points_idx, dtype=np.int32)
        self.hash_grid = self.planner.hash_grid
        for i, idx in enumerate(self.candidate_path_idx):
            self.candidate_points_xyz[i] = candidate_points_xyz[idx] + np.array([0,0,0.5])
            self.candidate_points_angles[i] = candidate_points_angles[idx]    
            self.candidate_points_idx[i] = candidate_points_idx[idx]
        
        self.next_candidate_xyz_idx = 0
        self.target_voxels, self.target_voxels_candidates = planner.compute_explored_voxels(self.candidate_points_xyz, self.candidate_points_angles)
        self.scanned_voxels = cp.zeros_like(self.target_voxels, dtype=cp.bool_)

        self.target_voxels_pc_pub = rospy.Publisher("target_voxels_pc", PointCloud2, queue_size=10)
        self.critical_points_pub = rospy.Publisher("critical_points", PointCloud2, queue_size=10)
        self.scanned_voxels_pc_pub = rospy.Publisher("scanned_voxels_pc", PointCloud2, queue_size=10)
        self.current_target_voxels_pc_pub = rospy.Publisher("current_target_voxels_pc", PointCloud2, queue_size=10)
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)
        self.added_voxels_pc_pub = rospy.Publisher("added_voxels_pc", PointCloud2, queue_size=10)
        self.tomogram_pub = rospy.Publisher("tomograph", PointCloud2, latch=True, queue_size=1)
        self.remaining_target_voxels_pc_pub = rospy.Publisher("remaining_target_voxels_pc", PointCloud2, queue_size=10)
        self.replanned_path_pub = rospy.Publisher("/replanned_coverage_path", Path, queue_size=10)
        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=10)

        self.unscanned_center_pub = rospy.Publisher("/unscanned_center", PointStamped, queue_size=1)
        

        rospy.Subscriber("/anymal/pose_in_sim_world", Odometry, self.robot_pose_callback)
        # rospy.Subscriber("/current_lidar_voxel_grid", PointCloud2, self.lidar_grid_callback)
        rospy.Subscriber("/current_lidar_voxel_indices", Int32MultiArray, self.lidar_grid_callback)
        rospy.Subscriber("/current_lidar_new_voxel_indices", Int32MultiArray, self.lidar_new_voxel_indices_callback)


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
        rospy.loginfo("Current state: %s", self.state)
        rospy.loginfo("Number of critical points: %d", len(self.critical_points))
        if self.critical_points or self.state == self.STATE_CRITICAL_POINT:
            self.state = self.STATE_CRITICAL_POINT
            rospy.loginfo("######### going to handle critical points #########")
            self.handle_critical_points()
        if self.state == self.STATE_FOLLOW_PATH:
            self.handle_follow_path()
        elif self.state == self.STATE_COVERAGE:
            self.handle_coverage()
        elif self.state == self.STATE_RECOVERY:
            self.handle_recovery()
    def handle_critical_points(self):
        if self.critical_points:
            critical_points_xyz = (np.array(self.critical_points) + self.planner.min_idx + 0.5) * self.planner.voxel_size
        else:
            self.state = self.STATE_FOLLOW_PATH
            return
    
        robot_pos = np.array(self.robot_position)
        # Find the closest critical point
        dists = np.linalg.norm(critical_points_xyz - robot_pos, axis=1)
        idx = np.argmin(dists)
        target_cp = critical_points_xyz[idx]
    
        # Publish the critical point as the goal (use robot's current z for safety)
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = target_cp[0]
        goal_msg.pose.position.y = target_cp[1]
        goal_msg.pose.position.z = robot_pos[2]
        # Optionally, face the critical point
        direction = target_cp[:2] - robot_pos[:2]
        yaw = math.atan2(direction[1], direction[0])
        orientation = tf_trans.quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.orientation = Quaternion(*orientation)
        self.goal_pub.publish(goal_msg)
    
        # --- Select local patch of projected added voxels around the critical point ---
        cp_idx = self.critical_points[idx]
        patch_radius_vox = 2/ self.planner.voxel_size  # search radius in voxels
        patch_voxels = []
        for v in self.added_voxels_projected_set:
            if abs(v[0] - cp_idx[0]) <= patch_radius_vox and abs(v[1] - cp_idx[1]) <= patch_radius_vox and abs(v[2] - cp_idx[2]) <= 1:
                patch_voxels.append(v)
        if len(patch_voxels) < 3:
            # Fallback: just use the critical point itself
            patch_coords = np.array([target_cp[:2]])
        else:
            patch_coords = (np.array(patch_voxels)[:, :2] + self.planner.min_idx[:2] + 0.5) * self.planner.voxel_size

        # --- Fit ellipse (PCA) in xy-plane ---
        center = target_cp[:2]
        centered = patch_coords - center
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        minor_axis = eigvecs[:, np.argmin(eigvals)]
        r_major = max(np.sqrt(eigvals.max()), 1.5)  # at least 1m from center
        r_minor = max(np.sqrt(eigvals.min()), 1.5)  # at least 1m from center
        
        # --- Circling the centroid: select viewpoints around the ellipse ---
        if not hasattr(self, 'ellipse_view_idx'):
            self.ellipse_view_idx = 0
        
        num_views = 100  # e.g., 15 degrees per step
        theta = 2 * np.pi * self.ellipse_view_idx / num_views
        theta2 = -theta 
        view_dir = np.array([np.cos(theta), np.sin(theta)])
        view_dir2 = np.array([np.cos(theta2), np.sin(theta2)])
        viewpoint_xy = center + r_major * view_dir[0] * major_axis + r_minor * view_dir[1] * minor_axis
        viewpoint_xy2 = center + r_major * view_dir2[0] * major_axis + r_minor * view_dir2[1] * minor_axis
        if np.linalg.norm(viewpoint_xy - patch_coords) < np.linalg.norm(viewpoint_xy2 - patch_coords):
            viewpoint_xy_final = viewpoint_xy2
        else:
            viewpoint_xy_final = viewpoint_xy
        viewpoint = np.array([viewpoint_xy_final[0], viewpoint_xy_final[1], robot_pos[2]])
        
        # Publish as goal, facing the critical point
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = viewpoint[0]
        goal_msg.pose.position.y = viewpoint[1]
        goal_msg.pose.position.z = viewpoint[2]
        direction = target_cp[:2] - viewpoint_xy_final
        yaw = math.atan2(direction[1], direction[0])
        orientation = tf_trans.quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.orientation = Quaternion(*orientation)
        self.goal_pub.publish(goal_msg)
        
        # After reaching the viewpoint, increment the index for the next call
        if np.linalg.norm(robot_pos[:2] - viewpoint_xy_final) < 0.5:
            self.ellipse_view_idx = (self.ellipse_view_idx + 1) % num_views
        
        # --- Visualize the ellipse in RViz ---
        ellipse_marker = Marker()
        ellipse_marker.header.frame_id = "map"
        ellipse_marker.header.stamp = rospy.Time.now()
        ellipse_marker.ns = "critical_point_ellipse"
        ellipse_marker.id = 0
        ellipse_marker.type = Marker.LINE_STRIP
        ellipse_marker.action = Marker.ADD
        ellipse_marker.scale.x = 0.05  # line width
        ellipse_marker.color.r = 1.0
        ellipse_marker.color.g = 0.5
        ellipse_marker.color.b = 0.0
        ellipse_marker.color.a = 1.0
        
        # Generate ellipse points in xy-plane
        num_points = 40
        theta = np.linspace(0, 2 * np.pi, num_points)
        ellipse_points = []
        for t in theta:
            pt = center + r_major * np.cos(t) * major_axis + r_minor * np.sin(t) * minor_axis
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = robot_pos[2]
            ellipse_points.append(p)
        ellipse_marker.points = ellipse_points
        
        # Publish the marker
        if not hasattr(self, 'ellipse_pub'):
            self.ellipse_pub = rospy.Publisher("/critical_point_ellipse", Marker, queue_size=1)
        self.ellipse_pub.publish(ellipse_marker)

        if not self.critical_points:
            self.state = self.STATE_FOLLOW_PATH
            return
        

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
                        if unscanned_patch_size > 5:
                            # unscanned_center = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
                            # shift_distance = self.coverage_shift_distance
                            # rotation_matrix = tf_trans.quaternion_matrix(tf_trans.quaternion_from_euler(0, 0, math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])))[:3, :3]
                            # shift_vector = rotation_matrix @ np.array([-shift_distance, 0, 0])
                            # unscanned_center += shift_vector
                            # angle = math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])
                            # orientation = tf_trans.quaternion_from_euler(0, 0, angle)
                            # orientation = Quaternion(*orientation)
                            # robot_pos = np.array(self.robot_position)
                            # unscanned_center[2] = robot_pos[2]
                            # self.coverage_target = unscanned_center
                            # self.coverage_orientation = orientation
                            # self.coverage_angle = angle
                            self.state = self.STATE_COVERAGE
                            return
    def find_critical_points(self,event):
        # --- Process new voxels in buffer ---
        if self.replanning:
            rospy.loginfo("Replanning in progress, skipping critical point processing.")
            return
        if self.new_voxel_buffer:
            unique_voxels = self.new_voxel_buffer
            self.added_voxels.extend(unique_voxels)
            self.new_voxel_buffer = []

            projected_hash_voxels_buffer = set()
            grid_shape = self.planner.grid_shape
            for idx in unique_voxels:
                x, y, z = idx
                for zz in range(z, -1, -1):
                    if self.hash_grid[x, y, zz]:
                        projected_hash_voxels_buffer.add((x, y, zz))
                        self.added_voxels_projected_set.add((x, y, zz))
                        break  # Only project to the first hash_grid cell below
        
            # === Find critical points among projected_hash_voxels ===
            critical_points = []
            scanned = self.scanned_voxels.get()
            target = self.target_voxels.get()
            remaining_target = target & ~scanned
            projected_list_buffer = list(projected_hash_voxels_buffer)
            # Use all added voxels for the projected set
            
            # 2x2 region directions: (x, y) offsets for each region
            region_offsets = [
                [ (1, 0), (0, 1), (1, 1)],      # +x, +y
                [ (1, 0), (0, -1), (1, -1)],    # +x, -y
                [ (-1, 0), (0, 1), (-1, 1)],    # -x, +y
                [ (-1, 0), (0, -1), (-1, -1)]   # -x, -y
            ]
            
            for idx in projected_list_buffer:
                #  or projected_list_buffer
                x, y, z = idx
                for region in region_offsets:
                    region_indices = []
                    for dx, dy in region:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                            region_indices.append((nx, ny, z))
                    if len(region_indices) < 3:
                        continue
                    # Count projected voxels in this region (using all added voxels)
                    projected_count = sum((ix, iy, iz) in self.added_voxels_projected_set for ix, iy, iz in region_indices) + 1 # +1 for the center voxel
                    if projected_count > 1:
                        continue
                    # Check for both scanned and unscanned target voxels
                    has_scanned = any(scanned[ix, iy, iz] for ix, iy, iz in region_indices)
                    has_unscanned = any(remaining_target[ix, iy, iz] for ix, iy, iz in region_indices)
                    if has_scanned and has_unscanned:
                        critical_points.append(idx)
                        self.critical_points.append(idx)
                        break  # Only need one region to satisfy
            self.prune_invalid_critical_points()

            # === Publish critical points as PointCloud2 for visualization ===
            if len(critical_points) > 0:
                critical_points_xyz = (np.array(self.critical_points) + self.planner.min_idx + 0.5) * self.planner.voxel_size
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "map"
                critical_pc2 = pc2.create_cloud_xyz32(header, critical_points_xyz.astype(np.float32))
                self.critical_points_pub.publish(critical_pc2)         
            
            ## Only update tomograph if there are new voxels
            if len(unique_voxels) > 0:
                self.add_new_obstacles_to_tomograph(unique_voxels)
                self.publishTomogram(self.planner.elev_g, self.planner.trav)
    def prune_invalid_critical_points(self):
        """
        Remove critical points that are no longer valid.
        A critical point is invalid if:
        - It is now scanned, or
        - The 2x2 region around it no longer meets the critical point criteria.
        """
        if not self.critical_points:
            return

        grid_shape = self.planner.grid_shape
        scanned = self.scanned_voxels.get()
        target = self.target_voxels.get()
        remaining_target = target & ~scanned
        projected_set = self.added_voxels_projected_set

        region_offsets = [
            [ (1, 0), (0, 1), (1, 1)],      # +x, +y
            [ (1, 0), (0, -1), (1, -1)],    # +x, -y
            [ (-1, 0), (0, 1), (-1, 1)],    # -x, +y
            [ (-1, 0), (0, -1), (-1, -1)]   # -x, -y
        ]

        valid_critical_points = []
        for idx in self.critical_points:
            x, y, z = idx
            still_critical = False
            for region in region_offsets:
                region_indices = []
                for dx, dy in region:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                        region_indices.append((nx, ny, z))
                if len(region_indices) < 3:
                    continue
                projected_count = sum((ix, iy, iz) in projected_set for ix, iy, iz in region_indices) + 1 # +1 for the center voxel
                if projected_count > 1:
                    continue
                has_scanned = any(scanned[ix, iy, iz] for ix, iy, iz in region_indices)
                has_unscanned = any(remaining_target[ix, iy, iz] for ix, iy, iz in region_indices)
                if has_scanned and has_unscanned:
                    still_critical = True
                    break
            if still_critical:
                valid_critical_points.append(idx)

        self.critical_points = valid_critical_points
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
    
        total_voxels_local = cp.sum(self.target_voxels_candidates[self.next_candidate_xyz_idx-1])
        unscanned_voxels_count = cp.sum(unscanned_voxels_local)
        unscanned_percentage = (unscanned_voxels_count / total_voxels_local) * 100
    
        if unscanned_percentage <= 5.0:
            rospy.loginfo("Unscanned percentage below threshold.")
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
    
        unscanned_patch_size = self.find_largest_continuous_patch(unscanned_indices)
        if unscanned_patch_size <= 5:
            rospy.loginfo("No sufficiently large unscanned patch found.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        # Compute the unscanned center and shift as in handle_follow_path
        unscanned_center_raw = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
        # Compute the normal direction of the unscanned patch
        unscanned_points = unscanned_indices * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
        centroid = np.mean(unscanned_points, axis=0)
        centered = unscanned_points - centroid
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]  # Normal vector (least variance)
        
        shift_distance = 1.5 
        
        # ... after computing normal_xy as before ...
        if abs(normal[2]) > 0.7:
            # Ground/ceiling: use original candidate angle
            angle = math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])
            shift_vector = np.array([math.cos(angle), math.sin(angle), 0]) * -shift_distance
        else:
            # Wall: choose normal_xy direction closest to -candidate angle
            angle = math.radians(self.candidate_points_angles[self.next_candidate_xyz_idx-1])
            candidate_dir = np.array([math.cos(angle), math.sin(angle), 0])
            normal_xy = normal.copy()
            normal_xy[2] = 0
            if np.linalg.norm(normal_xy) > 1e-3:
                normal_xy /= np.linalg.norm(normal_xy)
            else:
                normal_xy = np.array([1, 0, 0])  # fallback
        
            # Compare normal_xy and -candidate_dir, flip if needed
            if np.dot(normal_xy, -candidate_dir) < 0:
                normal_xy = -normal_xy
            shift_vector = normal_xy * shift_distance
        
        unscanned_center = unscanned_center_raw + shift_vector
        robot_pos = np.array(self.robot_position)
        unscanned_center[2] = robot_pos[2]  # Keep the same height as the robot
        unscanned_center_msg = PointStamped()
        unscanned_center_msg.header.stamp = rospy.Time.now()
        unscanned_center_msg.header.frame_id = "map"
        unscanned_center_msg.point.x = unscanned_center[0]
        unscanned_center_msg.point.y = unscanned_center[1]
        unscanned_center_msg.point.z = unscanned_center[2]
        self.unscanned_center_pub.publish(unscanned_center_msg)
        
        # Use online_local_replan to find a traversable goal and plan a path
        # robot_pos = np.array([self.robot_position[1], self.robot_position[0], self.robot_position[2]])
        # Wait until replanning is finished
        while self.replanning:
            rospy.loginfo("Waiting for replanning to finish...")
            rospy.sleep(0.1)  # Sleep for 100ms
        planned_traj = self.planner.online_local_replan(robot_pos, unscanned_center, height_tol=1.5)
        if planned_traj is None or len(planned_traj) < 2:
            rospy.logwarn("No valid path found to unscanned region. Skipping to next candidate.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        # === Publish replanned trajectory for visualization ===
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for pt in planned_traj:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose.pose.orientation = Quaternion(0, 0, 0, 1)
            path_msg.poses.append(pose)
        self.replanned_path_pub.publish(path_msg)
    
        # === Path following logic: always face the center of the unscanned area ===
        dists = np.linalg.norm(planned_traj - robot_pos, axis=1)
        closest_idx = np.argmin(dists)
    
        # If close to the last point, finish coverage
        distance_threshold = 0.2  # meters
        if np.linalg.norm(robot_pos - planned_traj[-1]) < distance_threshold:
            rospy.loginfo("Reached coverage target.")
            self.state = self.STATE_FOLLOW_PATH
            self.current_waypoint_idx += 1
            self.coverage_start_time = None
            return
    
        # Otherwise, send the next point as goal, always facing the unscanned center
        if closest_idx + 3 < len(planned_traj):
            next_goal = planned_traj[closest_idx + 3]
        else:
            next_goal = planned_traj[-1]
    
        # Orientation: always face the center of the unscanned area
        direction = np.array(unscanned_center_raw[:2]) - np.array(next_goal[:2])
        yaw = math.atan2(direction[1], direction[0])
        orientation = tf_trans.quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(*orientation)
    
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
        Also replan the path segment between current and next candidate points if needed.
        """
        if voxel_list is None or len(voxel_list) == 0:
            return False
    
        # Convert voxel indices to world coordinates
        indices = np.array(voxel_list)
        world_points = (indices + self.planner.min_idx + 0.5) * self.planner.voxel_size
    
        # Add these points to the tomograph (assuming your planner has such a method)
        self.planner.add_obstacle_points(world_points)
    
        rospy.loginfo(f"Added {len(world_points)} new obstacle points to tomograph.")
        self.tomo_update_flag = True
        return True
    def replan_global_path(self, event):
        if self.next_candidate_xyz_idx < len(self.candidate_points_xyz) - 1 and self.tomo_update_flag:
            rospy.loginfo("replan_global_path timer fired")
            self.replanning = True
            # Use candidate_points_xyz for candidate positions
            remaining_candidates = self.candidate_points_idx[self.next_candidate_xyz_idx:]
            full_trajectory = []
            segment_trajectories = {}
    
            for i in range(len(remaining_candidates) - 1):
                start_pos = remaining_candidates[i]
                end_pos = remaining_candidates[i + 1]
            
                traj_3d = self.planner.plan_with_idx_online(start_pos, end_pos)
                if traj_3d is not None:
                    full_trajectory.extend(traj_3d)
                    segment_trajectories[(i, i+1)] = np.array(traj_3d)
                else:
                    rospy.logwarn(f"Failed to compute trajectory between {start_pos} and {end_pos}, searching along segment path...")
                    # Use segment_path to reduce computation
                    segment = self.segment_path[(i+1,i+2)]  # segment is a dense array of points
                    stride = 10  # check every 10th point
                    
                    found = False
                    for idx in range(0, len(segment), stride):
                        pt = segment[idx]
                        pt_idx = self.planner.pos2idx_3D_plan(pt)
                        cost = self.planner.trav[pt_idx[0], pt_idx[2], pt_idx[1]]
                        rospy.logwarn(f"Checking sparse point {pt} with idx {pt_idx} and cost {cost:.2f}")
                        if cost < 20:
                            traj_3d_future = self.planner.plan_with_idx_online(start_pos, pt_idx)
                            if traj_3d_future is not None:
                                full_trajectory.extend(traj_3d_future)
                                segment_trajectories[(i, f"segment_{pt}")] = np.array(traj_3d_future)
                                found = True
                                break
                    if not found:
                        rospy.logwarn(f"No traversable path found from candidate {i} to any sparse point in segment.")
            self.replanning = False
            self.global_path = np.array(full_trajectory)
            self.tomo_update_flag = False
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

    def publish_global_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for pt in self.global_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose.pose.orientation = Quaternion(0, 0, 0, 1)
            path_msg.poses.append(pose)
        self.global_path_pub.publish(path_msg)
        
   
    def lidar_grid_callback(self, msg):
        indices = np.array(msg.data, dtype=np.int32).reshape(-1, 3)
        if indices.shape[0] == 0:
            return
    
        # Move indices to GPU for fast assignment
        indices_cp = cp.asarray(indices)
        x_idx, y_idx, z_idx = indices_cp[:, 0], indices_cp[:, 1], indices_cp[:, 2]
        self.scanned_voxels[x_idx, y_idx, z_idx] = True
    
        # # --- Dilation control for scanned voxels ---
        # if self.use_dilation:
        #     tolerance_voxels = self.dilation_size
        #     structure = cp.ones((2 * tolerance_voxels + 1,) * 3, dtype=cp.bool_)
        #     dilated_scanned_voxels = cupyx.scipy.ndimage.binary_dilation(self.scanned_voxels, structure=structure)
        # else:
        #     dilated_scanned_voxels = self.scanned_voxels
    
        # scanned_target_voxels = dilated_scanned_voxels & self.target_voxels
        # remaining_target_voxels = self.target_voxels & ~scanned_target_voxels

    def lidar_new_voxel_indices_callback(self, msg):
        indices = np.array(msg.data, dtype=np.int32).reshape(-1, 3)
        if indices.shape[0] == 0:
            return
    
        # Add to self.added_voxels (avoid duplicates)
            
        new_voxels = []
        for idx in indices:
            idx_tuple = tuple(idx)
            if idx_tuple not in self.added_voxels:
                # #########")
                new_voxels.append(idx_tuple)
        self.new_voxel_buffer.extend(new_voxels)
    
        # # --- Efficient projection of new voxels onto hash grid ---
        # projected_hash_voxels = set()
        # grid_shape = self.planner.grid_shape
        # for idx in new_voxels:
        #     x, y, z = idx
        #     for zz in range(z, -1, -1):
        #         if self.hash_grid[x, y, zz]:
        #             if (x, y, zz) not in projected_hash_voxels:
        #                 projected_hash_voxels.add((x, y, zz))
        #             break  # Only project to the first hash_grid cell below
    
        # # === Find critical points among projected_hash_voxels ===
        # critical_points = []
        # added_voxels_set = set(self.added_voxels)
        # explored_voxels = set(map(tuple, cp.argwhere(self.scanned_voxels).get()))
        # unexplored_voxels = set(
        #     (x, y, z)
        #     for x, y, z in projected_hash_voxels
        #     if (x, y, z) not in explored_voxels
        # )
        # neighbor_offsets = [
        #     (1, 0, 0), (-1, 0, 0),
        #     (0, 1, 0), (0, -1, 0),
        #     (0, 0, 1), (0, 0, -1)
        # ]
        # for idx in projected_hash_voxels:
        #     neighbor_added = 0
        #     has_unexplored = False
        #     has_explored = False
        #     for dx, dy, dz in neighbor_offsets:
        #         nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
        #         if (0 <= nidx[0] < grid_shape[0] and
        #             0 <= nidx[1] < grid_shape[1] and
        #             0 <= nidx[2] < grid_shape[2]):
        #             if nidx in added_voxels_set:
        #                 neighbor_added += 1
        #             if nidx in explored_voxels:
        #                 has_explored = True
        #             if nidx in unexplored_voxels:
        #                 has_unexplored = True
        #     if neighbor_added <= 1 and has_unexplored and has_explored:
        #         critical_points.append(idx)
    
        # # === Publish critical points as PointCloud2 for visualization ===
        # if len(critical_points) > 0:
        #     critical_points_xyz = (np.array(critical_points) + self.planner.min_idx + 0.5) * self.planner.voxel_size
        #     header = std_msgs.msg.Header()
        #     header.stamp = rospy.Time.now()
        #     header.frame_id = "map"
        #     critical_pc2 = pc2.create_cloud_xyz32(header, critical_points_xyz.astype(np.float32))
        #     self.critical_points_pub.publish(critical_pc2)
    
        # self.add_new_obstacles_to_tomograph(indices)
        

    def follow_path(self):
        self.start_time = time.time()
        rate = rospy.Rate(5)    # 10 Hz
        while not rospy.is_shutdown():
            self.publish_scanned_voxels()
            self.publish_remaining_target_voxels()
            self.publish_added_voxels()
            self.publish_global_path()

            self.step()
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("local_planner_node", anonymous=True)
    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)
    planner.loadTomogram("experiments/2F_2*1")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd", 0.1075)
    node = LidarMappingNode(planner)
    node.follow_path()
    rospy.spin()
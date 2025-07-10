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
import sensor_msgs.point_cloud2 as pc2
import math
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import open3d as o3d
import cupyx.scipy.ndimage
import time

class LidarMappingNode:
    def __init__(self, planner):
        self.planner = planner
        self.tf_listener = tf.TransformListener()  
        self.current_waypoint_idx = 0
        self.robot_position = None  
        self.persistence_counter = cp.zeros(self.planner.grid_shape, dtype=cp.uint16)
        self.persistence_threshold = 10  # Number of times a voxel must be seen to be added as structure

        candidate_points_xyz = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points.npy")
        candidate_points_angles = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points_angles.npy")
        self.candidate_path_idx = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/shortest_path_idx.npy")
        self.path_sequence = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/segment_trajectory.npy", allow_pickle=True)
        self.global_path = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/full_trajectory.npy")
        self.candidate_points_xyz = np.zeros_like(candidate_points_xyz)
        self.candidate_points_angles = np.zeros_like(candidate_points_angles)
        self.hash_grid = self.planner.hash_grid
        for i, idx in enumerate(self.candidate_path_idx):
            self.candidate_points_xyz[i] = candidate_points_xyz[idx] + np.array([0,0,0.6])
            self.candidate_points_angles[i] = candidate_points_angles[idx]    
        
        self.latest_valid_voxel_indices = None
        # rospy.Timer(rospy.Duration(0.5), self.process_lidar_voxel_grid)  # 10 Hz processing
        
        self.current_candidate_xyz_idx = 0

        self.target_voxels, self.target_voxels_candidates = planner.compute_explored_voxels(self.candidate_points_xyz, self.candidate_points_angles)
        self.scanned_voxels = cp.zeros_like(self.target_voxels, dtype=cp.bool_)

        self.target_voxels_pc_pub = rospy.Publisher("target_voxels_pc", PointCloud2, queue_size=10)
        self.scanned_voxels_pc_pub = rospy.Publisher("scanned_voxels_pc", PointCloud2, queue_size=10)
        self.current_target_voxels_pc_pub = rospy.Publisher("current_target_voxels_pc", PointCloud2, queue_size=10)

        # rospy.Subscriber("/point_cloud_filter/lidar_depth_camera/point_cloud_filtered", PointCloud2, self.lidar_callback)
        # rospy.Subscriber("/depth_camera_front_upper/point_cloud_self_filtered", PointCloud2, self.lidar_callback)
        rospy.Subscriber("/current_lidar_voxel_grid", PointCloud2, self.lidar_grid_callback)

        self.publish_target_voxels()
    
        self.path_pub = rospy.Publisher("/path_ahead", Marker, queue_size=10)

        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)
        self.added_voxels_pc_pub = rospy.Publisher("added_voxels_pc", PointCloud2, queue_size=10)

        rospy.Subscriber("/anymal/pose_in_sim_world", Odometry, self.robot_pose_callback)
        self.start_time = None
        self.total_time = None

        self.last_robot_pose = None
        self.stuck_counter = 0
        self.stuck_threshold = 10

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

    def calculate_orientation(self, current_waypoint, next_waypoint):
        dx = next_waypoint[0] - current_waypoint[0]
        dy = next_waypoint[1] - current_waypoint[1]
        yaw = math.atan2(dy, dx)  # Calculate yaw angle
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw)  # Convert yaw to quaternion
        return Quaternion(*quaternion)

    def publish_next_waypoint(self, distance_threshold=1.0):
        if self.current_waypoint_idx >= len(self.global_path):
            rospy.loginfo("Path completed.")
            if self.start_time is not None:
                if self.total_time is None:
                    self.total_time = time.time() - self.start_time
                    rospy.loginfo(f"Total time taken to traverse the path: {self.total_time:.2f} seconds")
            percentage_scanned = cp.sum(self.scanned_voxels & self.target_voxels).get() / cp.sum(self.target_voxels).get() * 100
            rospy.loginfo(f"Percentage of target voxels scanned: {percentage_scanned:.2f}%")
            return
    
        if self.robot_position is None:
            rospy.logwarn("Robot position not available. Skipping waypoint publishing.")
            return
    
        current_waypoint = self.global_path[self.current_waypoint_idx]
    
        try:
            distance = math.sqrt(
                (current_waypoint[0] - self.robot_position[0]) ** 2 +
                (current_waypoint[1] - self.robot_position[1]) ** 2 +
                (current_waypoint[2] - self.robot_position[2]) ** 2
            )
        except Exception as e:
            rospy.logerr(f"Error calculating distance to waypoint: {e}")
            return
    
        if distance > distance_threshold:
            rospy.logwarn(f"Next waypoint is too far ({distance:.2f} meters). Waiting...")
            return
    
        if self.current_waypoint_idx + 1 < len(self.global_path):
            next_waypoint = self.global_path[self.current_waypoint_idx + 1]
            orientation = self.calculate_orientation(current_waypoint, next_waypoint)
        else:
            next_waypoint = None
            orientation = Quaternion(0, 0, 0, 1)
    
        rospy.loginfo(f"Publishing waypoint: {current_waypoint}")
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = current_waypoint[0]
        goal_msg.pose.position.y = current_waypoint[1]
        goal_msg.pose.position.z = current_waypoint[2]
        goal_msg.pose.orientation = orientation
    
        self.goal_pub.publish(goal_msg)
    
        self.current_waypoint_idx += 1

    def find_largest_continuous_patch(self, unscanned_indices):
        from scipy.ndimage import label

        unscanned_grid = np.zeros(self.planner.grid_shape, dtype=bool)
        for idx in unscanned_indices:
            unscanned_grid[tuple(idx)] = True

        labeled_grid, num_features = label(unscanned_grid)

        patch_sizes = np.bincount(labeled_grid.ravel())[1:]  # Exclude the background (label 0)

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
        indices = cp.argwhere(self.target_voxels_candidates[self.current_candidate_xyz_idx]).get()
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
        tolerance_voxels = 0  # Adjust as needed
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
    
        # --- Vectorized persistence update ---
        self.persistence_counter[x_idx, y_idx, z_idx] += 1
    
        just_added_mask = (self.persistence_counter[x_idx, y_idx, z_idx] == self.persistence_threshold)
        just_added_mask = cp.asnumpy(just_added_mask)  
        just_added_indices = valid_voxel_indices[just_added_mask]
    
        if not hasattr(self, 'added_voxels'):
            self.added_voxels = []
    
        tolerance_voxels = 1  # Set to 1 for 6-neighborhood, 2 for wider, etc.
        structure = cp.ones((2 * tolerance_voxels + 1,) * 3, dtype=cp.bool_)
        dilated_target_voxels = cupyx.scipy.ndimage.binary_dilation(self.hash_grid, structure=structure)

        if just_added_indices.shape[0] > 0:
            just_added_indices_cp = cp.array(just_added_indices)
            tx = just_added_indices_cp[:, 0]
            ty = just_added_indices_cp[:, 1]
            tz = just_added_indices_cp[:, 2]
            not_in_dilated_target_mask = ~(dilated_target_voxels[tx, ty, tz])
            filtered_indices = just_added_indices[cp.asnumpy(not_in_dilated_target_mask)]
            for idx in filtered_indices:
                idx_tuple = tuple(idx)
                self.added_voxels.append(idx_tuple)
        
        dilated_scanned_voxels = cupyx.scipy.ndimage.binary_dilation(self.scanned_voxels, structure=structure)
        scanned_target_voxels = dilated_scanned_voxels & self.target_voxels
        remaining_target_voxels = self.target_voxels & ~scanned_target_voxels
    
        rospy.loginfo(f"Scanned target voxels: {cp.sum(scanned_target_voxels).get()}")
        rospy.loginfo(f"Remaining target voxels: {cp.sum(remaining_target_voxels).get()}")
    
        self.publish_scanned_voxels()
        self.publish_added_voxels()

    def follow_path(self):
        self.start_time = time.time()
        rate = rospy.Rate(5)  # 4 Hz
        while not rospy.is_shutdown():
            self.publish_next_waypoint(distance_threshold=1.0)  # Check distance before publishing
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("path_follower_node", anonymous=True)

    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)

    planner.loadTomogram("experiments/2F_2*1")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd", 0.1)

    node = LidarMappingNode(planner)
    node.follow_path()  # Start following the path
    rospy.spin()
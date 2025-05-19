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
import math
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import open3d as o3d
class LidarMappingNode:
    def __init__(self, planner):
        self.planner = planner
        self.tf_listener = tf.TransformListener()  

        candidate_points_xyz = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points.npy")
        candidate_points_angles = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/reachable_sampled_points_angles.npy")
        self.candidate_path_idx = np.load("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/planner/scripts/shortest_path_idx.npy")
        self.candidate_points_xyz = np.zeros_like(candidate_points_xyz)
        self.candidate_points_angles = np.zeros_like(candidate_points_angles)
        for i,idx in enumerate(self.candidate_path_idx):
            self.candidate_points_xyz[i] = candidate_points_xyz[idx]
            self.candidate_points_angles[i] = candidate_points_angles[idx] + 90.0 # +90 degree to align with the coordinate of the map (building_2F_4R)     

        self.current_candidate_xyz_idx = 0

        # Use CuPy arrays for target_voxels and scanned_voxels
        self.target_voxels, self.target_voxels_candidates = planner.compute_explored_voxels(self.candidate_points_xyz, self.candidate_points_angles)
        self.scanned_voxels = cp.zeros_like(self.target_voxels, dtype=cp.bool_)

        self.target_voxels_pub = rospy.Publisher("target_voxels", MarkerArray, queue_size=10)
        self.scanned_voxels_pub = rospy.Publisher("scanned_voxels", MarkerArray, queue_size=10)
        self.current_target_voxels_pub = rospy.Publisher("current_target_voxels", MarkerArray, queue_size=10)

        # rospy.Subscriber("/point_cloud_filter/lidar_depth_camera/point_cloud_filtered", PointCloud2, self.lidar_callback)
        rospy.Subscriber("/depth_camera_front_upper/depth/color/points", PointCloud2, self.lidar_callback)

        self.publish_target_voxels()
    
        rospy.Subscriber("/pct_path", Path, self.path_callback)
        self.path_pub = rospy.Publisher("/path_ahead", Marker, queue_size=10)

        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)

        rospy.Subscriber("/anymal/pose_in_sim_world", Odometry, self.robot_pose_callback)

        self.current_path = []
        self.current_waypoint_idx = 0
        self.robot_position = None  # Store the robot's current position

    def robot_pose_callback(self, odom_msg):
        """
        Callback to update the robot's current position from the /anymal/pose_in_sim_world topic.
        """
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

    def path_callback(self, path_msg):
        # Extract waypoints from the Path message
        self.current_path = [(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) for pose in path_msg.poses]
        self.current_waypoint_idx = 0
        rospy.loginfo(f"Received new path with {len(self.current_path)} waypoints.")

    def calculate_orientation(self, current_waypoint, next_waypoint):
        """
        Calculate the yaw angle (orientation) between two waypoints and convert it to a quaternion.
        """
        dx = next_waypoint[0] - current_waypoint[0]
        dy = next_waypoint[1] - current_waypoint[1]
        yaw = math.atan2(dy, dx)  # Calculate yaw angle
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw)  # Convert yaw to quaternion
        return Quaternion(*quaternion)

    def publish_next_waypoint(self, distance_threshold=1.0):
        if self.current_waypoint_idx >= len(self.current_path):
            rospy.loginfo("Path completed.")
            return
    
        if self.robot_position is None:
            rospy.logwarn("Robot position not available. Skipping waypoint publishing.")
            return
    
        current_waypoint = self.current_path[self.current_waypoint_idx]
    
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
    
        if self.current_waypoint_idx + 1 < len(self.current_path):
            next_waypoint = self.current_path[self.current_waypoint_idx + 1]
            orientation = self.calculate_orientation(current_waypoint, next_waypoint)
        else:
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
    
        if self.current_candidate_xyz_idx < len(self.candidate_points_xyz):
            candidate_point = self.candidate_points_xyz[self.current_candidate_xyz_idx]
            rospy.loginfo(f"Robot position: {self.robot_position}")
            rospy.loginfo(f"Candidate point: {candidate_point}")
    
            try:
                distance_to_next_candidate = math.sqrt(
                    (next_waypoint[0] - candidate_point[0]) ** 2 +
                    (next_waypoint[1] - candidate_point[1]) ** 2 
                    # (next_waypoint[2] - candidate_point[2]) ** 2
                )
            except Exception as e:
                rospy.logerr(f"Error calculating distance to next candidate: {e}")
                return
    
            # rospy.loginfo(f"Current candidate index: {self.current_candidate_xyz_idx}")
            if distance_to_next_candidate < 0.5:
                scanned_target_voxels_local = self.scanned_voxels & self.target_voxels_candidates[self.current_candidate_xyz_idx]
    
                unscanned_voxels_local = self.target_voxels_candidates[self.current_candidate_xyz_idx] & ~scanned_target_voxels_local
                self.publish_current_target_voxels()
    
                if cp.any(unscanned_voxels_local):
                    total_voxels_local = cp.sum(self.target_voxels_candidates[self.current_candidate_xyz_idx])
                    unscanned_voxels_count = cp.sum(unscanned_voxels_local)
                    unscanned_percentage = (unscanned_voxels_count / total_voxels_local) * 100
    
                    # rospy.loginfo(f"Unscanned percentage at candidate point {self.current_candidate_xyz_idx}: {unscanned_percentage:.2f}%")
    
                    if unscanned_percentage > 20.0:
                        unscanned_indices = cp.argwhere(unscanned_voxels_local).get()
                        unscanned_patch_size = self.find_largest_continuous_patch(unscanned_indices)
    
                        # rospy.loginfo(f"Largest unscanned patch size: {unscanned_patch_size} voxels")
    
                        if unscanned_patch_size > 10:
                            rospy.loginfo(f"Large unscanned patch detected at candidate point {self.current_candidate_xyz_idx}.")
    
                            unscanned_center = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
                            shift_distance = 1
                            rotation_matrix = tf_trans.quaternion_matrix(tf_trans.quaternion_from_euler(0, 0, math.radians(self.candidate_points_angles[self.current_candidate_xyz_idx])))[:3, :3]
                            shift_vector = rotation_matrix @ np.array([-shift_distance, 0, 0])  # Shift along the negative Z-axis
                            unscanned_center += shift_vector
                            angle = self.candidate_points_angles[self.current_candidate_xyz_idx]
                            angle = math.radians(angle)
                            orientation = tf_trans.quaternion_from_euler(0, 0, angle)  # Roll = 0, Pitch = 0, Yaw = angle
                            orientation = Quaternion(*orientation)
                            rospy.loginfo(f"Navigating to the center of the unscanned region: {unscanned_center}")
    
                            # Save the current global path pose
                            saved_pose = current_waypoint
    
                            # Navigate to the unscanned region and keep scanning until it is fully scanned
                            while True:
                                # Recalculate the unscanned voxels
                                scanned_target_voxels_local = self.scanned_voxels & self.target_voxels_candidates[self.current_candidate_xyz_idx]
                                unscanned_voxels_local = self.target_voxels_candidates[self.current_candidate_xyz_idx] & ~scanned_target_voxels_local
                            
                                # Check if the unscanned area has been fully scanned
                                if not cp.any(unscanned_voxels_local):
                                    rospy.loginfo("Unscanned area has been fully scanned.")
                                    break
                            
                                # Calculate the percentage of unscanned voxels
                                total_voxels_local = cp.sum(self.target_voxels_candidates[self.current_candidate_xyz_idx])
                                unscanned_voxels_count = cp.sum(unscanned_voxels_local)
                                unscanned_percentage = (unscanned_voxels_count / total_voxels_local) * 100
                            
                                rospy.loginfo(f"Unscanned percentage: {unscanned_percentage:.2f}%")
                            
                                # Break if the unscanned percentage is below the threshold
                                coverage_threshold = 10.0  # threshold in percentage
                                if unscanned_percentage <= coverage_threshold:
                                    rospy.loginfo(f"Unscanned area coverage is below the threshold ({coverage_threshold}%). Breaking loop.")
                                    break
                            
                                # Recalculate the unscanned center dynamically
                                unscanned_indices = cp.argwhere(unscanned_voxels_local).get()
                                unscanned_center = np.mean(unscanned_indices, axis=0) * self.planner.voxel_size + self.planner.min_idx * self.planner.voxel_size
                                rotation_matrix = tf_trans.quaternion_matrix(tf_trans.quaternion_from_euler(0, 0, math.radians(self.candidate_points_angles[self.current_candidate_xyz_idx])))[:3, :3]
                                shift_vector = rotation_matrix @ np.array([-shift_distance, 0, 0 ])  # Shift along the negative Z-axis
                                unscanned_center += shift_vector
                                rospy.loginfo(f"Updated unscanned center: {unscanned_center}")
                            
                                # Publish the goal to the updated unscanned center
                                goal_msg2 = PoseStamped()
                                goal_msg2.header.stamp = rospy.Time.now()
                                goal_msg2.header.frame_id = "map"
                                goal_msg2.pose.position.x = unscanned_center[0]
                                goal_msg2.pose.position.y = unscanned_center[1]
                                goal_msg2.pose.position.z = unscanned_center[2]
                                goal_msg2.pose.orientation = orientation
                            
                                self.goal_pub.publish(goal_msg2)
                            
                                rospy.sleep(1.0)  # Wait for a short duration before checking again
    
                            # Navigate back to the saved pose
                            rospy.loginfo(f"Returning to the saved pose on the global path: {saved_pose}")
                            return_msg = PoseStamped()
                            return_msg.header.stamp = rospy.Time.now()
                            return_msg.header.frame_id = "map"
                            return_msg.pose.position.x = saved_pose[0]
                            return_msg.pose.position.y = saved_pose[1]
                            return_msg.pose.position.z = saved_pose[2]
                            return_msg.pose.orientation = Quaternion(*self.robot_orientation)  # Use the robot's current orientation
    
                            self.goal_pub.publish(return_msg)
    
    
                            # Increment the waypoint index to continue following the path
                            self.current_waypoint_idx += 1
                            return  # Exit this iteration to allow the robot to continue following the path
    
                self.goal_pub.publish(goal_msg)
                self.current_candidate_xyz_idx += 1
                if self.current_candidate_xyz_idx >= len(self.candidate_points_xyz):
                    self.current_candidate_xyz_idx = len(self.candidate_points_xyz) - 1
    
        self.current_waypoint_idx += 1
    def find_largest_continuous_patch(self, unscanned_indices):
        """
        Find the largest continuous patch of unscanned voxels.

        Args:
            unscanned_indices (np.ndarray): Indices of unscanned voxels.

        Returns:
            int: Size of the largest continuous patch.
        """
        from scipy.ndimage import label

        unscanned_grid = np.zeros(self.planner.grid_shape, dtype=bool)
        for idx in unscanned_indices:
            unscanned_grid[tuple(idx)] = True

        labeled_grid, num_features = label(unscanned_grid)

        patch_sizes = np.bincount(labeled_grid.ravel())[1:]  # Exclude the background (label 0)

        return np.max(patch_sizes) if len(patch_sizes) > 0 else 0
    def detect_unscanned_region(self, roi_size=5.0):
        """
        Detect the unscanned region of the target voxels around the current position.
    
        Args:
            roi_size (float): The size of the region of interest (ROI) in meters.
    
        Returns:
            np.ndarray: Indices of unscanned voxels in the ROI.
        """
        if self.robot_position is None:
            rospy.logwarn("Robot position not available. Cannot detect unscanned region.")
            return None
    
        robot_voxel_idx = np.floor(np.array(self.robot_position) / self.planner.voxel_size).astype(int)
    
        roi_voxel_radius = int(roi_size / self.planner.voxel_size)
        min_idx = np.maximum(robot_voxel_idx - roi_voxel_radius, 0)
        max_idx = np.minimum(robot_voxel_idx + roi_voxel_radius, np.array(self.planner.grid_shape) - 1)
    
        target_roi = self.target_voxels[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
        scanned_roi = self.scanned_voxels[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
    
        unscanned_roi = target_roi & ~scanned_roi
    
        unscanned_indices = np.argwhere(cp.asnumpy(unscanned_roi))
    
        global_unscanned_indices = unscanned_indices + min_idx
    
        rospy.loginfo(f"Detected {len(global_unscanned_indices)} unscanned voxels in the ROI.")
        return global_unscanned_indices
    def publish_target_voxels(self):
        """Publish the total target voxels as a MarkerArray."""
        marker_array = MarkerArray()
        indices = cp.argwhere(self.target_voxels).get()  # Convert CuPy array to NumPy for visualization
        for i, idx in enumerate(indices):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "target_voxels"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (idx[0] + self.planner.min_idx[0]) * self.planner.voxel_size
            marker.pose.position.y = (idx[1] + self.planner.min_idx[1]) * self.planner.voxel_size
            marker.pose.position.z = (idx[2] + self.planner.min_idx[2]) * self.planner.voxel_size
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.planner.voxel_size
            marker.scale.y = self.planner.voxel_size
            marker.scale.z = self.planner.voxel_size
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)
        self.target_voxels_pub.publish(marker_array)
    def publish_current_target_voxels(self):
        """Publish the current target voxels as a MarkerArray."""
        marker_array = MarkerArray()
        indices = cp.argwhere(self.target_voxels_candidates[self.current_candidate_xyz_idx]).get()
        for i, idx in enumerate(indices):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "current_target_voxels"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (idx[0] + self.planner.min_idx[0]) * self.planner.voxel_size
            marker.pose.position.y = (idx[1] + self.planner.min_idx[1]) * self.planner.voxel_size
            marker.pose.position.z = (idx[2] + self.planner.min_idx[2]) * self.planner.voxel_size
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.planner.voxel_size
            marker.scale.y = self.planner.voxel_size
            marker.scale.z = self.planner.voxel_size
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)
        self.current_target_voxels_pub.publish(marker_array)
    def publish_scanned_voxels(self):
        """Publish the scanned target voxels as a MarkerArray."""
        marker_array = MarkerArray()
        indices = cp.argwhere(self.scanned_voxels & self.target_voxels).get()  # Convert CuPy array to NumPy
        for i, idx in enumerate(indices):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "scanned_voxels"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (idx[0] + self.planner.min_idx[0]) * self.planner.voxel_size
            marker.pose.position.y = (idx[1] + self.planner.min_idx[1]) * self.planner.voxel_size
            marker.pose.position.z = (idx[2] + self.planner.min_idx[2]) * self.planner.voxel_size
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.planner.voxel_size
            marker.scale.y = self.planner.voxel_size
            marker.scale.z = self.planner.voxel_size
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)
        self.scanned_voxels_pub.publish(marker_array)
    
    def lidar_callback(self, msg):
        try:
            # Get the transformation from 'base' frame to 'map' frame
            # "depth_camera_front_upper_depth_optical_frame" frame for depth camera
            
            (trans, rot) = self.tf_listener.lookupTransform('map', 'depth_camera_front_upper_depth_optical_frame', rospy.Time(0))
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[:3, 3] = trans
    
            points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            voxel_size = 0.2
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    
            # Transform the downsampled points to the 'map' frame
            points_transformed = np.asarray(pcd_downsampled.points)
            points_transformed = (transform_matrix[:3, :3] @ points_transformed.T).T + transform_matrix[:3, 3]
            points_cp = cp.array(points_transformed)
            voxel_indices = cp.floor(points_cp / self.planner.voxel_size).astype(cp.int32) - cp.array(self.planner.min_idx)
            # Add tolerance to the voxelization process
            tolerance = 0.2  # tolerance in meters
            tolerance_voxels = int(tolerance / self.planner.voxel_size)
    
            # Generate neighboring voxel indices within the tolerance range
            offsets = cp.array([[dx, dy, dz] for dx in range(-tolerance_voxels, tolerance_voxels + 1)
                                for dy in range(-tolerance_voxels, tolerance_voxels + 1)
                                for dz in range(-tolerance_voxels, tolerance_voxels + 1)])
            voxel_indices_with_tolerance = voxel_indices[:, None, :] + offsets[None, :, :]
            voxel_indices_with_tolerance = voxel_indices_with_tolerance.reshape(-1, 3)
    
            valid_mask = cp.all((voxel_indices_with_tolerance >= 0) & (voxel_indices_with_tolerance < cp.array(self.planner.grid_shape)), axis=1)
            valid_voxel_indices = voxel_indices_with_tolerance[valid_mask]

            for idx in valid_voxel_indices:
                self.scanned_voxels[tuple(idx.get())] = True  
    
            scanned_target_voxels = self.scanned_voxels & self.target_voxels
            remaining_target_voxels = self.target_voxels & ~scanned_target_voxels
            rospy.loginfo(f"Scanned target voxels: {cp.sum(scanned_target_voxels).get()}")
            rospy.loginfo(f"Remaining target voxels: {cp.sum(remaining_target_voxels).get()}")
    

            self.publish_scanned_voxels()
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF transformation failed: {e}")
    def follow_path(self):
        rate = rospy.Rate(4)  # 4 Hz
        while not rospy.is_shutdown():
            self.publish_next_waypoint(distance_threshold=1.0)  # Check distance before publishing
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("lidar_mapping_node")

    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)

    planner.loadTomogram("building_2F_4R")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd", 0.2)

    node = LidarMappingNode(planner)
    node.follow_path()  # Start following the path
    rospy.spin()
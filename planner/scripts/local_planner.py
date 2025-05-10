import rospy
import numpy as np
import cupy as cp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from planner_wrapper_coveragev2 import TomogramCoveragePlanner
from config import Config
import tf


class LidarMappingNode:
    def __init__(self, planner):
        self.planner = planner
        self.tf_listener = tf.TransformListener()  # Initialize TF listener

        # Load candidate points
        candidate_points_xyz = np.load("reachable_sampled_points.npy")
        candidate_points_angles = np.load("reachable_sampled_points_angles.npy")

        # Use CuPy arrays for target_voxels and scanned_voxels
        self.target_voxels = planner.compute_explored_voxels(candidate_points_xyz, candidate_points_angles)
        self.scanned_voxels = cp.zeros_like(self.target_voxels, dtype=cp.bool_)

        # ROS Publishers for visualization
        self.target_voxels_pub = rospy.Publisher("target_voxels", MarkerArray, queue_size=10)
        self.scanned_voxels_pub = rospy.Publisher("scanned_voxels", MarkerArray, queue_size=10)

        # Subscribe to the LiDAR topic
        rospy.Subscriber("/point_cloud_filter/lidar_depth_camera/point_cloud_filtered", PointCloud2, self.lidar_callback)

        # Publish the initial target voxels
        self.publish_target_voxels()

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
            # Get the transformation from 'lidar' frame to 'map' frame
            (trans, rot) = self.tf_listener.lookupTransform('map', 'lidar', rospy.Time(0))
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[:3, 3] = trans

            # Convert PointCloud2 to a NumPy array
            points = np.array([[p[0], p[1], p[2], 1.0] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])

            # Transform points to the 'map' frame
            points_transformed = (transform_matrix @ points.T).T[:, :3]

            # Convert points to a CuPy array for GPU processing
            points_cp = cp.array(points_transformed)

            # Convert points to voxel indices using CuPy
            voxel_indices = cp.floor(points_cp / self.planner.voxel_size).astype(cp.int32) - cp.array(self.planner.min_idx)

            # Filter valid voxel indices (within grid bounds)
            valid_mask = cp.all((voxel_indices >= 0) & (voxel_indices < cp.array(self.planner.grid_shape)), axis=1)
            valid_voxel_indices = voxel_indices[valid_mask]

            # Update the scanned voxel map
            for idx in valid_voxel_indices:
                self.scanned_voxels[tuple(idx.get())] = True  # Convert CuPy index to NumPy for assignment

            # Compare scanned voxels with target voxels using CuPy
            scanned_target_voxels = self.scanned_voxels & self.target_voxels
            remaining_target_voxels = self.target_voxels & ~scanned_target_voxels

            # Log the results (convert to NumPy for logging)
            rospy.loginfo(f"Scanned target voxels: {cp.sum(scanned_target_voxels).get()}")
            rospy.loginfo(f"Remaining target voxels: {cp.sum(remaining_target_voxels).get()}")

            # Publish the scanned voxels for visualization
            self.publish_scanned_voxels()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF transformation failed: {e}")


if __name__ == "__main__":
    rospy.init_node("lidar_mapping_node")

    # Initialize the planner
    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)

    # Load voxel map and tomogram
    planner.loadTomogram("building_2F_4R")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd", 0.2)

    # Start the node
    node = LidarMappingNode(planner)
    rospy.spin()
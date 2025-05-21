import rospy
import numpy as np
import cupy as cp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf
import open3d as o3d
import rospy
import numpy as np
import cupy as cp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from planner_wrapper_coveragev2 import TomogramCoveragePlanner
from config import Config
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
class LidarMapperNode:
    def __init__(self, planner):
        rospy.init_node("lidar_mapper_node")
        self.planner = planner
        self.tf_listener = tf.TransformListener()
        self.voxel_size = planner.voxel_size 
        self.grid_shape = planner.grid_shape  
        self.min_idx = planner.min_idx

        self.lidar_voxel_pub = rospy.Publisher("/current_lidar_voxel_grid", PointCloud2, queue_size=10)
        rospy.Subscriber("/depth_camera_front_upper/point_cloud_self_filtered", PointCloud2, self.lidar_callback)

    def lidar_callback(self, msg):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('map', 'depth_camera_front_upper_depth_optical_frame', rospy.Time(0))
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[:3, 3] = trans

            points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
            if points.shape[0] == 0:
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_downsampled = pcd.voxel_down_sample(self.voxel_size)

            points_cp = cp.array(np.asarray(pcd_downsampled.points))
            transform_cp = cp.array(transform_matrix)
            ones = cp.ones((points_cp.shape[0], 1), dtype=points_cp.dtype)
            points_hom = cp.concatenate([points_cp, ones], axis=1)
            points_transformed_cp = points_hom @ transform_cp.T
            points_cp = points_transformed_cp[:, :3]

            voxel_indices = cp.floor(points_cp / self.voxel_size).astype(cp.int32) - cp.array(self.min_idx)
            valid_mask = cp.all((voxel_indices >= 0) & (voxel_indices < cp.array(self.grid_shape)), axis=1)
            valid_voxel_indices = voxel_indices[valid_mask]
            indices_np = cp.asnumpy(valid_voxel_indices)

            # Publish as PointCloud2 for visualization or further processing
            points = []
            for idx in indices_np:
                x = (idx[0] + self.min_idx[0] + 0.5) * self.voxel_size
                y = (idx[1] + self.min_idx[1] + 0.5) * self.voxel_size
                z = (idx[2] + self.min_idx[2] + 0.5) * self.voxel_size
                points.append([x, y, z])
            header = msg.header
            header.stamp = rospy.Time.now()
            header.frame_id = "map"
            pc2_msg = pc2.create_cloud_xyz32(header, points)
            self.lidar_voxel_pub.publish(pc2_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF transformation failed: {e}")

if __name__ == "__main__":
    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)

    # planner.loadTomogram("building_2F_4R")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_2F_4R.pcd", 0.2)
    node = LidarMapperNode(planner)
    rospy.spin()
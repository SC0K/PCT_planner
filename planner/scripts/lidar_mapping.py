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
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
import cupyx.scipy.ndimage
class LidarMapperNode:
    def __init__(self, planner):
        rospy.init_node("lidar_mapper_node")
        self.planner = planner
        self.tf_listener = tf.TransformListener()
        self.voxel_size = planner.voxel_size 
        self.grid_shape = planner.grid_shape  
        self.min_idx = planner.min_idx

        self.lidar_voxel_pub = rospy.Publisher("/current_lidar_voxel_grid", PointCloud2, queue_size=1)
        rospy.Subscriber("/depth_camera_front_upper/point_cloud_self_filtered", PointCloud2, self.lidar_callback)
        self.lidar_voxel_idx_pub = rospy.Publisher("/current_lidar_voxel_indices", Int32MultiArray, queue_size=1)
        self.lidar_new_voxel_idx_pub = rospy.Publisher("/current_lidar_new_voxel_indices", Int32MultiArray, queue_size=1)
        self.persistence_counter = np.zeros(planner.grid_shape, dtype=np.uint16)
        self.persistence_threshold = 10
        self.added_voxels = set()
        self.dilate_scanned_voxels = False
        self.dilation_size_scanned = 1

    def lidar_callback(self, msg):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                'map', 'depth_camera_front_upper_depth_optical_frame', msg.header.stamp)
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[:3, 3] = trans
    
            points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
            if points.shape[0] == 0:
                return
    
            # Downsample and transform points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_downsampled = pcd.voxel_down_sample(self.voxel_size)
            points_np = np.asarray(pcd_downsampled.points)
            if points_np.shape[0] == 0:
                return
    
            points_cp = cp.array(points_np)
            transform_cp = cp.array(transform_matrix)
            ones = cp.ones((points_cp.shape[0], 1), dtype=points_cp.dtype)
            points_hom = cp.concatenate([points_cp, ones], axis=1)
            points_transformed_cp = points_hom @ transform_cp.T
            points_cp = points_transformed_cp[:, :3]
    
            # Compute voxel indices
            voxel_indices_cp = cp.floor(points_cp / self.voxel_size).astype(cp.int32) - cp.array(self.min_idx)
            valid_mask = cp.all((voxel_indices_cp >= 0) & (voxel_indices_cp < cp.array(self.grid_shape)), axis=1)
            valid_voxel_indices_cp = voxel_indices_cp[valid_mask]
            indices_np = cp.asnumpy(valid_voxel_indices_cp)
            if indices_np.shape[0] == 0:
                return

            # === Optionally dilate scanned voxels ===
            if self.dilate_scanned_voxels:
                scanned_grid = cp.zeros(self.grid_shape, dtype=cp.bool_)
                scanned_grid[valid_voxel_indices_cp[:, 0], valid_voxel_indices_cp[:, 1], valid_voxel_indices_cp[:, 2]] = True
                structure = cp.ones((2 * self.dilation_size_scanned + 1,) * 3, dtype=cp.bool_)
                scanned_grid = cupyx.scipy.ndimage.binary_dilation(scanned_grid, structure=structure)
                dilated_indices_cp = cp.argwhere(scanned_grid)
                indices_np = cp.asnumpy(dilated_indices_cp)

            # Publish as PointCloud2 for visualization
            vis_points = []
            for idx in indices_np:
                x = (idx[0] + self.min_idx[0] + 0.5) * self.voxel_size
                y = (idx[1] + self.min_idx[1] + 0.5) * self.voxel_size
                z = (idx[2] + self.min_idx[2] + 0.5) * self.voxel_size
                vis_points.append([x, y, z])
            header = msg.header
            header.stamp = rospy.Time.now()
            header.frame_id = "map"
            pc2_msg = pc2.create_cloud_xyz32(header, vis_points)
            self.lidar_voxel_pub.publish(pc2_msg)
    
            # Publish all voxel indices
            idx_msg = Int32MultiArray()
            idx_msg.data = indices_np.flatten().tolist()
            idx_msg.layout.dim.append(MultiArrayDimension(label="voxel", size=indices_np.shape[0], stride=indices_np.size))
            idx_msg.layout.dim.append(MultiArrayDimension(label="xyz", size=3, stride=3))
            self.lidar_voxel_idx_pub.publish(idx_msg)
    
            # --- Find and publish new voxels (outside hash grid with dilation) ---
            if indices_np.shape[0] > 0:
                indices_cp = cp.array(indices_np)
                tx = indices_cp[:, 0]
                ty = indices_cp[:, 1]
                tz = indices_cp[:, 2]
    
                # Dilation control for hash grid
                if hasattr(self.planner, "hash_grid"):
                    hash_grid = self.planner.hash_grid
                else:
                    hash_grid = cp.zeros(self.grid_shape, dtype=cp.bool_)
                dilation_size = getattr(self, "dilation_size", 1)
                use_dilation = getattr(self, "use_dilation_new_obstacles", True)
                if use_dilation:
                    structure = cp.ones((2 * dilation_size + 1,) * 3, dtype=cp.bool_)
                    dilated_hash_grid = cupyx.scipy.ndimage.binary_dilation(hash_grid, structure=structure)
                else:
                    dilated_hash_grid = hash_grid
    
                not_in_dilated_hash_mask = ~(dilated_hash_grid[tx, ty, tz])
                filtered_indices = indices_np[cp.asnumpy(not_in_dilated_hash_mask)]
            if len(filtered_indices) > 0:
                msg_indices = Int32MultiArray()
                msg_indices.data = np.array(filtered_indices, dtype=np.int32).flatten().tolist()
                msg_indices.layout.dim.append(MultiArrayDimension(label="voxel", size=len(filtered_indices), stride=len(filtered_indices)*3))
                msg_indices.layout.dim.append(MultiArrayDimension(label="xyz", size=3, stride=3))
                self.lidar_new_voxel_idx_pub.publish(msg_indices)
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF transformation failed: {e}")

if __name__ == "__main__":
    cfg = Config()
    planner = TomogramCoveragePlanner(cfg)

    # planner.loadTomogram("building_2F_4R")
    planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/experiments/2F_2*1.pcd", 0.1075)
    # planner.loadVoxelMap("/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/rsc/pcd/building_LEE_1F.pcd", 0.2)
    node = LidarMapperNode(planner)
    rospy.spin()
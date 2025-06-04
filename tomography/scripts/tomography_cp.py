#!/usr/bin/python3
import os
import sys
import time
import pickle
import numpy as np
import open3d as o3d
  
import rospy
from std_msgs.msg import Header, Empty
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker

from tomogram_cp import Tomogram

sys.path.append('../')
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(object):
    def __init__(self, cfg, scene_cfg):
        self.export_dir = rsg_root + cfg.map.export_dir
        self.pcd_file = scene_cfg.pcd.file_name
        self.resolution = scene_cfg.map.resolution
        self.ground_h = scene_cfg.map.ground_h
        self.slice_dh = scene_cfg.map.slice_dh

        self.center = np.zeros(2, dtype=np.float32)
        self.tomogram = Tomogram(scene_cfg)
        points = self.loadPCD(self.pcd_file)

        # Process
        self.process(points)

    def initROS(self):
        self.map_frame = cfg.ros.map_frame

        pointcloud_topic = cfg.ros.pointcloud_topic
        self.pointcloud_pub = rospy.Publisher(pointcloud_topic, PointCloud2, latch=True, queue_size=1)

        self.layer_G_pub_list = []
        self.layer_C_pub_list = []
        layer_G_topic = cfg.ros.layer_G_topic
        layer_C_topic = cfg.ros.layer_C_topic
        for i in range(self.n_slice):
            layer_G_pub = rospy.Publisher(layer_G_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_G_pub_list.append(layer_G_pub)
            layer_C_pub = rospy.Publisher(layer_C_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_C_pub_list.append(layer_C_pub)

        tomogram_topic = cfg.ros.tomogram_topic
        self.tomogram_pub = rospy.Publisher(tomogram_topic, PointCloud2, latch=True, queue_size=1)
        self.tomogram_pub_raw = rospy.Publisher(tomogram_topic + "_raw", PointCloud2, latch=True, queue_size=1)

    def loadPCD(self, pcd_file):
        pcd = o3d.io.read_point_cloud(rsg_root + "/rsc/pcd/" + pcd_file)
        points = np.asarray(pcd.points).astype(np.float32)
        rospy.loginfo("PCD points: %d", points.shape[0])

        if points.shape[1] > 3:
            points = points[:, :3]
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        rospy.loginfo("Map center: [%.2f, %.2f]", self.center[0], self.center[1])
        rospy.loginfo("Dim_x: %d", self.map_dim_x)
        rospy.loginfo("Dim_y: %d", self.map_dim_y)
        rospy.loginfo("Num slices init: %d", n_slice_init)

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points
        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 1

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu,raw_cost = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        self.n_slice = layers_g.shape[0]

        map_file = os.path.splitext(self.pcd_file)[0]
        # self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c,raw_cost)), map_file)

        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)
        self.publishRawTomogram(layers_g, raw_cost)
        self.layers_g = layers_g

    def process_modified(self, points, start_end_indices):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 1

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu,raw_cost = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        

        radius = int(0.3 / self.resolution)

        # Process all start and end point pairs
        for start_idx, end_idx in start_end_indices:
            rospy.loginfo("Start point: %s", start_idx)
            rospy.loginfo("End point: %s", end_idx)
            s1, y1, x1 = start_idx
            s2, y2, x2 = end_idx

            num_steps = int(np.linalg.norm([s2 - s1, y2 - y1, x2 - x1]) * 2)
            s_vals = np.linspace(s1, s2, num_steps).astype(int)
            y_vals = np.linspace(y1, y2, num_steps).astype(int)
            x_vals = np.linspace(x1, x2, num_steps).astype(int)

            for s, y, x in zip(s_vals, y_vals, x_vals):
                if 0 <= s < layers_g.shape[0] and 0 <= y < self.map_dim_y and 0 <= x < self.map_dim_x:
                    y_min, y_max = max(0, y - radius), min(self.map_dim_y, y + radius + 1)
                    x_min, x_max = max(0, x - radius), min(self.map_dim_x, x + radius + 1)
                    layers_t[s, x_min:x_max, y_min:y_max] = 50


        map_file = os.path.splitext(self.pcd_file)[0]
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c,raw_cost)), map_file)

        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)
        self.publishRawTomogram(layers_g, raw_cost)


    def exportTomogram(self, tomogram, map_file):        
        data_dict = {
            'data': tomogram.astype(np.float32),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Tomogram exported: %s", file_name)

    def publishPoints(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        point_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_msg)

    def publishLayers(self, pub_list, layers, color=None):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        for i in range(layers.shape[0]):
            layer_points[:, 2] = layers[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            if color is not None:
                layer_points[:, 3] = color[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            else:
                layer_points[:, 3] = 1.0
        
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, valid_points)
            pub_list[i].publish(points_msg) 

    def publishTomogram(self, layers_g, layers_t):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        vis_g = layers_g.copy()
        vis_t = layers_t.copy() 
        print("vis_g shape: ", vis_g.shape)
        print("vis_t shape: ", vis_t.shape)
        
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        for i in range(n_slice - 1):
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
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

    def publishRawTomogram(self, layers_g, layers_t):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        vis_g = layers_g.copy()
        vis_t = layers_t.copy() 
        print("vis_g shape: ", vis_g.shape)
        print("vis_t shape: ", vis_t.shape)
        
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        for i in range(n_slice - 1):
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
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
        self.tomogram_pub_raw.publish(points_msg)

    def initPathInjection(self):
        self.clicked_points = []
        self.all_paths = []  # Store all pairs of traversable paths
        self.clicked_points_pub = rospy.Publisher("/clicked_points_marker", Marker, queue_size=10)
        rospy.Subscriber("/clicked_point", PointStamped, self.clicked_point_callback)
        rospy.Subscriber("/end_input", Empty, self.end_input_callback)  # Subscribe to an "end input" topic
        rospy.loginfo("Click pairs of points in RViz to inject traversable paths. Use rostopic pub /end_input std_msgs/Empty \"{}\" to finish input.")
    
    def clicked_point_callback(self, msg):
        self.clicked_points.append(msg.point)
        rospy.loginfo(f"Clicked: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})")
    
        # Publish the clicked points for visualization
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "clicked_points"
        marker.id = len(self.clicked_points)  # Unique ID for each point
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = msg.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # Adjust size as needed
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
        self.clicked_points_pub.publish(marker)
    
        if len(self.clicked_points) == 2:
            # Store the pair of points
            self.all_paths.append((self.clicked_points[0], self.clicked_points[1]))
            rospy.loginfo("Traversable path added between points.")
            self.clicked_points = []  # Reset for the next pair
    
    def end_input_callback(self, msg):
        rospy.loginfo("End of input detected. Processing all paths.")
        points = self.loadPCD(self.pcd_file)
    
        # Convert all point pairs to start and end indices
        start_end_indices = []
        for start_point, end_point in self.all_paths:
            start_xyz = np.array([start_point.x, start_point.y, start_point.z])
            end_xyz = np.array([end_point.x, end_point.y, end_point.z])
            start_idx = self.pos2idx_3D(start_xyz)
            end_idx = self.pos2idx_3D(end_xyz)
            start_end_indices.append((start_idx, end_idx))
    
        # Pass all pairs to process_modified
        self.process_modified(points, start_end_indices)
    
        rospy.loginfo("All paths processed.")
        self.all_paths = []  # Clear the paths after processing
    def pos2idx_3D(self, pos):
        """
        Convert a 3D position (x, y, z) to grid indices (s, y, x), where s is the layer number.
        
        Args:
            pos (np.ndarray): The 3D position (x, y, z).
        
        Returns:
            np.ndarray: The grid indices (s, y, x).
        """
        # Subtract the center to align with the grid
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy - self.center
        
        # Calculate x and y indices
        offset = np.array([int(self.map_dim_x / 2), int(self.map_dim_y / 2)], dtype=np.int32)
        idx_xy = np.round(pos_xy / self.resolution).astype(np.int32) + offset
        idx_xy = np.array([idx_xy[1], idx_xy[0]], dtype=np.int32)  # Swap x and y for grid indexing
        
        # Search for the z index (layer number) using the precomputed layer modes
        z_height = pos[2]  # Extract the z-coordinate
        z_idx = -1  # Default to -1 if no valid layer is found
        for s in range(self.layers_g.shape[0]):
            print(f"Layer height {s}: {self.layers_g[s, idx_xy[1], idx_xy[0]]}")
            if abs(z_height - self.layers_g[s, idx_xy[1], idx_xy[0]]) <= self.resolution*2:
                z_idx = s
                break
        
        # Combine z_idx with x and y indices
        idx = np.array([z_idx, idx_xy[0], idx_xy[1]], dtype=np.float32)
        return idx



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
    args = parser.parse_args()

    cfg = Config()
    scene_cfg = getattr(__import__('config'), 'Scene' + args.scene)

    rospy.init_node('pointcloud_tomography', anonymous=True)

    mapping = Tomography(cfg, scene_cfg)
    mapping.initPathInjection()

    rospy.spin()
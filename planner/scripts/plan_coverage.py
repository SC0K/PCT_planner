import sys
import argparse
import numpy as np

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from utils import *
from planner_wrapper_coverage import TomogramCoveragePlanner

sys.path.append('../')
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
args = parser.parse_args()

cfg = Config()

if args.scene == 'Spiral':
    tomo_file = 'spiral0.3_2'
    start_pos = np.array([-16.0, -6.0], dtype=np.float32)
    end_pos = np.array([-26.0, -5.0], dtype=np.float32)
elif args.scene == 'Building':
    tomo_file = 'building2_9'
    start_pos = np.array([5.0, 4.0, 5], dtype=np.float32)
    end_pos = np.array([-6.0, -1.0, 5], dtype=np.float32)
else:
    tomo_file = 'plaza3_10'
    start_pos = np.array([0.0, 0.0], dtype=np.float32)
    end_pos = np.array([23.0, 10.0], dtype=np.float32)

path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)
planner = TomogramCoveragePlanner(cfg)

sampled_points_pub = rospy.Publisher("/sampled_points", PointCloud2, latch=True, queue_size=1)

def pct_plan():
    planner.loadTomogram(tomo_file)

    # traj_3d = planner.plan(start_pos, end_pos)
    # if traj_3d is not None:
    #     path_pub.publish(traj2ros(traj_3d))
    #     print("Trajectory published")

    # sampled_points_idx, sampled_points_xyz = planner.sampleTraversablePoints( num_samples=1000)
    # sampled_points_idx, sampled_points_xyz = planner.sampleUniformPointsInSpace()
    # print("Candidate points:", sampled_points_xyz.shape)
    # publish_points(sampled_points_xyz)

    # computeNBVpoints()

    # Publish sampled points
    candidate_points_xyz = np.load("sampled_points.npy")
    candidate_points_idx = np.load("sampled_points_idx.npy").astype(np.int32)
    print("Candidate points:", candidate_points_xyz.shape)

    candidate_points_idx = np.array([[4, 20,  80],[  6, 140, 200]])
    candidate_points_xyz = np.zeros_like(candidate_points_idx, dtype=np.float32)
    candidate_points_xyz[0] = planner.idx2pos_3D(candidate_points_idx[0])
    candidate_points_xyz[1] = planner.idx2pos_3D(candidate_points_idx[1])

    publish_points(candidate_points_xyz)
    traj_3d = planner.plan(candidate_points_idx[0], candidate_points_idx[1])
    if traj_3d is not None:
        path_pub.publish(traj2ros(traj_3d))
        print("Trajectory published")
    # adjacency = planner.compute_adjacency_matrix(candidate_points_idx)
    # print("Adjacency matrix:", adjacency)

    
def computeNBVpoints():
    # Compute the next best view points
    candidate_points_idx, candidate_angles, candidate_points_xyz = planner.nextBestView()
    print("Candidate points:", candidate_points_xyz)
    np.save("sampled_points.npy", candidate_points_xyz)
    np.save("sampled_points_idx.npy", candidate_points_idx)
    np.save("sampled_points_angles.npy", candidate_angles)

    # Publish sampled points
    candidate_points_idx = candidate_points_idx[:, [0, 2, 1]].astype(np.int32)  # Switch the order of x and y for planning and ensure integers

    # Filter out points with the same x, y values in layers with the same mode heights
    unique_points_idx = []
    unique_points_xyz = []
    unique_angles = []
    seen_xy = {}

    for idx, point in enumerate(candidate_points_idx):
        s, y, x = point
        xy_key = (x, y)
        if xy_key not in seen_xy or seen_xy[xy_key] != planner.layer_modes[s]:
            unique_points_idx.append(point)
            unique_points_xyz.append(candidate_points_xyz[idx])  # Keep the corresponding xyz map
            unique_angles.append(candidate_angles[idx])  # Keep the corresponding angle
            seen_xy[xy_key] = planner.layer_modes[s]

    candidate_points_idx = np.array(unique_points_idx, dtype=np.int32)
    candidate_points_xyz = np.array(unique_points_xyz, dtype=np.float32)
    candidate_angles = np.array(unique_angles, dtype=np.float32)

    print("Filtered sampled points (indices):", candidate_points_idx)
    print("Filtered sampled points (xyz):", candidate_points_xyz)
    print("Filtered sampled points (angles):", candidate_angles)
    # Save the sampled points to a file
    np.save("sampled_points.npy", candidate_points_xyz)
    np.save("sampled_points_idx.npy", candidate_points_idx)
    np.save("sampled_points_angles.npy", candidate_angles)


def publish_points(points_xyz, frame_id="map"):
    """
    Publish sampled points as a PointCloud2 message.

    Args:
        sampled_points (np.ndarray): Array of sampled points (x, y indices).
        resolution (float): Resolution of the grid.
        center (np.ndarray): The center of the grid in map coordinates.
        frame_id (str): The frame ID for the PointCloud2 message.
    """
    #     # Convert sampled points to 3D coordinates (x, y, z)
    # points_3d = []
    # for s, x, y in sampled_points_idx:
    #     # Convert grid indices to map coordinates
    #     map_x = (x -  // 2) * resolution + center[0]
    #     map_y = (y -  // 2) * resolution + center[1]
    #     map_z = 0.0  # Assume z = 0 for visualization
    #     points_3d.append([map_x, map_y, map_z])

    # Create a PointCloud2 message
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]

    point_cloud_msg = pc2.create_cloud(header, fields, points_xyz)

    # Publish the message
    sampled_points_pub.publish(point_cloud_msg)
if __name__ == '__main__':
    rospy.init_node("pct_planner", anonymous=True)

    pct_plan()

    rospy.spin()
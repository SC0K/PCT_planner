import open3d as o3d
import os

input_path = "/home/sitong/catkin_workspaces/pct_planning/src/PCT_planner/World/map/building_2F_4R.ply"
output_path = os.path.splitext(input_path)[0] + ".pcd"

pcd = o3d.io.read_point_cloud(input_path)
o3d.io.write_point_cloud(output_path, pcd)

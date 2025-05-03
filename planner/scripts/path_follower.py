import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import math
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

class MotionController:
    def __init__(self):
        rospy.init_node("motion_controller", anonymous=True)

        # Subscribe to the trajectory path
        rospy.Subscriber("/pct_path", Path, self.path_callback)
        self.path_pub = rospy.Publisher("/path_ahead", Marker, queue_size=10)

        # Publisher for goal waypoints
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)

        # Subscribe to the robot's current position
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

    def publish_next_waypoint(self, distance_threshold=2.0):
        """
        Publish the next waypoint if it is within the distance threshold from the robot.

        Args:
            distance_threshold (float): Maximum allowed distance to the next waypoint.
        """
        if self.current_waypoint_idx >= len(self.current_path):
            rospy.loginfo("Path completed.")
            return

        if self.robot_position is None:
            rospy.logwarn("Robot position not available. Skipping waypoint publishing.")
            return

        # Get the current waypoint
        current_waypoint = self.current_path[self.current_waypoint_idx]

        # Calculate the distance to the current waypoint
        distance = math.sqrt(
            (current_waypoint[0] - self.robot_position[0]) ** 2 +
            (current_waypoint[1] - self.robot_position[1]) ** 2 +
            (current_waypoint[2] - self.robot_position[2]) ** 2
        )

        if distance > distance_threshold:
            rospy.logwarn(f"Next waypoint is too far ({distance:.2f} meters). Waiting...")
            return

        # Determine the next waypoint for orientation calculation
        if self.current_waypoint_idx + 1 < len(self.current_path):
            next_waypoint = self.current_path[self.current_waypoint_idx + 1]
            orientation = self.calculate_orientation(current_waypoint, next_waypoint)
        else:
            # If it's the last waypoint, keep the orientation unchanged
            orientation = Quaternion(0, 0, 0, 1)

        rospy.loginfo(f"Publishing waypoint: {current_waypoint}")

        # Create a PoseStamped message for the waypoint
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"  # Replace with the appropriate frame ID
        goal_msg.pose.position.x = current_waypoint[0]
        goal_msg.pose.position.y = current_waypoint[1]
        goal_msg.pose.position.z = current_waypoint[2]
        goal_msg.pose.orientation = orientation

        # Publish the waypoint to the /goal topic
        self.goal_pub.publish(goal_msg)

        # Move to the next waypoint
        self.current_waypoint_idx += 1

    def run(self):
        rate = rospy.Rate(4)  # 4 Hz
        while not rospy.is_shutdown():
            self.publish_next_waypoint(distance_threshold=1.2)  # Check distance before publishing
            rate.sleep()
if __name__ == "__main__":
    controller = MotionController()
    controller.run()
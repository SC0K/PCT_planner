#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker

clicked_points = []

def clicked_point_callback(msg):
    global clicked_points
    clicked_points.append(msg.point)
    rospy.loginfo(f"Point clicked: {msg.point}")

    if len(clicked_points) == 2:
        draw_line(clicked_points[0], clicked_points[1])
        clicked_points.clear()

def draw_line(p1, p2):
    marker = Marker()
    marker.header.frame_id = "map"  # or "base_link" depending on your setup
    marker.header.stamp = rospy.Time.now()
    marker.ns = "clicked_line"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.points = [p1, p2]

    marker.scale.x = 0.02  # Line width

    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration(0)  # 0 = forever

    pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('click_to_connect_points')
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rospy.Subscriber('/clicked_point', PointStamped, clicked_point_callback)
    rospy.loginfo("Click two points in RViz using the 'Publish Point' tool to connect them.")
    rospy.spin()

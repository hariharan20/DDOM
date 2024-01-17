#! /usr/bin/env python
import rospy
from tracker.msg import BoundingBox
from darknet_ros_msgs.msg import BoundingBoxes

rospy.init_node('human_boxes')
pub = rospy.Publisher('h_boxes' , BoundingBox  ,queue_size = 1)

def callback(data):
	msg = BoundingBox()
	for i in range(len(data.bounding_boxes)):
		if(data.bounding_boxes[i].Class == 'person'):
			msg.ids.append(i) 
			msg.xmin.append( data.bounding_boxes[i].xmin)
			msg.ymin.append( data.bounding_boxes[i].ymin)
			msg.xmax.append( data.bounding_boxes[i].xmax)
			msg.ymax.append( data.bounding_boxes[i].ymax)
	# rospy.loginfo(msg)
	pub.publish(msg)


rospy.Subscriber('/darknet_ros/bounding_boxes' , BoundingBoxes , callback) 
rospy.spin()
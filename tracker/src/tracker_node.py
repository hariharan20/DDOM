#! /usr/bin/env python3
import rospy
from ddom_msgs.msg import AnomalyScore
from tracker.msg import BoundingBox
from sensor_msgs.msg import Image
from sort import Sort
import numpy as np
from std_msgs.msg import Bool
# import ros_numpy
import cv_bridge
class tracklet:
    def __init__(self,  x , y , z):
        self.x = []
        self.y = []
        self.z = []
        self.last_depth = float('inf')
        self.update(x , y , z)
        self.not_known_for = 0
    def update(self , x_new , y_new , z_new):
        self.x.append(float(x_new/1000))
        self.y.append(float(y_new/1000))
        self.z.append(float(z_new/1000))
        
        self.not_known_for = 0
        if len(self.x) > 60:
            self.x.pop(0)
            self.y.pop(0)
            self.z.pop(0)
            self.last_depth  =z_new
    def get_tracklet(self):
        if len(self.x) <60:
            return None
        else:
            return [self.x , self.y, self.z]


class t2ts:
    def __init__(self):
        bounding_box_topic_name = rospy.get_param('bounding_box_topic')
        rgb_image_topic_name  = rospy.get_param('rgb_image_topic')
        depth_image_topic_name = rospy.get_param('depth_image_topic')
        sort_topic_name = rospy.get_param('sort_topic')
        self.principal_point_x = rospy.get_param('principal_point_x')
        self.principal_point_y = rospy.get_param('principal_point_y')
        self.focal_length_x = rospy.get_param('focal_length_x')
        self.focal_length_y = rospy.get_param('focal_length_y')
        self.sort_obj =  Sort()
        self.known_people = {}
        self.depth_image = None
        self.rgb_image = None
        self.empty_np = np.empty((0, 5))
        self.cvb = cv_bridge.CvBridge()
        self.pub = rospy.Publisher(sort_topic_name , AnomalyScore , queue_size=10)
        self.sub1 = rospy.Subscriber(bounding_box_topic_name , BoundingBox , self.cb1)
        self.sub2 = rospy.Subscriber(rgb_image_topic_name , Image  , self.rgb_cb)
        self.sub3 = rospy.Subscriber(depth_image_topic_name , Image  , self.depth_cb)
    
    def get_depth(self , xAvg , yAvg , depth_map):
        Zinm = depth_map[yAvg][xAvg]
        Xinm = ( xAvg - self.principal_point_x) * (Zinm / self.focal_length_x)
        Yinm = ( yAvg - self.principal_point_y) * (Zinm / self.focal_length_y)
        return Xinm , Yinm , Zinm
    
    def nearest_track(self):
        depths = []
        for person in self.known_people.keys():
            depths.append(self.known_people[person].last_depth)
        min_ = np.argmin(np.array(depths))
        return self.known_people[list(self.known_people.keys())[min_]].get_tracklet()

    def sort_to_ts(self , bb_list):
        # print(bb_list)
        # fp  =True
        depth_map = self.depth_image
        tracked_bbs = self.sort_obj.update(bb_list)
        currentIds = []
        for bb in tracked_bbs:
            person_id = bb[-1]
            currentIds.append(person_id)
            x =  round((bb[0] + bb[2])/2)
            y = round((bb[1] + bb[3])/2)
            xInMeters , yInMeters ,zInMeters  = self.get_depth(x  ,y , depth_map)
            if person_id in self.known_people.keys():
                self.known_people[person_id].update(xInMeters , yInMeters , zInMeters)

            else :
                self.known_people[person_id] = tracklet(xInMeters , yInMeters , zInMeters)
            
        for id in self.known_people.keys():
            if id not in currentIds:
                self.known_people[id].not_known_for += 1
                if self.known_people[id].not_known_for == 4 : 
                    # if len(self.known_people[id].x) < 60:
                        # fp = True
                    del self.known_people[id]

    def cb1(self , data):
        a = np.zeros((len(data.ids), 5))
        a[: , 0] = np.array(data.xmin)
        a[: , 1] = np.array(data.ymin)
        a[: , 2] = np.array(data.xmax)
        a[: , 3] = np.array(data.ymax)
        self.sort_to_ts(a)
        self.x , self.y , self.z  = self.nearest_track()
        print(self.x)


    def depth_cb(self , data):
        self.depth_image = self.cvb.imgmsg_to_cv2(data)

    def rgb_cb(self , data):
        pub_msg  = AnomalyScore()
        pub_msg.image = data
        # hdm_feed =  rospy.wait_for_message('/safe_operation'  , Bool)
        # if not hdm_feed:
            # self.sort_obj.update(self.empty_np) 
        # pub_msg.no_feed = hdm_feed
        pub_msg.data.x = self.x
        pub_msg.data.y = self.y
        pub_msg.data.z = self.z
        self.pub.publish(pub_msg)
    

if __name__ == "__main__":
    rospy.init_node('tracker_node')
    obj = t2ts()
    rospy.spin()
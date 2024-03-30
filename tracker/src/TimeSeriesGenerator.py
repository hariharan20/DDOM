#! /usr/bin/env python
import rospy
from ddom_msgs.msg import Data 
from sensor_msgs.msg import Image
from sort.sort import Sort
import numpy as np
from std_msgs.msg import Bool
# import cv_bridge
import ros_numpy
from tracker.msg import BoundingBox
class PersonTrackLet:
    def __init__(self,  x  ,y , z):
        self.x_array = []
        self.y_array = []
        self.z_array = []
        self.last_depth = float('inf')

    def update(self, x_new , y_new , z_new):
        self.x_array.append(x_new)
        self.y_array.append(y_new)
        self.z_array.append(z_new)
        self.not_known_for = 0
        if (len(self.x_array) > 60):
            self.x_array.pop(0)
            self.y_array.pop(0)
            self.z_array.pop(0)
            self.last_depth = z_new
    
    def get_tracklet(self):
        if len(self.x_array) == 60:
            return [self.x_array ,self.y_array , self.z_array]
        else :
            return None
    
class TimeSeriesGenerator:
    def __init__(self):
        BOUNDING_BOX_TOPIC_NAME = rospy.get_param('bounding_box_topic')
        DEPTH_IMAGE_TOPIC_NAME = rospy.get_param('depth_image_topic')
        TSG_TOPIC_NAME = rospy.get_param('tsg_topic')
        SENTOR_TOPIC_NAME = rospy.get_param('sentor_topic')
        self.PRINCIPAL_POINT_X = rospy.get_param('principal_point_x')
        self.PRINCIPAL_POINT_Y = rospy.get_param('principal_point_y')
        self.FOCAL_LENGTH_X = rospy.get_param('focal_length_x')
        self.FOCAL_LENGTH_Y = rospy.get_param('focal_length_y')
        self.sort_obj = Sort()
        self.known_people = {}
        self.depth_image = None
        self.tracklet_x = None
        self.tracklet_y = None
        self.tracklet_z = None
        self.sentor = True
        self.indexer = 4
        self.EMPTY_NP = np.empty((0 , 5))
        # self.cvb = cv_bridge.CvBridge()
        self.timeseries_msg = Data()
        self.tsg_publisher = rospy.Publisher(TSG_TOPIC_NAME , Data , queue_size =1)
        self.sentor_subscriber = rospy.Subscriber(SENTOR_TOPIC_NAME, Bool , self.sentor_callback)
        self.bounding_box_subscriber = rospy.Subscriber(BOUNDING_BOX_TOPIC_NAME , BoundingBox , self.bounding_box_callback)
        self.depth_subscriber = rospy.Subscriber(DEPTH_IMAGE_TOPIC_NAME , Image , self.depth_image_callback)

    def get_depth(self , xAvg  ,yAvg , depth_map):
        # Zinm = depth_map[yAvg][xAvg]/1000
        # rospy.logerr(str(xAvg)  + '_' + str(yAvg) + ' ' +str(Zinm))
        Zinm =np.average( depth_map[yAvg-self.indexer : yAvg+self.indexer , xAvg-self.indexer:xAvg+ self.indexer]) / 1000
        Xinm = (xAvg  - self.PRINCIPAL_POINT_X) * (Zinm / self.FOCAL_LENGTH_X)
        Yinm = (yAvg  - self.PRINCIPAL_POINT_Y) * (Zinm / self.FOCAL_LENGTH_Y)
        rospy.logerr(str(Xinm)  + '_' + str(Yinm    ) + ' ' +str(Zinm))

        return Xinm , Yinm , Zinm

    def get_nearest_tracklet(self):
        depths = []
        for person in self.known_people.keys():
            depths.append(self.known_people[person].last_depth)
        if not len(depths) == 0:
            minimum_depth_index = np.argmin(np.array(depths))
            # rospy.logerr( depths[minimum_depth_index])
            if not depths[minimum_depth_index] == float('inf'):
                # rospy.logerr(self.known_people[list(self.known_people.keys())[minimum_depth_index]].get_tracklet())
                return self.known_people[list(self.known_people.keys())[minimum_depth_index]].get_tracklet()
            else:
                return False
        else:
            return False
        
    def sort_to_timeseries(self, bb_list):
        if self.sentor:
            depth_map = self.depth_image
            if not depth_map is None:
                tracked_bbs  =self.sort_obj.update(bb_list)
                current_ids  =[]
                for bb in tracked_bbs:
                    person_id = bb[-1]
                    current_ids.append(person_id)
                    centroid_x = round((bb[0] + bb[2])/2)
                    centroid_y = round((bb[1] + bb[3])/2)
                    centroid_x_in_meters, centroid_y_in_meters , centroid_z_in_meters  = self.get_depth(centroid_x , centroid_y  ,depth_map)
                    if person_id in self.known_people.keys():
                        self.known_people[person_id].update(centroid_x_in_meters, centroid_y_in_meters , centroid_z_in_meters)

                    else : 
                        self.known_people[person_id] = PersonTrackLet(centroid_x_in_meters , centroid_y_in_meters , centroid_z_in_meters)
                known_people_ids = list(self.known_people.keys())
                for id in known_people_ids:
                    if id not in current_ids:
                        self.known_people[id].not_known_for +=1
                        if self.known_people[id].not_known_for == 4:
                            del self.known_people[id]

    def bounding_box_callback(self , data):
        bb_list = np.zeros((len(data.ids) , 5))
        bb_list[: , 0] = np.array(data.xmin)
        bb_list[: , 1] = np.array(data.ymin)
        bb_list[: , 2] = np.array(data.xmax)
        bb_list[: , 3] = np.array(data.ymax)
        self.sort_to_timeseries(bb_list)
        nearest_tracklet  = self.get_nearest_tracklet()
        # rospy.logerr(not nearest_tracklet is None)
        # rospy.logerr(nearest_tracklet)
        if (not nearest_tracklet is False) and (not nearest_tracklet is None) :
            # rospy.logerr(type(nearest_tracklet))
            self.timeseries_msg.x = nearest_tracklet[0]
            self.timeseries_msg.y = nearest_tracklet[1]
            self.timeseries_msg.z = nearest_tracklet[2]
            self.tsg_publisher.publish(self.timeseries_msg)

    def sentor_callback(self , data):
        self.sentor = data.data
    
    def depth_image_callback(self , data):
        if self.sentor:
            # self.depth_image = self.cvb.imgmsg_to_cv2(data)
            self.depth_image = ros_numpy.numpify(data)
    
if __name__ == "__main__":
    rospy.init_node('Time_Series_Generator')
    tsg_obj  =TimeSeriesGenerator()
    rospy.spin()


    
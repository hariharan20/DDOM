#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from loss import anomaly_loss
from geometry_msgs.msg import PoseArray
from JustCNN import justCNNEncoder
from std_msgs.msg import Float32
import numpy as np
import torch
import rospkg
from ddom_msgs.msg import AnomalyScore , Data
class ad_ros():
    def __init__(self, radius):
        self.anomaly = anomaly_loss(radius)
        self.model = justCNNEncoder()
        rp = rospkg.RosPack()
        self.device = rospy.get_param('/device')
        self.timeseries_topic = rospy.get_param('/timeseries_topic')
        self.anomaly_score_topic = rospy.get_param('/anomaly_score_topic')
        self.model_directory = rp.get_path('anomaly_detector')   + rospy.get_param('/ad_model_location')
        self.model.load_state_dict(torch.load(self.model_directory))
        self.model.to(self.device)#
        self.model.eval()
        self.Publisher = rospy.Publisher(self.anomaly_score_topic , AnomalyScore , queue_size=10 )
        self.sub = rospy.Subscriber(self.timeseries_topic , AnomalyScore  , self.callback )
        print('------INSIDE AD CLASS-----')
    
    def normalize(self, time_series_data):
        data_normalized = []
        for data_ in time_series_data:
            min_ = min(data_)
            data_ = np.array(data_) - min_
            max_ = max(data_)
            data_normalized.append((np.array(data_)/max_).tolist())
        
        return torch.unsqueeze(torch.Tensor(data_normalized), 0)
 
    def callback(self , data):
        # print('INSIDE CALLBACK')
        print(bool(data.no_feed.data))
        if not bool(data.no_feed.data):
            # print('------- INSIDE MODEL ------')
            input_data = self.normalize([data.data.x , data.data.y , data.data.z])
            output = self.model(input_data)
            anomaly_score = self.anomaly.anomaly_score(output)
            rospy.loginfo(anomaly_score)
        else:
            anomaly_score = 0.0

        data.anomaly_score = Float32(anomaly_score)
        self.Publisher.publish(data)

if __name__ == "__main__":
    rospy.init_node('ad_node')
    obj = ad_ros(0.1)
    rospy.spin()
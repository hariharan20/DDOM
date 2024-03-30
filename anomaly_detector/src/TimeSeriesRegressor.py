#! /usr/bin/env python
import rospy
from ddom_msgs.msg import Data
# from JustCNN import justCNN
from JustCNN_mod import justCNN
from std_msgs.msg import Float32
import numpy as np 
import rospkg
import torch

class TimeSeriesRegressor:
    def __init__(self):
        self.DEVICE = rospy.get_param('device')
        self.TSG_TOPIC_NAME = rospy.get_param('tsg_topic')
        self.TSR_TOPIC_NAME = rospy.get_param('tsr_topic')
        ros_pack = rospkg.RosPack()
        self.model_directory = ros_pack.get_path('anomaly_detector') + rospy.get_param('/tsg_model_location')
        self.model = justCNN((3, 60))
        self.model.load_state_dict(torch.load(self.model_directory , map_location= self.DEVICE))
        self.model.to(self.DEVICE)
        self.model.eval()
        self.tsr_publisher = rospy.Publisher(self.TSR_TOPIC_NAME , Float32 , queue_size=10)
        self.tsg_subscriber = rospy.Subscriber(self.TSG_TOPIC_NAME , Data , self.tsg_callback)

    def normalize(self , time_series_data):
        data_normalized = []
        min_ = 100
        max_ = 0
        data_minus_min = []
        for data_ in time_series_data:
            if min(data_) < min_:
                min_ = min(data_)
            # if max(data_) > max_:
                # max_ = max(data_)
        for data__ in time_series_data:
            data_minus_min.append((np.array(data__) - min_))
        for data_ in data_minus_min:
            if max(data_) > max_:
                max_ = max(data_)
        for data__ in data_minus_min:
            data_normalized.append((np.array(data__) / max_).tolist()) 
        return torch.unsqueeze(torch.Tensor(data_normalized)  , 0)

    def tsg_callback(self , data):
        input_data = self.normalize([data.x ,  data.y , data.z])
        output = self.model(input_data)
        self.tsr_publisher.publish(Float32(output))

if __name__ == "__main__":
    rospy.init_node('Time_Series_Regressor')
    tsr_obj = TimeSeriesRegressor()
    rospy.spin()


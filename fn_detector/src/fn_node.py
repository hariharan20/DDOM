#! /usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32 , Bool
import cv_bridge
from torch import nn 
import torch
from torchvision import models
import ros_numpy
from torchvision.transforms import Resize
from ddom_msgs.msg import AnomalyScore
import rospkg
class fn_node:
    def __init__(self):
        self.fn_topic = rospy.get_param('/false_negative_topic')
        self.anomaly_score_topic = rospy.get_param('/anomaly_score_topic')
        self.pub = rospy.Publisher(self.fn_node , AnomalyScore , queue_size=10 )
        self.transform = Resize(256)
        self.model_path = rospy.get_param('/fnd_model_location')
        self.model = self.load_model()
        self.sub1 = rospy.Subscriber(self.anomaly_score_topic , AnomalyScore , self.cb1)
    
    def load_model(self):
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(1024 , 512)
        model.classifier.add_module('a' ,nn.Hardswish())
        model.classifier.add_module('b' ,nn.Linear(512  , 256))
        model.classifier.add_module('c' ,nn.Hardswish() )
        model.classifier.add_module('d' ,nn.Linear(256 , 32))
        model.classifier.add_module('e' ,nn.Hardswish())
        model.classifier.add_module('f' ,nn.Linear(32 , 1))
        model.classifier.add_module('act' , nn.Sigmoid())
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def preprocess(self, image):
        image = torch.Tensor(image/ 255.0)
        image = image.permute(2,0,1)
        image = torch.unsqueeze(image , 0)
        image = self.transform(image)
        return image
    
    def cb1(self, data):
        if data.no_feed:
            if bool(data.image):
                data_np = ros_numpy.numpify(data.image)
                data_np = self.preprocess(data_np)
                pred = self.model(data_np)
                if pred > 0.17:
                    fn = True
                else :
                    fn = False

        else: 
            fn = False
        data.false_negative = fn
        self.pub.publish(data)

if __name__ == "__main__":
    rospy.init_node('fn_node')
    obj = fn_node()
    rospy.spin()
    

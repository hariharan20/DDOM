#! /usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
# import cv_bridge
import ros_numpy
from torch import nn
import torch  
from torchvision import models
from torchvision.transforms import Resize
import rospkg
class FalseNegativeDetector:
    def __init__(self):
        FND_TOPIC_NAME = rospy.get_param('fnd_topic')
        RGB_IMAGE_TOPIC_NAME = rospy.get_param('rgb_image_topic')
        SENTOR_TOPIC_NAME  =rospy.get_param('sentor_topic')
        self.DEVICE = rospy.get_param('device')
        self.threshold = rospy.get_param('fnd_threshold')
        self.transform = Resize(256)
        ros_pack = rospkg.RosPack()
        self.model_directory =  ros_pack.get_path('fn_detector') + rospy.get_param('/fnd_model_location')
        self.model = self.load_model()
        self.sentor = True
        # self.cvb = cv_bridge.CvBridge()
        self.fnd_publisher = rospy.Publisher(FND_TOPIC_NAME , Bool, queue_size=10)
        self.sentor_subscriber = rospy.Subscriber(SENTOR_TOPIC_NAME , Bool , self.sentor_callback)
        self.rgb_image_subscriber  = rospy.Subscriber(RGB_IMAGE_TOPIC_NAME , Image , self.rgb_image_callback)
    
    def load_model(self):
        model = models.mobilenet_v3_small()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features , 512)
        model.classifier.add_module('a' ,nn.Hardswish())
        model.classifier.add_module('b' ,nn.Linear(512  , 256))
        model.classifier.add_module('c' ,nn.Hardswish() )
        model.classifier.add_module('d' ,nn.Linear(256 , 32))
        model.classifier.add_module('e' ,nn.Hardswish())
        model.classifier.add_module('f' ,nn.Linear(32 , 1))
        model.classifier.add_module('act' , nn.Sigmoid())
        model.load_state_dict(torch.load(self.model_directory , map_location=self.DEVICE))
        return model
    
    def preprocess(self, image):
        image = torch.Tensor(image/255.0)
        image = image.permute(2 , 0 , 1)
        image = torch.unsqueeze(image , 0)
        image = self.transform(image)
        return image

    def sentor_callback(self, data):
        self.sentor = data.data

    def rgb_image_callback(self  , data):
        if not self.sentor:
            # image_np = self.cvb.imgmsg_to_cv2(data)
            image_np = ros_numpy.numpify(data)
            image_np = self.preprocess(image_np)
            model_output = self.model(image_np)
            # rospy.logerr(model_output)
            if model_output > self.threshold:
                false_negative = False
            else : 
                false_negative = True
            self.fnd_publisher.publish(Bool(false_negative))

if __name__ == "__main__":
    rospy.init_node('False_Negative_Detector')
    fnd_obj = FalseNegativeDetector()
    rospy.spin()
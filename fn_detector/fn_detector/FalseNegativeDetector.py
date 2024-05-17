import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from torch import nn
import torch
from torchvision import models
from torchvision.transforms import Resize
import rospkg

class FalseNegativeDetector(Node):
	def __init__(self):
		super().__init__('FalseNegativeDetector')
		FND_TOPIC_NAME = self.get_parameter('fnd_topic')
		RGB_TOPIC_NAME = self.get_paramter('rgb_image_topic')
		SENTOR_TOPIC_NAME = self.get_parameter('sentor_topic')
		self.DEVICE = self.get_parameter('device')
		self.threshold  =self.get_parameter('fnd_threshold')
		self.transform = Resize(256)
		ros_pack = rospkg.RosPack()
		self.model_directory = ros_pack.get_path('fn_detector') + self.get_parameter('fnd_model_location')
		self.model = self.load_model()
		self.sentor = True
		self.fnd_publisher = self.create_publisher(Bool , FND_TOPIC_NANE , 10)
		self.sentor_subscriber = self.create_subscriber(Image , 

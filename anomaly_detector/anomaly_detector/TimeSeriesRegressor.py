import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from JustCNN_mode import JustCNN
import numpy as np
from ddom_msgs.msg import Data

class TimeSeriesRegressor(Node):
	def __init__(self):
		super.__init__('TimeSeriesRegressor')
		self.TSG_TOPIC_NAME = self.get_parameter('tsg_topic')
		self.TSR_TOPIC_NAME = self.get_parameter('tsr_topic')
		self.tsr_publisher = self.create_publisher(Float32 , self.TSR_TOPIC_NAME , 10)
		self.tsg_subscriber = self.create_subscriber(Data , self.TSG_TOPIC_NAME , self.tsg_callback)
	def normalize(self, time_series_data):
		data_normalized = []
		min_ = 100
		max_ = 0
		data_minus_min = []
		for data_ in time_series_data:
			if min(data_) < min_ :
				min_ = min(data_)
		for data__ in time_series_data:
			data_minus_min.append((np.array(data__) - min_))
		for data_ in data_minus_min:
			if max(data_) > max_:
				max_ = max(data_)
		for data__ in data_minus_min:
			data_normalized.append((np.array(data__)/max_).tolist())
		return torch.unsqueeze(torch.Tensor(data_normalized) , 0)
	def tsg_callback(self, data):
		input_data =  self.normalize([data.x , data.y  ,data.z])
		output = self.model(input_data)
		self.tsr_publisher.publish(Float32(output))






def main(args=None):
	rclpy.init(args=args)
	TimeSeriesRegressorNode = TimeSeriesRegressor()
	rclpy.spin(TimeSeriesRegressorNode)

if __name__=="__main__":
	main()

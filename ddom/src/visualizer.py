#!/usr/bin/env python
from ddom_msgs.msg import AnomalyScore
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
class vis:
    def __init__(self):
        self.cvb = CvBridge()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.white = (255, 255, 255)
        self.red = (50, 50 , 255 )
        self.green = (50 ,255 , 50)
        self.line_type = 2
        self.pub = rospy.Publisher('/DDOM_vis' , Image , queue_size=10)
        self.sub = rospy.Subscriber('/fn_topic' , AnomalyScore , self.cb)
    
    def cb(self,data):
        cv_image = self.cvb.imgmsg_to_cv2(data.image , 'bgr8')
        anomaly_text = 'Anomaly Score = ' + str(round(float(data.anomaly_score.data) ,2))
        # rospy.logerr(anomaly_text)
        if bool( data.false_negative.data )== True:
            fn_text = 'False Negative'
            cv2.putText(cv_image , fn_text , (50  , 200) , self.font , self.font_scale  , self.red , self.line_type)
        else :
            if not bool(data.no_feed.data):
                fn_text = 'True Positive'
            else:
                fn_text = 'No False Negative'
            cv2.putText(cv_image , fn_text , (50  , 200) , self.font , self.font_scale  , self.green , self.line_type)
        
        

        cv2.putText(cv_image , anomaly_text , (50 ,  50) , self.font , self.font_scale  , self.white , self.line_type)
        self.pub.publish(self.cvb.cv2_to_imgmsg(cv_image , "bgr8") )

if __name__ == '__main__':
    rospy.init_node('vis')
    obj  = vis()
    rospy.spin()

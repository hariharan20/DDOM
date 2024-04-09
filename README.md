# DDOM (Data Driven Online Monitoring) System
The ROS Package is a framework which can be used for monitoring Perception Modules to measure the uncertainty of detection.

The Framework uses two models for monitoring the outputs from a Detection Model

1) An Anomaly Detector for estimated the Deviation in the Detections (essentially a measure of precision)
2) A False Negative Detector, based on *ALERT* framework.

The Framework is built on top of **SORT** Tracking and **SENTOR** ros package. 

The **SORT** tracker is used to genrate tracklets of objects.


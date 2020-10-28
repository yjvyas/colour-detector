#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import cv2
import sys
import numpy as np
from cv_bridge import CvBridge
# from sklearn.cluster import DBSCAN

class ColourDetector(DTROS):

    def __init__(self, node_name, vehicle_name, colour='blue'):
        # initialize the DTROS parent class
        super(ColourDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.cap = cv2.VideoCapture(2)
        self.sub = rospy.Subscriber('/{}/image_extractor/images/compressed'.format(vehicle_name), CompressedImage, self.callback)
        self.pub = rospy.Publisher('~colour_detector/compressed', CompressedImage, queue_size=10)
        
        if colour not in ['red', 'blue']:
            rospy.loginfo('Colour {} not supported, defaulting to blue'.format(colour))
            colour = blue

        if colour == 'blue':
            self.colour_lower = np.array([35, 140, 60]) 
            self.colour_upper = np.array([255, 255, 180])
        elif colour == 'red':
            self.colour_lower = np.array([160, 100, 100])
            self.colour_upper = np.array([179, 255, 255])
        
        self.cvbr = CvBridge()

    def callback(self, msg):
        image = self.cvbr.compressed_imgmsg_to_cv2(msg)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, self.colour_lower, self.colour_upper)
        img_fil = cv2.bitwise_and(image_hsv, image_hsv, mask = mask)
        [x, y] = np.meshgrid(np.arange(0, img_fil.shape[1]), np.arange(0, img_fil.shape[0]))
        print(np.stack([x, y, img_fil], axis=2))
        img_coords = np.reshape(np.stack([x, y, img_fil], axis=2), [-1, 3])
        

    def run(self):
        # publish message every 1 second
        rate = rospy.Rate(30) # 30Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                cmprsmsg = self.cvbr.cv2_to_compressed_imgmsg(frame)  # Convert the image to a compress message
                self.pub.publish(cmprsmsg)
                rospy.loginfo('Publishing image')
                rate.sleep()

if __name__ == '__main__':
    # create the node
    args = rospy.myargv(argv=sys.argv)
    
    node = ColourDetector(node_name='colour_detector', vehicle_name=args[1], colour=args[2])
    # run node
    # keep spinning
    rospy.spin()

#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append('/home/msa/catkin_ws/src/sign_language_translation/models')
# print(sys.path)
from model import Net

# Load the trained model
model = Net()
model.load_state_dict(torch.load('/home/msa/catkin_ws/src/sign_language_translation/trained_model/model_trained.pt'))
model.eval()

# Define the gesture-to-sign mapping
signs = {'for', 'goodluck', 'hello', 'iloveyou', 'need', 'no', 'please', 'sorry', 'thanks', 'yes'}

class GestureRecognitionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.sign_pub = rospy.Publisher("/sign_language/gesture", String, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocess the image
        img = cv_image[20:250, 20:250]
        res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res = np.reshape(res, (1, 1, 28, 28)) / 255.0
        res = torch.from_numpy(res).float()

        # Perform gesture recognition
        with torch.no_grad():
            out = model(res)
            probs, label = torch.topk(out, 25)
            probs = F.softmax(probs, dim=1)
            pred = out.max(1, keepdim=True)[1]

        # Determine the detected gesture
        if float(probs[0, 0]) < 0.4:
            detected_sign = "Sign not detected"
        else:
            detected_sign = signs.get(str(int(pred)), "Unknown sign")

        # Publish the detected gesture
        rospy.loginfo("Detected Sign: {}".format(detected_sign))
        self.sign_pub.publish(detected_sign)

def main():
    rospy.init_node('gesture_recognition_node', anonymous=True)
    grn = GestureRecognitionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down gesture recognition node.")

if __name__ == '__main__':
    main()



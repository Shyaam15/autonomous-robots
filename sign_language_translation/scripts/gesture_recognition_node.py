#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv2
import torch
import numpy as np
import sys

sys.path.append('/home/msa/catkin_ws/src/sign_language_translation/models')
# print(sys.path)
from model import Net

class SignLanguageTranslator:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('sign_language_translator', anonymous=True)
        
        # Publisher for recognized gestures
        self.sign_pub = rospy.Publisher("/sign_language/gesture", String, queue_size=10)
        
        # Load the trained model
        self.model = Net()
        self.model.load_state_dict(torch.load('/home/msa/catkin_ws/src/sign_language_translation/trained_model/model_trained.pt'))
        self.model.eval()
        # Dictionary to map output labels to sign language gestures
        self.signs = {'0': 'okay', '1': 'hello', '2': 'stop', '3': 'yes', '4': 'pray', '5': 'no', '6': 'sorry', '7': 'iloveyou', '8': 'goodluck', '9': 'thanks'}
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Camera could not be opened.")
            return
        
        rospy.loginfo("Sign Language Translator node started.")

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                continue
            
            # Show the image for debugging purposes
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)
            
            # Process the image and recognize gestures
            gesture = self.recognize_gesture(frame)
            rospy.sleep(0.5)
            
            # Publish the recognized gesture
            if gesture:
                self.sign_pub.publish(gesture)
                print(gesture)
            
            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("Shutting down Sign Language Recognizer node.")
                break
        
        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

    def recognize_gesture(self, image):
        # Preprocess the image (e.g., resize, convert to grayscale)
        img = cv2.resize(image[20:250, 20:250], (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (1, 1, 28, 28)) / 255.0
        img_tensor = torch.from_numpy(img).float()
        
        # Run the image through the model
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Get the predicted gesture
        _, pred = torch.max(output, 1)
        gesture = self.signs.get(str(pred.item()))
        
        return gesture

if __name__ == '__main__':
    try:
        translator = SignLanguageTranslator()
        translator.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        translator.cap.release()
        cv2.destroyAllWindows()

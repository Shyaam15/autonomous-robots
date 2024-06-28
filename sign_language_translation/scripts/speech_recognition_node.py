#!/usr/bin/env python

import rospy
import speech_recognition as sr
from std_msgs.msg import String

if __name__ == "__main__":

    rospy.init_node('speech_recognition_node')
    rospy.loginfo('Speech Recognition Node has been started')

    recognizer = sr.Recognizer()
    pub = rospy.Publisher('recognized_speech', String, queue_size=10)

    with sr.Microphone() as source:
        while not rospy.is_shutdown():
            print("Say something!")
            audio = recognizer.listen(source)

            try:
                recognized_text = recognizer.recognize_google(audio)
                rospy.loginfo("You said: %s", recognized_text)
                pub.publish(recognized_text)
                print(recognized_text)
            except sr.UnknownValueError:
                rospy.loginfo("Could not understand the audio")
            except sr.RequestError as e:
                rospy.loginfo("Could not request results; {0}".format(e))

    rospy.loginfo("Exit now")

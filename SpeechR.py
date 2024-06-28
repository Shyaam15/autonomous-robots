#!/usr/bin/env python
#import os
#import sys

#sys.stderr = open(os.devnull, 'w')
import rospy
import speech_recognition as sr

if __name__ == "__main__":

	rospy.init_node('Speech_Recognition_Node')

	rospy.loginfo('This node has been started')

	recognizer =sr.Recognizer()

	with sr.Microphone() as source:

    while not rospy.is_shutdown():
      print("Say something!")
      audio = recognizer.listen(source)

      try:
        print("You Said:" + recognizer.recognize_google(audio))
      except sr.UnknownValueError:
        print("could not understand")
      except sr.RequestError as e:
        print("could not request results; {0}".format(e))

  rospy.loginfo("Exit now")

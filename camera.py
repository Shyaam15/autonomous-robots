import os
import cv2
import time
import uuid

IMAGE_PATH='C:\\Users\\user\\Desktop\\Code'

labels=['Hello','Yes','Stop','Thanks','IloveYou','Okay','Goodluck','Sorry','No','Pray']

number_of_images=100

label = 'pray'
img_path = os.path.join(IMAGE_PATH, label)
cap=cv2.VideoCapture(0)
print('Collecting images for {}'.format(label))
time.sleep(2)
for imgnum in range(number_of_images):
    ret,frame=cap.read()
    imagename=os.path.join(IMAGE_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
    cv2.imwrite(imagename,frame)
    cv2.imshow('frame',frame)
    time.sleep(0.5)
cap.release()

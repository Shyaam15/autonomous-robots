# autonomous-robots

## 1. Traverse to catkin_ws/src folder 
cd ~/catkin_ws/src

## 2. Clone the repo
git clone https://github.com/Shyaam15/autonomous-robots.git

## 3. Go inside the folder
cd autonomous-robots

## 4. Install the dependencies and required modules
chmod +x modules_dependencies_installations.sh \
./modules_dependencies_installations.sh

## 5. Put the project folder inside src
mv sign_language_translation ..

## 6. Delete the repo
cd .. \
rm -rf autonomous-robots

## 6. Turn the nodes executable
chmod +x ~/catkin_ws/src/sign_language_translation/scripts/*

## 7. Traverse to catkin_ws folder
cd ~/catkin_ws

## 8. Catkin_make to update the catkin workspace
catkin_make \
source devel/setup.bash

## 9. roscore to start the ros environment
roscore

## 10. Launch the nodes in two new terminals simultaneously or one after the other
roslaunch sign_language_translation gesture_recognition.launch \
roslaunch sign_language_translation speech_recognition.launch

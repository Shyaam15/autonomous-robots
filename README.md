# autonomous-robots

## 1. Traverse to catkin_ws/src folder 
cd ~/catkin_ws/src

## 2. Clone the repo
git clone https://github.com/Shyaam15/autonomous-robots.git

## 3. Install the dependencies and required modules
./modules_dependencies_installations.sh

## 4. Delete unnecessary files
rm README.md camera.py cnn_torch.py file_splitter.py modules_dependencies_installations.sh

## 5. Turn the nodes executable
chmod +x ~/catkin_ws/src/sign_language_translation/scripts/*

## 6. Traverse to catkin_ws folder
cd ~/catkin_ws

## 7. Catkin_make to update the catkin workspace
catkin_make\
source devel/setup.bash

## 8. roscore to start the ros environment
roscore

## 9. Launch the nodes in two new terminals
roslaunch sign_language_translation gesture_recognition.launch\
roslaunch sign_language_translation speech_recognition.launch

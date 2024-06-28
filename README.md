# autonomous-robots

## 1. Traverse to catkin_ws/src folder 
cd ~/catkin_ws/src

## 2. Clone the repo
git clone https://github.com/Shyaam15/autonomous-robots.git

## 3. Rename the repo and go inside the folder
mv autonomous-robots sign_language_translation \
cd sign_language_translation

## 4. Install the dependencies and required modules
chmod +x modules_dependencies_installations.sh \
./modules_dependencies_installations.sh

## 5. Delete unnecessary files
rm README.md camera.py cnn_torch.py file_splitter.py modules_dependencies_installations.sh

## 6. Turn the nodes executable
chmod +x ~/catkin_ws/src/sign_language_translation/scripts/*

## 7. Traverse to catkin_ws folder
cd ~/catkin_ws

## 8. Catkin_make to update the catkin workspace
catkin_make \
source devel/setup.bash

## 9. roscore to start the ros environment
roscore

## 10. Launch the nodes in two new terminals
roslaunch sign_language_translation gesture_recognition.launch \
roslaunch sign_language_translation speech_recognition.launch

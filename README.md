# autonomous-robots

## 1. traverse to catkin_ws/src folder 
cd ~/catkin_ws/src

## 2. clone the repo
git clone https://github.com/Shyaam15/autonomous-robots.git

## 3. Install the dependencies and required modules
./modules_dependencies_installations.sh

## 4. Traverse to catkin_ws folder
cd ~/catkin_ws

## 5. Catkin_make to update the catkin workspace
catkin_make

## 6. Launch the nodes
roslaunch sign_language_translation translation.launch

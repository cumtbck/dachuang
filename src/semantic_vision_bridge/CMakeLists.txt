cmake_minimum_required(VERSION 3.0.2)
project(semantic_vision_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/rgb_sender.py
  scripts/semantic_solver.py
  scripts/traffic_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 2.8.3)
project(grasping_benchmarks_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cmake_modules eigen_conversions message_generation roscpp rospy tf2_ros sensor_msgs std_msgs geometry_msgs pcl_conversions)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  BenchmarkGrasp.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  UserCmd.srv
  GraspPlanner.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES sensor_msgs std_msgs geometry_msgs)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${catkin_INCLUDE_DIRS})

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS cmake_modules message_runtime eigen_conversions roscpp rospy tf2_ros sensor_msgs std_msgs geometry_msgs
)

option(GPD_ROS "Compile gpd ros" OFF)
if(GPD_ROS)
  add_subdirectory(gpd_ros)
endif()

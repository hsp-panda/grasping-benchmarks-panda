# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## System dependencies are found with CMake's conventions
set(INSTALL_DIR $ENV{WS_INSTALL})
find_library(GPD_LIB NAMES gpd PATHS /usr/local/lib ${INSTALL_DIR} PATH_SUFFIXES lib NO_DEFAULT_PATH)
if (GPD_LIB)
  message(STATUS "Library GPD found in ${GPD_LIB}")
else()
  message(FATAL_ERROR "Library GPD not found")
endif()

set(GPD_LIB_INCLUDE_DIR ${INSTALL_DIR}/include)
include_directories(${GPD_LIB_INCLUDE_DIR})
message(STATUS "gpd_include_dir: ${GPD_LIB_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++14")

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

## Declare a C++ library
add_library(${PROJECT_NAME}_grasp_messages src/gpd_ros/grasp_messages.cpp)
add_dependencies(${PROJECT_NAME}_grasp_messages ${catkin_EXPORTED_TARGETS})
add_library(${PROJECT_NAME}_grasp_plotter src/gpd_ros/grasp_plotter.cpp)

## Declare a C++ executable
add_executable(${PROJECT_NAME}_gpd_grasp_planner_service src/gpd_ros/gpd_grasp_planner_service.cpp)

## Rename C++ executable without prefix
set_target_properties(${PROJECT_NAME}_gpd_grasp_planner_service PROPERTIES OUTPUT_NAME gpd_grasp_planner_service PREFIX "")

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_gpd_grasp_planner_service
                      ${PROJECT_NAME}_grasp_messages
                      ${PROJECT_NAME}_grasp_plotter
                      ${GPD_LIB}
                      ${PCL_LIBRARIES}
                      ${catkin_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_grasp_plotter
                      ${GPD_LIB}
                      ${catkin_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_grasp_messages
                      ${GPD_LIB}
                      ${catkin_LIBRARIES})

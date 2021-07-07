/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef GRASP_DETECTION_SERVER_H_
#define GRASP_DETECTION_SERVER_H_


// ROS
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>

// GPD
#include <gpd/util/cloud.h>
#include <gpd/grasp_detector.h>

// BENCHMARK SERVICE/MSG
#include <grasping_benchmarks_ros/GraspPlanner.h>
#include <grasping_benchmarks_ros/BenchmarkGrasp.h>

// this project (headers)
#include <gpd_ros/grasp_messages.h>
#include <gpd_ros/grasp_plotter.h>

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudPointNormal;

class GpdGraspPlannerService
{
public:

  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GpdGraspPlannerService(ros::NodeHandle& node, std::string config_file, std::vector<float> grasp_pose_offset, std::string grasp_service_name="",
                      bool publish_rviz=false, std::string grasp_publisher_name="");

  /**
   * \brief Destructor.
  */
  ~GpdGraspPlannerService()
  {
    delete cloud_camera_;
    delete grasp_detector_;
    delete rviz_plotter_;
  }

  /**
   * \brief Service callback for detecting grasps.
   * \param req the service request
   * \param res the service response
   */
  bool planGrasps(grasping_benchmarks_ros::GraspPlanner::Request& req,
                    grasping_benchmarks_ros::GraspPlanner::Response& res);


private:

  ros::Publisher grasps_pub_; ///< ROS publisher for grasp list messages

  std_msgs::Header cloud_camera_header_; ///< stores header of the point cloud
  std::string frame_; ///< point cloud frame

  gpd::GraspDetector* grasp_detector_; ///< used to run the grasp pose detection
  gpd::util::Cloud* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals
  GraspPlotter* rviz_plotter_; ///< used to plot detected grasps in rviz

  bool use_rviz_; ///< if rviz is used for visualization instead of PCL
  std::vector<double> workspace_; ///< workspace limits
  Eigen::Vector3d view_point_; ///< (input) view point of the camera onto the point cloud

  Eigen::Vector3d grasp_pose_offset_; ///< offset traslation to apply to the grasp pose (in its ref frame) before returning it to the benchmark
};

#endif /* GRASP_DETECTION_SERVER_H_ */

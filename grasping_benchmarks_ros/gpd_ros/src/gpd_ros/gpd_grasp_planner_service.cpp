#include <gpd_ros/gpd_grasp_planner_service.h>


GpdGraspPlannerService::GpdGraspPlannerService(ros::NodeHandle& node, std::string config_file, std::string grasp_service_name,
                                    bool publish_rviz, std::string grasp_publisher_name)
{
  cloud_camera_ = NULL;

  // set camera viewpoint to default origin
  view_point_ << 0.0, 0.0, 0.0;

  grasp_detector_ = new gpd::GraspDetector(config_file);

  // Visualize planned grasps in Rviz
  if (publish_rviz == true)
  {
    rviz_plotter_ = new GraspPlotter(node, grasp_detector_->getHandSearchParameters().hand_geometry_);
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // Advertise ROS topic for detected grasps.
  if (!grasp_publisher_name.empty())
  {
    grasps_pub_ = node.advertise<geometry_msgs::PoseStamped>(grasp_publisher_name, 10);
  }

  node.getParam("workspace", workspace_); // Do not know where this wariable is used

}

bool GpdGraspPlannerService::planGrasps(grasping_benchmarks_ros::GraspPlannerCloud::Request& req,
                                        grasping_benchmarks_ros::GraspPlannerCloud::Response& res)
{
  ROS_INFO("Received service request from benchmark...");

  // 1. Initialize cloud camera.
  cloud_camera_ = NULL;

  const sensor_msgs::PointCloud2 & cloud_ros = req.cloud;

  // Set view points.
  Eigen::Matrix3Xd view_points(3,1);
  view_points.col(0) = view_point_;
  view_points.col(0) << req.view_point.position.x, req.view_point.position.y, req.view_point.position.z;

  // Set point cloud.
  if (cloud_ros.fields.size() == 6 && cloud_ros.fields[3].name == "normal_x"
    && cloud_ros.fields[4].name == "normal_y" && cloud_ros.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(cloud_ros, *cloud);

    cloud_camera_ = new gpd::util::Cloud(cloud, 0, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(req.cloud, *cloud);

    cloud_camera_ = new gpd::util::Cloud(cloud, 0, view_points);
  }

  frame_ = cloud_ros.header.frame_id;

  ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points");

  // 2. Preprocess the point cloud.
  grasp_detector_->preprocessPointCloud(*cloud_camera_);

  // 3. Detect grasps in the point cloud.
  std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps = grasp_detector_->detectGrasps(*cloud_camera_);

  if (grasps.size() > 0)
  {
    // Visualize the detected grasps in rviz.
    if (use_rviz_)
    {
      rviz_plotter_->drawGrasps(grasps, frame_);
    }

    cloud_camera_header_.frame_id = cloud_ros.header.frame_id;

    // 4. Create benchmark grasp reply.
    grasping_benchmarks_ros::BenchmarkGrasp bench_grasp = GraspMessages::convertToBenchmarkGraspMsg(*grasps[0], cloud_camera_header_);

    // Publish grasp on topic
    grasps_pub_.publish(bench_grasp.pose);

    res.grasp = bench_grasp;
    ROS_INFO_STREAM("Detected grasp.");
    return true;
  }

  ROS_WARN("No grasps detected!");
  return false;
}

int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "Gpd_Grasp_Planner");
  ros::NodeHandle node("~");

  std::string grasp_service_name;
  std::string grasp_publisher_name;
  std::string config_file;
  node.param<std::string>("grasp_planner_service_name", grasp_service_name, "gpd_grasp_planner_service");
  node.param<std::string>("grasp_publisher_name", grasp_publisher_name, "");

  node.getParam("config_file", config_file);

  bool publish_rviz;
  node.param<bool>("publish_rviz", publish_rviz, false);

  GpdGraspPlannerService grasp_detection_server(node, config_file, grasp_service_name);

   // Setup grasp service
  ros::ServiceServer service = node.advertiseService(grasp_service_name, &GpdGraspPlannerService::planGrasps,
                          &grasp_detection_server);

  ROS_INFO("GPD Grasp planner service is waiting for a point cloud ...");

  ros::spin();

  return 0;
}

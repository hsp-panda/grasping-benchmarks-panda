#include <iostream>
#include <sstream>

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/String.h>

// PCL types include
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

//  Plane sample consensus segmentation includes
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

//  Point cloud filters
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

ros::Publisher plane_pub;
ros::Publisher objects_pub;
ros::Publisher coeff_pub;

pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

//  Create segmentation object
pcl::SACSegmentation<PointC> seg;

int sign (const float& arg)
{
  return (arg > 0) - (arg < 0);
}

void splitPointCloudPlane (const PointCloudC& pc_to_split, const PointCloudC::Ptr pc_target, const pcl::ModelCoefficients::Ptr plane_coefficients)
{

  //  Translate to eigen coeff vector
  Eigen::Vector4f plane_coefficients_eigen;
  plane_coefficients_eigen[0] = plane_coefficients->values[0];
  plane_coefficients_eigen[1] = plane_coefficients->values[1];
  plane_coefficients_eigen[2] = plane_coefficients->values[2];
  plane_coefficients_eigen[3] = plane_coefficients->values[3];

  //  Compute where the camera is wrt the segmentation plane
  Eigen::Vector4f camera_position = pc_to_split.sensor_origin_;
  camera_position[3] = 1.0; //  homogeneous coordinates
  auto camera_position_proj = plane_coefficients_eigen.dot(camera_position);
  int camera_position_sign = sign(camera_position_proj);

  //  For each point, compute which side of the plane it is on
  for (PointC point : pc_to_split.points)
  {
    Eigen::Vector4f point_eigen = point.getVector4fMap();
    point_eigen[3] = 1.0;

    //  If on the same side as camera, accumulate in target point cloud
    if (sign(point_eigen.dot(plane_coefficients_eigen) * camera_position_sign) > 0)
    {
      pc_target->push_back(point);
    }
  }
}

void cloud_cb (const sensor_msgs::PointCloud2& input_msg)
{
  // Convert to PCL container
  PointCloudC::Ptr full_cloud(new PointCloudC());
  pcl::fromROSMsg(input_msg, *full_cloud);

  ROS_DEBUG("PC received! Size: %d", full_cloud->size());

  //  Crop the point cloud
  PointCloudC::Ptr cropped_cloud(new PointCloudC());

  double min_x, min_y, min_z, max_x, max_y, max_z;
  ros::param::param("crop_min_x", min_x, -0.5);
  ros::param::param("crop_min_y", min_y, -0.5);
  ros::param::param("crop_min_z", min_z, 0.2);
  ros::param::param("crop_max_x", max_x, 0.5);
  ros::param::param("crop_max_y", max_y, 0.5);
  ros::param::param("crop_max_z", max_z, 0.9);
  Eigen::Vector4f min_pt(min_x, min_y, min_z, 1);
  Eigen::Vector4f max_pt(max_x, max_y, max_z, 1);

  pcl::CropBox<PointC> cropper;

  cropper.setInputCloud(full_cloud);
  cropper.setMin(min_pt);
  cropper.setMax(max_pt);
  cropper.filter(*cropped_cloud);

  ROS_DEBUG("Cropped to %d points", cropped_cloud->size());

  // Segmentation
  seg.setInputCloud(cropped_cloud);
  double dist_thresh;
  ros::param::param("plane_distance_threshold", dist_thresh, seg.getDistanceThreshold());
  seg.setDistanceThreshold(dist_thresh);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() > 0)
  {
    // Extract indices and build a new point cloud
    PointCloudC::Ptr plane_cloud(new PointCloudC);
    PointCloudC::Ptr plane_outliers_cloud(new PointCloudC);
    pcl::ExtractIndices<PointC> extractor(true);  //  true means we can recover the outliers
    extractor.setInputCloud(cropped_cloud);
    extractor.setIndices(inliers);
    extractor.filter(*plane_cloud);

    // Recover points that do not belong in the plane
    pcl::PointIndices::Ptr outliers(new pcl::PointIndices);
    extractor.getRemovedIndices(*outliers);
    extractor.setIndices(outliers);
    extractor.filter(*plane_outliers_cloud);

    // Split the point cloud according to the plane
    PointCloudC::Ptr objects_cloud(new PointCloudC);
    splitPointCloudPlane(*plane_outliers_cloud, objects_cloud, coefficients);

    ROS_DEBUG("Object: %d", objects_cloud->size());

    // Create a container for the data.
    sensor_msgs::PointCloud2 output_plane_msg;
    pcl::toROSMsg(*plane_cloud, output_plane_msg);
    //output_plane_msg.header = input_msg.header;

    sensor_msgs::PointCloud2 output_objects_msg;
    pcl::toROSMsg(*objects_cloud, output_objects_msg);
    output_objects_msg.header = input_msg.header;

    // Publish the data.
    pcl_msgs::ModelCoefficients ros_coefficients;
    pcl_conversions::fromPCL(*coefficients, ros_coefficients);
    coeff_pub.publish(ros_coefficients);
    plane_pub.publish(output_plane_msg);
    objects_pub.publish(output_objects_msg);

    ROS_DEBUG("PC segmented and rerouted. Inliers: %d", inliers->indices.size());

  }
  else
  {
    ROS_INFO("No plane was found");
  }
  
  return;

}

int main (int argc, char** argv) 
{
  // Initialize ROS
  ros::init (argc, argv, "tabletop_segment");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("input_cloud", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  plane_pub =   nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1, true);
  objects_pub = nh.advertise<sensor_msgs::PointCloud2>("objects_cloud", 1, true);
  coeff_pub =   nh.advertise<pcl_msgs::ModelCoefficients>("output_coefficients", 1, true);

  // Initialize the segmentation object
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);

  // Initialize the node parameters
  nh.setParam("crop_min_x", -0.5);
  nh.setParam("crop_min_y", -0.5);
  nh.setParam("crop_min_z", 0.2);
  nh.setParam("crop_max_x", 0.5);
  nh.setParam("crop_max_y", 0.5);
  nh.setParam("crop_max_z", 0.9);
  nh.setParam("plane_distance_threshold", seg.getDistanceThreshold());

  // Info 
  ROS_INFO("Node initialized, spinning up");

  // Spin
  ros::spin ();
}
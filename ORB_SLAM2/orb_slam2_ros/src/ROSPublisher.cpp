//
// Created by sebastiano on 8/18/16.
//

#include <include/ROSPublisher.h>
#include "FrameDrawer.h"
#include "Tracking.h"
#include "LoopClosing.h"
#include "utils.h"
#include "System.h"

#include <thread>
#include <sstream>
#include <cassert>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Path.h>

#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/UInt32.h>

#include <orb_slam2_ros/ORBState.h>
#include <cv_bridge/cv_bridge.h>

#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>

#include <chrono>
#include <stdint.h>

using namespace ORB_SLAM2;

ROSPublisher::ROSPublisher(Map *map, double frequency, ros::NodeHandle nh) :
    IMapPublisher(map),
    drawer_(GetMap()),
    nh_(std::move(nh)),
    name_of_node_(ros::this_node::getName()),
    pub_rate_(frequency),
    lastBigMapChange_(-1),
    octomap_tf_based_(false),
    octomap_(PublisherUtils::getROSParam<float>(nh, name_of_node_ + "/octomap/resolution", 0.1)),
    pointcloud_chunks_stashed_(0),
    clear_octomap_(false),
    localize_only(false),
    map_scale_(1.50),
    perform_scale_correction_(true),
    scaling_distance_(1.00),
    camera_height_(0.0),
    camera_height_mult_(1.0),
    camera_height_corrected_(camera_height_*camera_height_mult_),
    publish_octomap_(false), publish_projected_map_(true), publish_gradient_map_(false)
{

  initializeParameters(nh);
  orb_state_.state = orb_slam2_ros::ORBState::UNKNOWN;
  loop_close_state_ = false;
  num_loop_closures_ = 0;

  // initialize publishers
  map_pub_         = nh_.advertise<sensor_msgs::PointCloud2>("map", 3);
  map_updates_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_updates", 3);
  image_pub_       = nh_.advertise<sensor_msgs::Image>("frame", 5);
  state_pub_       = nh_.advertise<orb_slam2_ros::ORBState>("info/state", 10);
  state_desc_pub_  = nh_.advertise<std_msgs::String>("info/state_description", 10);
  kp_pub_          = nh_.advertise<std_msgs::UInt32>("info/frame_keypoints", 1);
  kf_pub_          = nh_.advertise<std_msgs::UInt32>("info/map_keyframes", 1);
  mp_pub_          = nh_.advertise<std_msgs::UInt32>("info/matched_points", 1);
  loop_close_pub_  = nh_.advertise<std_msgs::Bool>("info/loop_closed", 2);
  num_loop_closures_pub_  = nh_.advertise<std_msgs::UInt32>("info/num_loop_closures", 2);
  cam_pose_pub_    = nh_.advertise<geometry_msgs::PoseStamped>("cam_pose", 2);
  trajectory_pub_  = nh_.advertise<nav_msgs::Path>("cam_path", 2);

  // initialize subscribers
  mode_sub_       = nh_.subscribe("switch_mode",    1, &ROSPublisher::localizationModeCallback,   this);
  clear_path_sub_ = nh_.subscribe("clear_cam_path", 1, &ROSPublisher::clearCamTrajectoryCallback, this);

  // Initialization transformation listener
  tfBuffer_.reset(new tf2_ros::Buffer);
  tfListener_.reset(new tf2_ros::TransformListener(*tfBuffer_));

  if (octomap_enabled_)
  {
    if ( publish_octomap_ ) {
      octomap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap", 3);
    }
    if ( publish_projected_map_ ) {
      projected_map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("projected_map", 5, 10);
      projected_morpho_map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("projected_morpho_map", 5, 10);
    }
    if ( publish_gradient_map_ ) {
      gradient_map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("gradient_map", 5, 10);
    }
  }

  // set up filter for height range, also removes NANs:
  // pass_x_.setFilterFieldName("x");
  // pass_x_.setFilterLimits(pointcloud_min_x_, pointcloud_max_x_);
  // pass_y_.setFilterFieldName("y");
  // pass_y_.setFilterLimits(pointcloud_min_y_, pointcloud_max_y_);

  if ( perform_scale_correction_ ) { // TODO: make available only for monocular cameras

    try {

      // only "z" translation component is used (to put PCL/octomap higher/lower)
      tf::StampedTransform tf_target;
      tf::Vector3 cam_base_translation_tf_;

      // tf_listener_.waitForTransform(camera_frame_, base_frame_, ros::Time::now(), ros::Duration(1.0));
      tf_listener_.waitForTransform(camera_frame_, base_frame_, current_frame_time_, ros::Duration(1.0));
      tf_listener_.lookupTransform (camera_frame_, base_frame_, ros::Time(0), tf_target);

      cam_base_translation_tf_.setX(tf_target.getOrigin().x());
      cam_base_translation_tf_.setY(tf_target.getOrigin().y());
      cam_base_translation_tf_.setZ(tf_target.getOrigin().z());
      cam_base_translation_tf_.setW(tfScalar(1.0));

      // rotation base -> camera applied to corresponding translation from real world
      tf::Transform tf = PublisherUtils::createTF(cam_base_translation_tf_,
                                                  PublisherUtils::quaternionFromRPY(0.0, 0.0, 0.0));

      tf::Transform tf_correction = PublisherUtils::createTF(tf::Vector3(0.0, 0.0, 0.0),
                                                              PublisherUtils::quaternionFromRPY(90.0, 0.0, -90.0));
      tf = tf_correction * tf;

      camera_height_ = tf.getOrigin().z();
      ROS_INFO("camera height: %.1f", camera_height_);

    } catch (tf::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      ros::Duration(3.0).sleep();
    }

  } else {

    ROS_INFO("camera height: %.1f", camera_height_);

  }

  // used because of scale correction coefficient non-ideal estimation
  camera_height_corrected_ = camera_height_ * camera_height_mult_;

}

void ROSPublisher::initializeParameters(ros::NodeHandle &nh) {

  nh.param<std::string>(name_of_node_ + "/topic/map", map_topic_, "map");
  nh.param<std::string>(name_of_node_ + "/topic/map_updates", map_updates_topic_, "map_updates");
  nh.param<std::string>(name_of_node_ + "/topic/orb_image", image_topic_, "orb_image");
  nh.param<std::string>(name_of_node_ + "/topic/state", state_topic_, "info/state");
  nh.param<std::string>(name_of_node_ + "/topic/state_description", state_disc_topic_, "info/state_description");
  nh.param<std::string>(name_of_node_ + "/topic/frame_keypoints", kp_topic_, "info/frame_keypoints");
  nh.param<std::string>(name_of_node_ + "/topic/map_keyframes", kf_topic_, "info/map_keyframes");

  nh.param<std::string>(name_of_node_ + "/topic/matched_points", mp_topic_, "info/matched_points");
  nh.param<std::string>(name_of_node_ + "/topic/loop_closed", loop_close_topic_, "info/loop_closed");
  nh.param<std::string>(name_of_node_ + "/topic/num_loop_closures", num_loop_closures_topic_, "info/num_loop_closures");
  nh.param<std::string>(name_of_node_ + "/topic/cam_pose", cam_pose_topic_, "cam_pose");
  nh.param<std::string>(name_of_node_ + "/topic/cam_path", trajectory_topic_, "cam_path");
  
  nh.param<std::string>(name_of_node_ + "/topic/octomap", octomap_topic_, "octomap");
  nh.param<std::string>(name_of_node_ + "/topic/projected_map", projected_map_topic_, "projected_map");
  nh.param<std::string>(name_of_node_ + "/topic/projected_morpho_map", projected_morpho_map_topic_, "projected_morpho_map");
  nh.param<std::string>(name_of_node_ + "/topic/gradient_map", gradient_map_topic_, "igradient_map");


  // freq and image_topic are defined in ros_mono.cc
  nh.param<float>(name_of_node_ + "/topic/orb_state_republish_rate", orb_state_republish_rate_, 1);
  nh.param<bool>(name_of_node_ + "/topic/requires_subscriber", requires_subscriber_, true);
  nh.param<bool>(name_of_node_ + "/octomap/requires_subscriber", octomap_requires_subscriber_, true);
  nh.param<bool>(name_of_node_ + "/topic/image_requires_subscriber", image_requires_subscriber_, true);


  // odom topic defined in ScaleCorrector.cpp
  nh.param<bool>(name_of_node_ + "/map_scale/perform_correction",        perform_scale_correction_,  true);
  nh.param<float>(name_of_node_ + "/map_scale/scaling_distance",         scaling_distance_,          1.000);
  nh.param<float>(name_of_node_ + "/map_scale/set_manually",             map_scale_,                 1.500);
  nh.param<float>(name_of_node_ + "/map_scale/camera_height",            camera_height_,             0.205);
  nh.param<float>(name_of_node_ + "/map_scale/camera_height_multiplier", camera_height_mult_,        1.000);

  nh.param<bool>       (name_of_node_ + "/frame/adjust_map_frame",   adjust_map_frame_,   false);
  nh.param<std::string>(name_of_node_ + "/frame/map_frame",          map_frame_,          ROSPublisher::DEFAULT_MAP_FRAME);
  nh.param<std::string>(name_of_node_ + "/frame/map_frame_adjusted", map_frame_adjusted_, "/orb_slam2/odom");
  nh.param<std::string>(name_of_node_ + "/frame/camera_frame",       camera_frame_,       ROSPublisher::DEFAULT_CAMERA_FRAME);
  nh.param<std::string>(name_of_node_ + "/frame/base_frame",         base_frame_,         "/orb_slam2/base_link");

  nh.param<bool>(name_of_node_ + "/octomap/enabled",                octomap_enabled_,        true);
  nh.param<bool>(name_of_node_ + "/octomap/publish_octomap",        publish_octomap_,        false);
  nh.param<bool>(name_of_node_ + "/octomap/publish_projected_map",  publish_projected_map_,  true);
  nh.param<bool>(name_of_node_ + "/octomap/publish_gradient_map",   publish_gradient_map_,   false);

  nh.param<bool>(name_of_node_ + "/octomap/rebuild",  octomap_rebuild_, false);
  nh.param<float>(name_of_node_ + "/octomap/rate",    octomap_rate_,    1.0);
  // resolution is set default in constructor

  nh.param<double>(name_of_node_ + "/occupancy/projected_map/min_height", projection_min_height_,  -10.0);
  nh.param<double>(name_of_node_ + "/occupancy/projected_map/max_height", projection_max_height_,  +10.0);

  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/erode_se_size",  erode_se_size_,  3);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/erode_nb",       erode_nb_,       1);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/open_se_size",   open_se_size_,   3);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/open_nb",        open_nb_,        1);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/close_se_size",  close_se_size_,  3);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/close_nb",       close_nb_,       1);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/erode2_se_size", erode2_se_size_, 3);
  nh.param<int>   (name_of_node_ + "/occupancy/projected_map/morpho_oprations/erode2_nb",      erode2_nb_,      1);

  nh.param<float> (name_of_node_ + "/occupancy/height_gradient_map/max_height",   gradient_max_height_,   0);
  nh.param<int>   (name_of_node_ + "/occupancy/height_gradient_map/nb_erosions",  gradient_nb_erosions_,  1);
  nh.param<float> (name_of_node_ + "/occupancy/height_gradient_map/low_slope",    gradient_low_slope_,    M_PI / 4.0);
  nh.param<float> (name_of_node_ + "/occupancy/height_gradient_map/high_slope",   gradient_high_slope_,   M_PI / 3.0);

  std::cout << endl;
  std::cout << "ROS Publisher parameters" << endl;
  std::cout << "TOPIC" << endl;
  std::cout << "- orb_state_republish_rate:  " << orb_state_republish_rate_ << std::endl;
  std::cout << "MAP SCALE" << endl;
  std::cout << "- perform_correction:  " << perform_scale_correction_ << std::endl;
  std::cout << "- set_manually:  " << map_scale_ << std::endl;
  std::cout << "- camera_height:  " << camera_height_ << std::endl;
  std::cout << "- camera_height_multiplier:  " << camera_height_mult_ << std::endl;
  std::cout << "FRAME" << endl;
  std::cout << "- map_frame:  " << map_frame_ << std::endl;
  std::cout << "- map_frame_adjusted:  " << map_frame_adjusted_ << std::endl;
  std::cout << "- camera_frame:  " << camera_frame_ << std::endl;
  std::cout << "- base_frame:  " << base_frame_ << std::endl;
  std::cout << "OCTOMAP" << endl;
  std::cout << "- octomap/enabled:  " << octomap_enabled_ << std::endl;
  std::cout << "- octomap/publish_octomap:  " << publish_octomap_ << std::endl;
  std::cout << "- octomap/publish_projected_map:  " << publish_projected_map_ << std::endl;
  std::cout << "- octomap/publish_gradient_map:  " << publish_gradient_map_ << std::endl;
  std::cout << "- octomap/rebuild:  " << octomap_rebuild_ << std::endl;
  std::cout << "- octomap/rate:  " << octomap_rate_ << std::endl;
  std::cout << "OCCUPANCY/PROJECTED_MAP" << endl;
  std::cout << "- projected_map/min_height:  " << projection_min_height_ << std::endl;
  std::cout << "- projected_map/max_height:  " << projection_max_height_ << std::endl;
  std::cout << "OCCUPANCY/PROJECTED_MAP/MORPHO" << endl;
  std::cout << "- open_se_size:  " << open_se_size_ << std::endl;
  std::cout << "- open_nb:  " << open_nb_ << std::endl;
  std::cout << "- close_se_size:  " << close_se_size_ << std::endl;
  std::cout << "- close_nb:  " << close_nb_ << std::endl;
  std::cout << "- erode_se_size:  " << erode_se_size_ << std::endl;
  std::cout << "- erode_nb:  " << erode_nb_ << std::endl;
  std::cout << "OCCUPANCY/GRADIENT_MAP" << endl;
  std::cout << "- max_height:  " << gradient_max_height_ << std::endl;
  std::cout << "- nb_erosions:  " << gradient_nb_erosions_ << std::endl;
  std::cout << "- low_slope:  " << gradient_low_slope_ << std::endl;
  std::cout << "- high_slope:  " << gradient_high_slope_ << std::endl;
  std::cout << endl;

  // DEPRECATED
  // nh.param<bool>(name_of_node_ + "/octomap/tf_based", octomap_tf_based_, false);
  // nh.param<bool>(name_of_node_ + "/frame/align_map_to_cam_frame",   align_map_to_cam_frame_, true);
  // nh.param<float>(name_of_node_ + "/topic/loop_close_republish_rate_", loop_close_republish_rate_, ROSPublisher::LOOP_CLOSE_REPUBLISH_RATE);

}


tf2::Transform ROSPublisher::TransformFromMat (cv::Mat position_mat) {
  cv::Mat rotation(3,3,CV_32F);
  cv::Mat translation(3,1,CV_32F);

  rotation = position_mat.rowRange(0,3).colRange(0,3);
  translation = position_mat.rowRange(0,3).col(3);


  tf2::Matrix3x3 tf_camera_rotation (rotation.at<float> (0,0), rotation.at<float> (0,1), rotation.at<float> (0,2),
                                     rotation.at<float> (1,0), rotation.at<float> (1,1), rotation.at<float> (1,2),
                                     rotation.at<float> (2,0), rotation.at<float> (2,1), rotation.at<float> (2,2)
                                    );

  tf2::Vector3 tf_camera_translation (translation.at<float> (0), 
                                      translation.at<float> (1), 
                                      translation.at<float> (2)
                                     );

  //Coordinate transformation matrix from orb coordinate system to ros coordinate system
  const tf2::Matrix3x3 tf_orb_to_ros (0, 0, 1,
                                     -1, 0, 0,
                                      0,-1, 0);

  tf2::Matrix3x3 tf_rot_orbCam_to_rosCam, tf_rot_rosCam_to_orbCam, tf_rot_orbCam_to_rosMap;
  tf2::Vector3   tf_trans_orbCam_to_rosCam, tf_trans_rosCam_to_orbCam, tf_trans_orbCam_to_rosMap;
  //Transform from orb coordinate system to ros coordinate system on camera coordinates
  tf_rot_orbCam_to_rosCam = tf_orb_to_ros*tf_camera_rotation;
  tf_trans_orbCam_to_rosCam = tf_orb_to_ros*tf_camera_translation;

  //Inverse matrix
  tf_rot_rosCam_to_orbCam = tf_rot_orbCam_to_rosCam.transpose();
  tf_trans_rosCam_to_orbCam = -(tf_rot_rosCam_to_orbCam*tf_trans_orbCam_to_rosCam);

  //Transform from orb coordinate system to ros coordinate system on map coordinates
  tf_rot_orbCam_to_rosMap = tf_orb_to_ros*tf_rot_rosCam_to_orbCam;
  tf_trans_orbCam_to_rosMap = tf_orb_to_ros*tf_trans_rosCam_to_orbCam;

  return tf2::Transform (tf_rot_orbCam_to_rosMap, tf_trans_orbCam_to_rosMap);
}


tf2::Transform ROSPublisher::TransformToTarget (tf2::Transform tf_in, std::string frame_in, std::string frame_target) {
  // Transform tf_in from frame_in to frame_target
  tf2::Transform tf_map2orig = tf_in;
  tf2::Transform tf_orig2target;
  tf2::Transform tf_map2target;

  tf2::Stamped<tf2::Transform> transformStamped_temp;
  try {
    // Get the transform from camera to target
    geometry_msgs::TransformStamped tf_msg = tfBuffer_->lookupTransform(frame_in, frame_target, ros::Time(0));
    // Convert to tf2
    tf2::fromMsg(tf_msg, transformStamped_temp);
    tf_orig2target.setBasis(transformStamped_temp.getBasis());
    tf_orig2target.setOrigin(transformStamped_temp.getOrigin());

  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s",ex.what());
    //ros::Duration(1.0).sleep();
    tf_orig2target.setIdentity();
  }

  /* 
    // Print debug info
    double roll, pitch, yaw;
    // Print debug map2orig
    tf2::Matrix3x3(tf_map2orig.getRotation()).getRPY(roll, pitch, yaw);
    ROS_INFO("Static transform Map to Orig [%s -> %s]",
                    map_frame_id_param_.c_str(), frame_in.c_str());
    ROS_INFO(" * Translation: {%.3f,%.3f,%.3f}",
                    tf_map2orig.getOrigin().x(), tf_map2orig.getOrigin().y(), tf_map2orig.getOrigin().z());
    ROS_INFO(" * Rotation: {%.3f,%.3f,%.3f}",
                    RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
    // Print debug tf_orig2target
    tf2::Matrix3x3(tf_orig2target.getRotation()).getRPY(roll, pitch, yaw);
    ROS_INFO("Static transform Orig to Target [%s -> %s]",
                    frame_in.c_str(), frame_target.c_str());
    ROS_INFO(" * Translation: {%.3f,%.3f,%.3f}",
                    tf_orig2target.getOrigin().x(), tf_orig2target.getOrigin().y(), tf_orig2target.getOrigin().z());
    ROS_INFO(" * Rotation: {%.3f,%.3f,%.3f}",
                    RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
    // Print debug map2target
    tf2::Matrix3x3(tf_map2target.getRotation()).getRPY(roll, pitch, yaw);
    ROS_INFO("Static transform Map to Target [%s -> %s]",
                    map_frame_id_param_.c_str(), frame_target.c_str());
    ROS_INFO(" * Translation: {%.3f,%.3f,%.3f}",
                    tf_map2target.getOrigin().x(), tf_map2target.getOrigin().y(), tf_map2target.getOrigin().z());
    ROS_INFO(" * Rotation: {%.3f,%.3f,%.3f}",
                    RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
  */

  // Transform from map to target
  tf_map2target = tf_map2orig * tf_orig2target;
  return tf_map2target;
}



/*
 * Either appends all GetReferenceMapPoints to the pointcloud stash or clears the stash and re-fills it
 * with GetAllMapPoints, in case there is a big map change in ORB_SLAM 2 or all_map_points is set to true.
 */
void ROSPublisher::stashMapPoints(bool all_map_points)
{
    std::vector<MapPoint*> map_points;

    pointcloud_map_points_mutex_.lock();

    if (all_map_points || GetMap()->GetLastBigChangeIdx() > lastBigMapChange_) {
      pointcloud_map_points_.clear();
      octomap::pointCloud2ToOctomap(all_map_points_, pointcloud_map_points_);
      
      // map_points = GetMap()->GetAllMapPoints();
      lastBigMapChange_ = GetMap()->GetLastBigChangeIdx();
      clear_octomap_ = true;
      pointcloud_chunks_stashed_ = 1;

    } 
    else {
      // pass_x_.setInputCloud(reference_map_points_.makeShared());
      // pass_x_.filter(reference_map_points_);
      // pass_y_.setInputCloud(reference_map_points_.makeShared());
      // pass_y_.filter(reference_map_points_);
      octomap::pointCloud2ToOctomap(reference_map_points_, pointcloud_map_points_);

      // map_points = GetMap()->GetReferenceMapPoints();
      pointcloud_chunks_stashed_++;
    }

    // for (MapPoint *map_point : map_points) {
    //     if (map_point->isBad()) {
    //         continue;
    //     }
    //     cv::Mat pos = map_point->GetWorldPos();
    //     PublisherUtils::transformPoint(pos, map_scale_, true, 1, camera_height_corrected_);
    //     pointcloud_map_points_.push_back(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    // }

    pointcloud_map_points_mutex_.unlock();
}

/*
 * Octomap worker thread function, which has exclusive access to the octomap. Updates and publishes it.
 */

void ROSPublisher::octomapWorker()
{

    static std::chrono::system_clock::time_point this_cycle_time;

    octomap::pose6d frame;
    octomap::point3d origin = { 0.0, 0.0, 0.0 };
    bool got_tf = false;

    // wait until ORB_SLAM 2 is up and running
    ROS_INFO("octomapWorker thread: waiting for ORBState OK");

    while (orb_state_.state != orb_slam2_ros::ORBState::OK)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    ROS_INFO("octomapWorker thread: starting to work (ORBState is OK)");

    // main thread loop
    while (!isStopped())
    {
        this_cycle_time = std::chrono::system_clock::now();

        if ( !got_tf ) {

          try {

            tf::StampedTransform transform_in_target_frame;
            tf_listener_.waitForTransform(base_frame_, camera_frame_, ros::Time(0), ros::Duration(1.0));
            tf_listener_.lookupTransform( base_frame_, camera_frame_, ros::Time(0), transform_in_target_frame);
            
            // geometry_msgs::TransformStamped tf_msg = tfBuffer_->lookupTransform(frame_in, frame_target, ros::Time(0));

            static const tf::Transform tf_octomap = PublisherUtils::createTF(tf::Vector3(tfScalar(0.0),
                                                                                      tfScalar(0.0),
                                                                                      tfScalar(camera_height_corrected_)),
                                                                          transform_in_target_frame.getRotation() );
            frame = octomap::poseTfToOctomap(tf_octomap);
            got_tf = true;

          } catch (tf::TransformException &ex) {

            frame = octomap::pose6d(0, 0, 0, 0, 0, 0);
            got_tf = false;

          }

        }

        if (got_tf || octomap_rebuild_ )
        {
          // clear whenever TF mode changes
          clear_octomap_ |= (got_tf != octomap_tf_based_);

          if (clear_octomap_)
          {
            // WARNING: causes ugly segfaults in octomap 1.8.0
            octomap_.clear();
            ROS_INFO("octomapWorker: octomap cleared, rebuilding...");

            /*
             * TODO: if pointcloud is supposed to be a lidar scan result, this is problematic
             * (multiple hits on one beam/previous hits getting overwritten etc.)
             *
             */
            stashMapPoints(true);     // stash whole map
            clear_octomap_ = false;   // TODO: mutex?
          }

          pointcloud_map_points_mutex_.lock();
          octomap_.insertPointCloud(pointcloud_map_points_, origin, frame);

          pointcloud_map_points_.clear();
          int pointcloud_chunks_stashed = pointcloud_chunks_stashed_;
          pointcloud_chunks_stashed_ = 0;
          pointcloud_map_points_mutex_.unlock();

          octomap_tf_based_ = got_tf;

          if ( publish_octomap_ ) {
            //ROS_INFO("Publishing Octomap...");
            publishOctomap();
            //ROS_INFO("Octomap published");
          }
          if ( publish_projected_map_ ) {
            //ROS_INFO("Publishing Projected map...");
            publishProjectedMap();
            //ROS_INFO("Projected map published");
          }
          if ( publish_gradient_map_ ) {
            //ROS_INFO("Publishing Gradient map...");
            publishGradientMap();
            //ROS_INFO("Gradient map published");
          }

          if ( pointcloud_chunks_stashed > 0 ) {
            ROS_INFO("octomapWorker: finished cycle integrating %i pointcloud chunks.", pointcloud_chunks_stashed);
          }
        }
        else
        {

          ROS_INFO("octomapWorker thread: missing camera TF, losing %i pointcloud chunks.", pointcloud_chunks_stashed_);
          pointcloud_map_points_mutex_.lock();
          pointcloud_map_points_.clear();
          pointcloud_chunks_stashed_ = 0;
          pointcloud_map_points_mutex_.unlock();

        }

        std::this_thread::sleep_until(this_cycle_time + std::chrono::milliseconds((int) (1000. / octomap_rate_)));
    }

    ROS_INFO("octomapWorker thread: stopped");
}

/*
 * Creates a 2D Occupancy Grid from the Octomap.
 */
void ROSPublisher::octomapCutToOccupancyGrid(const octomap::OcTree& octree, 
                                             nav_msgs::OccupancyGrid& map, 
                                             nav_msgs::OccupancyGrid& map_erode, 
                                             const double minZ_, const double maxZ_ )
{

    static const uint8_t a = 1;
    static const uint8_t b = 0;
    static const uint8_t c = 2;

    map.info.resolution = octree.getResolution();
    double minX, minY, minZ;
    double maxX, maxY, maxZ;
    octree.getMetricMin(minX, minY, minZ);
    octree.getMetricMax(maxX, maxY, maxZ);
    ROS_DEBUG("Octree min %f %f %f", minX, minY, minZ);
    ROS_DEBUG("Octree max %f %f %f", maxX, maxY, maxZ);
    minZ = std::max(minZ_, minZ);
    maxZ = std::min(maxZ_, maxZ);

    octomap::point3d minPt(minX, minY, minZ);
    octomap::point3d maxPt(maxX, maxY, maxZ);
    octomap::OcTreeKey minKey, maxKey, curKey;

    if (!octree.coordToKeyChecked(minPt, minKey))
    {
        ROS_ERROR("Could not create OcTree key at %f %f %f", minPt.x(), minPt.y(), minPt.z());
        return;
    }
    if (!octree.coordToKeyChecked(maxPt, maxKey))
    {
        ROS_ERROR("Could not create OcTree key at %f %f %f", maxPt.x(), maxPt.y(), maxPt.z());
        return;
    }

    map.info.width = maxKey[b] - minKey[b] + 1;
    map.info.height = maxKey[a] - minKey[a] + 1;

    // might not exactly be min / max:
    octomap::point3d origin =   octree.keyToCoord(minKey, octree.getTreeDepth());

    /*
     * Aligns base_link with origin of map frame, but is not correct in terms of real environment
     * (real map's origin is in the origin of camera's origin)
     * map.info.origin.position.x = origin.x() - octree.getResolution() * 0.5 - cam_base_translation_.at<float>(0);
     * map.info.origin.position.y = origin.y() - octree.getResolution() * 0.5 - cam_base_translation_.at<float>(1);
     *
     */

    map.info.origin.position.x = origin.x() - octree.getResolution() * 0.5;
    map.info.origin.position.y = origin.y() - octree.getResolution() * 0.5;

    map.info.origin.orientation.x = 0.;
    map.info.origin.orientation.y = 0.;
    map.info.origin.orientation.z = 0.;
    map.info.origin.orientation.w = 1.;

    // Allocate space to hold the data
    map.data.resize(map.info.width * map.info.height, -1);

    // Matrix of map's size is inited with unknown (-1) value at each point
    for(std::vector<int8_t>::iterator it = map.data.begin(); it != map.data.end(); ++it) {
       *it = -1;
    }

    map_erode = map;  // plain copy of one struct - copy needed for second map version

    /*
     * Matrix for morphological operations
    ** Matrix's type set to Unsigned Char - additional loop for assigning matrix's
    ** values to map.data vector is still a must
    ** Another thing is that morphological operations behave strange with negative values inside a matrix
    ** It is probably because erode, close and open aren't supposed to deal with negative (-1) values inside matrix
    */

    // values in the matrix
    static const unsigned char MAT_UNKNOWN  = 10;
    static const unsigned char MAT_NON_OCC  = 50;
    static const unsigned char MAT_OCCUPIED = 100;

    cv::Mat map_data_matrix;
    map_data_matrix.create(map.info.height, map.info.width, CV_8U); // ensures that the matrix is continuous
    map_data_matrix.setTo(MAT_UNKNOWN);

    // map creation time
    ros::Time t_start = ros::Time::now();
    unsigned i, j;
    // iterate over all keys:
    for (curKey[a] = minKey[a], j = 0; curKey[a] <= maxKey[a]; ++curKey[a], ++j)
    {
        // pointer to the current row start
        uchar* mat_ptr = map_data_matrix.ptr<uchar>(j);

        for (curKey[b] = minKey[b], i = 0; curKey[b] <= maxKey[b]; ++curKey[b], ++i)
        {
            for (curKey[c] = minKey[c]; curKey[c] <= maxKey[c]; ++curKey[c])
            {   //iterate over height

                octomap::OcTreeNode* node = octree.search(curKey);
                if (node)
                {
                  // creates map data
                  bool occupied = octree.isNodeOccupied(node);
                  if(occupied) {

                      map.data[map.info.width * j + i] = 100;
                      mat_ptr[i] = MAT_OCCUPIED;
                      break;

                  } else {
                      map.data[map.info.width * j + i] = 0;
                      mat_ptr[i] = MAT_NON_OCC;
                  }
                }
            }
        }
    }

    /*
    ** Application of a morphological operations to map - they clear
    ** single points that are incorrectly interpreted as occupied
    */

    if ( projected_morpho_map_pub_.getNumSubscribers() > 0 ) {

      /* ERODE */
      if ( erode_nb_ > 0 ) {
        cv::erode(map_data_matrix,
                  map_data_matrix,
                  cv::getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                            cv::Size(erode_se_size_,erode_se_size_),
                                            cv::Point(-1,-1)),
                  cv::Point(-1,-1),
                  erode_nb_,
                  cv::BORDER_CONSTANT,
                  cv::morphologyDefaultBorderValue());
      }

      /* OPEN */
      if ( open_nb_ > 0 ) {
        cv::morphologyEx(map_data_matrix,
                         map_data_matrix,
                         cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, // MORPH_CROSS,
                                                   cv::Size(open_se_size_,open_se_size_),
                                                   cv::Point(-1,-1)),
                         cv::Point(-1,-1),
                         open_nb_,
                         cv::BORDER_CONSTANT,
                         cv::morphologyDefaultBorderValue());
      }

      /* CLOSE */
      if ( close_nb_ > 0 ) {
        cv::morphologyEx(map_data_matrix,
                         map_data_matrix,
                         cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, // MORPH_CROSS,
                                                   cv::Size(close_se_size_,close_se_size_),
                                                   cv::Point(-1,-1)),
                         cv::Point(-1,-1),
                         close_nb_,
                         cv::BORDER_CONSTANT,
                         cv::morphologyDefaultBorderValue());
      }

      /* ERODE */
      if ( erode2_nb_ > 0 ) {
        cv::erode(map_data_matrix,
                  map_data_matrix,
                  cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                            cv::Size(erode2_se_size_,erode2_se_size_),
                                            cv::Point(-1,-1)),
                  cv::Point(-1,-1),
                  erode2_nb_,
                  cv::BORDER_CONSTANT,
                  cv::morphologyDefaultBorderValue());
      }

      // nav_msgs/OccupancyGrid msg out of map after morphological operations
      for ( int j = 0; j < map_erode.info.height; j++) {

        // pointer to the current row start
        uchar* mat_ptr = map_data_matrix.ptr<uchar>(j);

        for ( int i = 0; i < map_erode.info.width; i++ ) {
          switch( mat_ptr[i] )
          {
            case MAT_UNKNOWN:
                map_erode.data[map_erode.info.width * j + i] = -1;
                break;
            case MAT_NON_OCC:
                map_erode.data[map_erode.info.width * j + i] = 0;
                break;
            case MAT_OCCUPIED:
                map_erode.data[map_erode.info.width * j + i] = 100;
                break;
          }
        }
      }
    }

    ros::Duration t_stop = (ros::Time::now() - t_start);
    ROS_INFO("Occupancy grid: %d x %d created in %.3f sec",  map_erode.info.width,
                                                             map_erode.info.height,
                                                             t_stop.toSec() );
}

/*
 * Constructs a 2-dimensional OccupancyGrid from an Octomap by evaluating its heightmap gradients.
 */
void ROSPublisher::octomapGradientToOccupancyGrid(const octomap::OcTree& octree, nav_msgs::OccupancyGrid& map, float max_height, int nb_erosions, float low_slope, float high_slope)
{
    // get tree dimensions
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    octree.getMetricMin(min_x, min_y, min_z);
    octree.getMetricMax(max_x, max_y, max_z);
    octomap::point3d min_point(min_x, min_y, min_z);
    octomap::point3d max_point(max_x, max_y, max_z);

    // fill in map dimensions
    map.info.resolution = octree.getResolution();
    map.info.width = (max_point.x() - min_point.x()) / map.info.resolution + 1;
    map.info.height = (max_point.y() - min_point.y()) / map.info.resolution + 1;

    map.info.origin.position.x = min_point.x() - map.info.resolution * 0.5;
    map.info.origin.position.y = min_point.y() - map.info.resolution * 0.5;

    map.info.origin.orientation.x = 0.;
    map.info.origin.orientation.y = 0.;
    map.info.origin.orientation.z = 0.;
    map.info.origin.orientation.w = 1.;

    // create CV matrix of proper size with 1 channel of 32 bit floats and init values to NaN for "unknown"
    cv::Mat height_map(map.info.height, map.info.width, CV_32FC1, NAN);

    // iterate over tree leafs to create height map
    octomap::point3d coord;
    int x, y;
    float z;
    for(octomap::OcTree::leaf_iterator it = octree.begin_leafs(), end=octree.end_leafs(); it != end; ++it)
    {
        if (octree.isNodeOccupied(*it))
        {
            coord = it.getCoordinate();
            x = (coord.x() - min_point.x()) / map.info.resolution;
            y = (coord.y() - min_point.y()) / map.info.resolution;
            z = coord.z(); // z-axis is facing UP
            if (z <= max_height) // only consider voxels up to specified height (e.g. for building indoor maps)
            {
                float current_height = height_map.at<float>(y, x);
                if (current_height != current_height || z > current_height)
                {
                    height_map.at<float>(y, x) = z;
                }
            }
        }
    }

    // fill in small holes
    PublisherUtils::erodeNaN(height_map, nb_erosions);
    // store where height is unknown
    cv::Mat mask_unknown = height_map != height_map; // is NaN

    PublisherUtils::erodeNaN(height_map, 1); // avoid discontinuity (and thus a "wall") around known area

    height_map.setTo(0, height_map != height_map); // get rid of all NaN trouble makers

    // get height gradient
    cv::Mat gradient_x, gradient_y, gradient_map;
    cv::Scharr(height_map, gradient_x, CV_32F, 1, 0, 1. / 16.);
    cv::Scharr(height_map, gradient_y, CV_32F, 0, 1, 1. / 16.);
    cv::addWeighted(cv::abs(gradient_x), 0.5, cv::abs(gradient_y), 0.5, 0, gradient_map); // TODO 0.5 rly?

    // height slope thresholds:
    // values < lower are considered free space
    // values > upper are considered obstacle
    // everything inbetween is literally a gray-zone
    float threshold_lower = sin(low_slope) / cos(low_slope) * map.info.resolution;
    float threshold_upper = sin(high_slope) / cos(high_slope) * map.info.resolution;

    // map data probabilities are in range [0,100].  Unknown is -1.
    gradient_map.setTo(threshold_upper, gradient_map > threshold_upper); // clip obstacles
    gradient_map.setTo(threshold_lower, gradient_map < threshold_lower); // clip free space
    gradient_map = (gradient_map - threshold_lower) / (threshold_upper - threshold_lower) * 100.0; // convert into map data range
    gradient_map.setTo(-1, mask_unknown); //replace NaNs

    // ensure correct size of map data vector
    map.data.resize(map.info.width * map.info.height);
    // fill in map data
    for(y = 0; y < gradient_map.rows; ++y) {
        for(x = 0; x < gradient_map.cols; ++x) {
            map.data[y * map.info.width + x] = gradient_map.at<float>(y, x);
        }
    }
}

/*
 * Publishes ORB_SLAM 2 GetAllMapPoints() as a PointCloud2.
 */
void ROSPublisher::publishMap()
{
    all_map_points_ = PublisherUtils::convertToPCL2(GetMap()->GetAllMapPoints(),
                                                                map_scale_,
                                                                camera_height_corrected_);

    if (map_pub_.getNumSubscribers() > 0 || !requires_subscriber_)
    {
        
        all_map_points_.header.frame_id = map_frame_;
        // msg.header.stamp = ros::Time::now();;
        all_map_points_.header.stamp = current_frame_time_;
        map_pub_.publish(all_map_points_);

    }
}

/*
 * Publishes ORB_SLAM 2 GetReferenceMapPoints() as a PointCloud2.
 */
void ROSPublisher::publishMapUpdates()
{
    reference_map_points_ = PublisherUtils::convertToPCL2(GetMap()->GetReferenceMapPoints(),
                                                          map_scale_,
                                                          camera_height_corrected_);

    if (map_updates_pub_.getNumSubscribers() > 0 || !requires_subscriber_)
    {
        // sensor_msgs::PointCloud2 msg = PublisherUtils::convertToPCL2(GetMap()->GetAllMapPoints(),
        //                                                              map_scale_,
        //                                                              camera_height_corrected_);
        
        reference_map_points_.header.frame_id = map_frame_;
        map_updates_pub_.publish(reference_map_points_);
    }
}


// void Node::PublishCameraPositionAsTransform (cv::Mat position) {
//   // Get transform from map to camera frame
//   tf2::Transform tf_pos_camera_in_map = TransformFromMat(position);

//   // Make transform from camera frame to target frame
//   tf2::Transform target_frame_id_param_ = TransformToTarget(tf_transform, camera_frame_, target_frame_id_param_);

//   // Make message
//   tf2::Stamped<tf2::Transform> tf_map2target_stamped;
//   tf_map2target_stamped = tf2::Stamped<tf2::Transform>(tf_map2target, current_frame_time_, map_frame_id_param_);
//   geometry_msgs::TransformStamped msg = tf2::toMsg(tf_map2target_stamped);
//   msg.child_frame_id = target_frame_id_param_;
//   // Broadcast tf
//   static tf2_ros::TransformBroadcaster tf_broadcaster;
//   tf_broadcaster.sendTransform(msg);
// }

geometry_msgs::PoseStamped ROSPublisher::getPoseBaseInMap(cv::Mat orb_Pmap_camera, ros::Time frame_time)
{
  tf2::Transform Tmap_camera = TransformFromMat(orb_Pmap_camera);

  // Make transform from camera frame to target frame
  tf2::Transform Tmap_base = TransformToTarget(Tmap_camera, camera_frame_, base_frame_);
  
  // Make message
  tf2::Stamped<tf2::Transform> Tmap_base_stamped;
  Tmap_base_stamped = tf2::Stamped<tf2::Transform>(Tmap_base, frame_time, map_frame_);
  
  geometry_msgs::PoseStamped msg_Pmap_base;
  tf2::toMsg(Tmap_base_stamped, msg_Pmap_base);

  return msg_Pmap_base;
}


geometry_msgs::TransformStamped ROSPublisher::getTransformBaseInMap(cv::Mat orb_Pmap_camera, ros::Time frame_time)
{
  tf2::Transform Tmap_camera = TransformFromMat(orb_Pmap_camera);

  // Make transform from camera frame to target frame
  tf2::Transform Tmap_base = TransformToTarget(Tmap_camera, camera_frame_, base_frame_);
  
  // Make message
  tf2::Stamped<tf2::Transform> Tmap_base_stamped;
  Tmap_base_stamped = tf2::Stamped<tf2::Transform>(Tmap_base, frame_time, map_frame_);
  

  geometry_msgs::TransformStamped msg_Tmap_base = tf2::toMsg(Tmap_base_stamped);
  msg_Tmap_base.child_frame_id = base_frame_;

  return msg_Tmap_base;
}


/*
 * Publishes ORB_SLAM 2 GetCameraPose() as a TF.
 */
void ROSPublisher::publishCurrentCameraPose(cv::Mat orbCameraPose)
{
  cam_pose_ = getPoseBaseInMap(orbCameraPose, current_frame_time_);
  cam_pose_pub_.publish(cam_pose_);

  geometry_msgs::TransformStamped msg_Tmap_base = getTransformBaseInMap(orbCameraPose, current_frame_time_);
  camera_tf_pub_.sendTransform(msg_Tmap_base);

  ResetCamFlag();
}

/*
 * Publishes the previously built Octomap. (called from the octomap worker thread)
 */
void ROSPublisher::publishOctomap()
{
    if (octomap_pub_.getNumSubscribers() > 0 || !octomap_requires_subscriber_)
    {
        auto t0 = std::chrono::system_clock::now();
        octomap_msgs::Octomap msgOctomap;
        msgOctomap.header.frame_id = map_frame_;
        /* TODO: add as a parameter
        if ( adjust_map_frame_ ) {

          msgOctomap.header.frame_id =  octomap_tf_based_ ?
                                                map_frame_adjusted_ :
                                                map_frame_;

          msgOctomap.header.frame_id = octomap_frame_;
        } else {
          msgOctomap.header.frame_id =  map_frame_;
        }
        */
        //  msgOctomap.header.stamp = ros::Time::now();
        msgOctomap.header.stamp = current_frame_time_;
        if (octomap_msgs::binaryMapToMsg(octomap_, msgOctomap))   // TODO: full/binary...?
        {
            auto tn = std::chrono::system_clock::now();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(tn - t0);
            //std::cout << "msg generation time: " << dt.count() << " ms" << std::endl;
            t0 = std::chrono::system_clock::now();
            octomap_pub_.publish(msgOctomap);
            tn = std::chrono::system_clock::now();
            dt = std::chrono::duration_cast<std::chrono::milliseconds>(tn - t0);
            //std::cout << "msg publish time: " << dt.count() << " ms" << std::endl;
        }
    }
}

/*
 * Publishes the ORB_SLAM 2 tracking state as ORBState int and/or as a description string.
 */
void ROSPublisher::publishState(Tracking *tracking)
{
  static orb_slam2_ros::ORBState orb_state_last;
  orb_state_last.state = orb_state_.state;

    if (tracking != NULL) {
        // save state from tracking, even if there are no subscribers
        orb_state_ = PublisherUtils::toORBStateMessage(tracking->mState);
    }

    if (orb_state_.state!=orb_state_last.state)
    {
      ROS_INFO("Updated ORBState from %s to %s\n", PublisherUtils::stateDescription(orb_state_last), 
                                                    PublisherUtils::stateDescription(orb_state_));
    }

    if (state_pub_.getNumSubscribers() > 0 || !requires_subscriber_)
    {
        // publish state as ORBState int
        // orb_state_.header.stamp = ros::Time::now();
        orb_state_.header.stamp = current_frame_time_;
        state_pub_.publish(orb_state_);
    }

    if (state_desc_pub_.getNumSubscribers() > 0 || !requires_subscriber_)
    {
        // publish state as string
        std_msgs::String state_desc_msg;
        state_desc_msg.data = PublisherUtils::stateDescription(orb_state_);
        state_desc_pub_.publish(state_desc_msg);
    }

    // last_state_publish_time_ = ros::Time::now();
}

/*
 * Publishes the current ORB_SLAM 2 status image.
 */
void ROSPublisher::publishImage(Tracking *tracking)
{

    if (image_pub_.getNumSubscribers() > 0 || image_requires_subscriber_)
    {

      drawer_.Update(tracking);

      std_msgs::Header hdr;
      cv_bridge::CvImage cv_img {hdr, "bgr8", drawer_.DrawFrame()};

      auto image_msg = cv_img.toImageMsg();
      image_msg->header = hdr;
      // image_msg->header.stamp = ros::Time::now();
      image_msg->header.stamp = current_frame_time_;
      image_pub_.publish(*image_msg);

    }
}

/*
 * Creates a 2D OccupancyGrid from the Octomap by performing a cut through a previously specified z interval and publishes it.
 */
void ROSPublisher::publishProjectedMap()
{

    int8_t proj_sub_nr = projected_map_pub_.getNumSubscribers();
    int8_t proj_ero_sub_nr = projected_morpho_map_pub_.getNumSubscribers();

    if ( proj_sub_nr || proj_ero_sub_nr || !octomap_requires_subscriber_) {

        static nav_msgs::OccupancyGrid msg;
        static nav_msgs::OccupancyGrid msg_eroded;

        msg.header.frame_id = map_frame_;
        msg_eroded.header.frame_id = map_frame_;
        /*
        if ( adjust_map_frame_ ) {

          msg.header.frame_id =  octomap_tf_based_ ?
                                                map_frame_adjusted_ :
                                                map_frame_;

          msg.header.frame_id = octomap_frame_; == map
          msg_eroded.header.frame_id = octomap_frame_;
        } else {
          msg.header.frame_id =  map_frame_;
          msg_eroded.header.frame_id =  map_frame_;
        }
        */
        // msg.header.stamp = ros::Time::now();;
        // msg_eroded.header.stamp = ros::Time::now();;
        msg.header.stamp = current_frame_time_;
        msg_eroded.header.stamp = current_frame_time_;

        octomapCutToOccupancyGrid(octomap_, msg, msg_eroded, projection_min_height_, projection_max_height_);

        // one of maps published
        if ( proj_sub_nr > 0 ) {
          projected_map_pub_.publish(msg);
        } 
        if ( proj_ero_sub_nr > 0) {
          projected_morpho_map_pub_.publish(msg_eroded);
        }
    }
}

/*
 * Creates a 2D OccupancyGrid from the Octomap by evaluating its heightmap gradients and publishes it.
 */
void ROSPublisher::publishGradientMap()
{

    if (gradient_map_pub_.getNumSubscribers() > 0)
    {
        static nav_msgs::OccupancyGrid msg;
        msg.header.frame_id = map_frame_;
        /*
        if ( adjust_map_frame_ ) {

          msg.header.frame_id =  octomap_tf_based_ ?
                                                map_frame_adjusted_ :
                                                map_frame_;

          msg.header.frame_id = octomap_frame_; == map
        } else {
          msg.header.frame_id =  map_frame_;
        }
        */
        // msg.header.stamp = ros::Time::now();
        msg.header.stamp = current_frame_time_;

        octomapGradientToOccupancyGrid(octomap_, msg,
                                       gradient_max_height_, gradient_nb_erosions_,
                                       gradient_low_slope_,  gradient_high_slope_);

        gradient_map_pub_.publish(msg);
    }
}


void ROSPublisher::publishCamTrajectory(bool finished_loop_closure)
{

  static nav_msgs::Path msg;

  if ( clear_path_ ) {
    std::cout << "Clear Camera Trajectory\n";
    msg.poses.clear();
    clear_path_ = false;
  }

  // Global bundle adjustment just finished
  if (finished_loop_closure) {
    static std::vector<KeyFrame*> current_key_frames = GetMap()->GetAllKeyFrames();

    for (unsigned int msg_iter=0; msg_iter<msg.poses.size(); msg_iter++) {
      // Update the pose and transform for each key point 
      cv::Mat orb_kf_pose_adjusted = current_key_frames[msg_iter]->GetPose();

      msg.poses[msg_iter] = getPoseBaseInMap(orb_kf_pose_adjusted, msg.poses[msg_iter].header.stamp);

      geometry_msgs::TransformStamped msg_Tmap_base = getTransformBaseInMap(orb_kf_pose_adjusted, msg.poses[msg_iter].header.stamp);
      camera_tf_pub_.sendTransform(msg_Tmap_base);
    }
    trajectory_pub_.publish(msg);
  }

  if ( trajectory_pub_.getNumSubscribers() > 0 || !requires_subscriber_) {
    static int num_key_frames;

    if (drawer_.GetKeyFramesNb() != num_key_frames) {
      msg.header.frame_id = map_frame_;
      // msg.header.stamp = ros::Time::now();
      msg.header.stamp = current_frame_time_;

      geometry_msgs::PoseStamped trajectory_pose = cam_pose_;
      msg.poses.push_back(trajectory_pose);
      trajectory_pub_.publish(msg);
    }
  }

}

void ROSPublisher::publishLoopState() {
  std_msgs::Bool msg;
  msg.data = loop_close_state_;
  loop_close_pub_.publish(msg);
}

void ROSPublisher::publishNumLoopClosures() {
  std_msgs::UInt32 msg;
  msg.data = num_loop_closures_;
  num_loop_closures_pub_.publish(msg);
}

void ROSPublisher::publishUInt32Msg(const ros::Publisher &pub, const unsigned long &data) {
  std_msgs::UInt32 msg;
  msg.data = data;
  pub.publish(msg);
}

void ROSPublisher::checkMode() {

  static bool mode_last_state = false;        // default start with SLAM mode
  if ( mode_last_state != localize_only )
  {
    if ( localize_only ) {
      GetSystem()->ActivateLocalizationMode();
    } else {
      GetSystem()->DeactivateLocalizationMode();
    }
  }
  mode_last_state = localize_only;            // to prevent from periodic mode activation/deactivation - switch will be applied in case of changing value

}

bool ROSPublisher::checkLoopClosure()
{
  // Increment on the falling edge of bundle adjustment (which takes care of loop closing updates)

  const bool loop_close_state_new = GetLoopCloser()->isRunningGBA();
  // Should publish number of loop closures for others to use in tracking for closure
  bool end_loop_closure = false;

  if(loop_close_state_ & !loop_close_state_new) {
    num_loop_closures_++;
    publishNumLoopClosures();
    end_loop_closure = true;
  }
  loop_close_state_ = loop_close_state_new;

  return end_loop_closure;
}

void ROSPublisher::camInfoUpdater() 
{

  // these operations are moved from Run() to separate thread, it was crucial in my application
  // to get camera pose updates as frequently as possible

  // static ros::Time last_camera_update = ros::Time::now();
  static ros::Time last_camera_update = current_frame_time_;

  while (WaitCycleStart()) {
    if ( isCamUpdated() ) {

      // float tf_delta = (ros::Time::now() - last_camera_update).toSec();
      // last_camera_update = ros::Time::now();
      float tf_delta = (current_frame_time_ - last_camera_update).toSec();
      last_camera_update = current_frame_time_;

      cv::Mat orb_camera_pose = GetCameraPose();

      if(!orb_camera_pose.empty()) {

        bool end_loop_closure = checkLoopClosure();

        publishCurrentCameraPose(orb_camera_pose);

        publishCamTrajectory(end_loop_closure);

        // for better visualization only
        if ( tf_delta < 0.75 ) {
          // ROS_INFO("Updated camera pose published after %.3f",  tf_delta);
        } else if ( tf_delta < 1.50 ) {
          ROS_WARN("Updated camera pose published after %.3f",  tf_delta);
        } else {
          ROS_ERROR("Updated camera pose published after %.3f", tf_delta);
        }

      }

      

      // if ( ros::Time::now() >= (last_state_publish_time_ +
      if ( current_frame_time_ >= (last_state_publish_time_ +
           ros::Duration(1. / orb_state_republish_rate_)) )
      {
        // it's time to re-publish info
        publishState(NULL);
        checkMode();
        publishUInt32Msg(kf_pub_, drawer_.GetKeyFramesNb());
        publishUInt32Msg(kp_pub_, drawer_.GetKeypointsNb());
        publishUInt32Msg(mp_pub_, drawer_.GetMatchedPointsNb());
        publishLoopState(); // GBA is quite time-consuming task so it will probably be detected here
      //  last_state_publish_time_ = ros::Time::now();
        last_state_publish_time_ = current_frame_time_;
      }
    }
  }
  ROS_ERROR("InfoUpdater finished");
  SetFinish(true);


}

void ROSPublisher::Run()
{
    using namespace std::this_thread;
    using namespace std::chrono;

    ROS_INFO("ROS publisher started");

    // if ( perform_scale_correction_ && GetSystem()->GetSensorType() == ORB_SLAM2::System::eSensor::MONOCULAR) {

    //   bool scale_correction = false; // flag to check state at the end
    //   ScaleCorrector scale_corrector(scaling_distance_);
    //   ROS_INFO("Waiting for initialization...");
    //   while ( GetSystem()->GetTrackingState() <= ORB_SLAM2::Tracking::NOT_INITIALIZED ) {

    //     if ( isStopRequested() ) {
    //       Stop();
    //       // GetSystem()->Shutdown();
    //     }
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //   }

    //   ROS_WARN("Starting the map scale correction procedure...");
    //   ROS_WARN("You should move the robot constantly in one direction");

    //   while ( !scale_corrector.isScaleUpdated() ) {
    //     scale_correction = true;
    //     // TODO: freezes when trying to shutdown here
    //     cv::Mat xf = PublisherUtils::computeCameraTransform(GetCameraPose());
    //     if ( !scale_corrector.gotCamPosition() ) {
    //       scale_corrector.setCameraStartingPoint(xf.at<float>(0, 3),
    //                                              xf.at<float>(1, 3),
    //                                              xf.at<float>(2, 3));
    //     }

    //     if ( scale_corrector.isReady() ) {
    //       scale_corrector.calculateScale(xf.at<float>(0, 3),
    //                                      xf.at<float>(1, 3),
    //                                      xf.at<float>(2, 3));
    //     }
    //     if ( isStopRequested() ) {
    //       Stop();
    //       scale_correction = false;
    //       // GetSystem()->Shutdown();
    //       ROS_WARN("Scale correction procedure must be stopped");
    //     }
    //     if ( GetSystem()->GetTrackingState() == ORB_SLAM2::Tracking::LOST ) {
    //       ROS_WARN("Scale correction procedure couldn't be fully performed - tracking lost. Try to re-initialize");
    //       scale_correction = false;
    //       break;
    //     }
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));

    //   }

    //   if ( scale_correction ) {
    //     map_scale_ = scale_corrector.getScale();
    //     ROS_INFO("Map scale corrected!");
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // just to check scale
    //   }

    // }

    /*
     * Moved here from constructor - there is a small map
     * created just before start of the correction procedure
     */
    if (octomap_enabled_) {
      octomap_worker_thread_ = std::thread( [this] { octomapWorker(); } );
    }
    info_updater_thread_ = std::thread( [this] { camInfoUpdater(); } );

    SetFinish(false);
    while (WaitCycleStart()) {

        // only publish map, map updates and camera pose, if camera pose was updated
        // TODO: maybe there is a way to check if the map was updated
        if (isCamUpdated()) {

            publishMap();
            publishMapUpdates();
            if (octomap_enabled_)
            {
              // stashMapPoints(); // store current reference map points for the octomap worker
              stashMapPoints(false);
            }
        }
    }

    ROS_INFO("ROS publisher finished");
    SetFinish(true);
}

bool ROSPublisher::WaitCycleStart()
{
  if (!IPublisherThread::WaitCycleStart())
      return false;
  pub_rate_.sleep();
  return true;
}

void ROSPublisher::Update(Tracking *tracking)
{
  static std::mutex mutex;
  if (tracking == nullptr)
      return;

  publishState(tracking);

  // TODO: Make sure the camera TF is correctly aligned. See:
  // <http://docs.ros.org/jade/api/sensor_msgs/html/msg/Image.html>

  current_frame_time_ = ros::Time::now();
  // current_frame_time_ = ros::Time (tracking->mCurrentFrame.mTimeStamp);

  publishImage(tracking);
}

void ROSPublisher::clearCamTrajectoryCallback(const std_msgs::Bool::ConstPtr& msg) {
  if ( msg->data == true ) {
    clear_path_ = true;
  } else if ( msg->data == false ) {
    clear_path_ = false;
  }
}

void ROSPublisher::localizationModeCallback(const std_msgs::Bool::ConstPtr& msg) {

  if ( msg->data == true ) {
    localize_only = true;
  } else if ( msg->data == false ) {
    localize_only = false;
  }

}

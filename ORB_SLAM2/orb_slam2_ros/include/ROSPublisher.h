//
// Created by sebastiano on 8/18/16.
//

#ifndef ORB_SLAM2_ROSPUBLISHER_H
#define ORB_SLAM2_ROSPUBLISHER_H

#include "IPublisherThread.h"
#include "IFrameSubscriber.h"
#include "IMapPublisher.h"
#include "FrameDrawer.h"
#include "System.h"
#include "Map.h"
#include "LoopClosing.h"
#include "PublisherUtils.h"
#include "ScaleCorrector.h"

#include <chrono>
#include <mutex>

#include <ros/ros.h>
#include <ros/time.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>


#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>

#include <octomap/OcTree.h>

// #include <pcl/filters/passthrough.h>

#include <orb_slam2_ros/ORBState.h>

namespace ORB_SLAM2
{
    class Map;
    class Tracking;
    class LoopClosing;
}

class ROSPublisher :
    public ORB_SLAM2::IPublisherThread,
    public ORB_SLAM2::IMapPublisher,
    public ORB_SLAM2::IFrameSubscriber
{
public:

    static constexpr const char *DEFAULT_MAP_FRAME = "/orb_slam2/map";
    static constexpr const char *DEFAULT_CAMERA_FRAME = "/orb_slam2/camera";
    static constexpr const char *DEFAULT_IMAGE_TOPIC = "/orb_slam2/image";

    // `frequency` is max amount of messages emitted per second
    explicit ROSPublisher(
                ORB_SLAM2::Map *map,
                double frequency,
                ros::NodeHandle nh = ros::NodeHandle());

    virtual void Run() override;
    virtual void Update(ORB_SLAM2::Tracking*);

protected:

    bool WaitCycleStart();

private:

    void initializeParameters(ros::NodeHandle &nh);
    void stashMapPoints(bool all_map_points = false);
    void octomapWorker();
    void camInfoUpdater();

    void updateOctoMap();
    void integrateMapPoints(const std::vector<ORB_SLAM2::MapPoint*> &, const octomap::point3d &, const octomap::pose6d &, octomap::OcTree &);

    void octomapCutToOccupancyGrid(const octomap::OcTree& octree, nav_msgs::OccupancyGrid& map, nav_msgs::OccupancyGrid& map_erode, const double minZ_, const double maxZ_ );
    void octomapGradientToOccupancyGrid(const octomap::OcTree& octree, nav_msgs::OccupancyGrid& map, float max_height, int nb_erosions, float low_slope, float high_slope); // void octomapGradientToOccupancyGrid(const octomap::OcTree& octree, nav_msgs::OccupancyGrid& map, float max_height = GRADIENT_MAX_HEIGHT, int nb_erosions = GRADIENT_NB_EROSIONS, float low_slope = GRADIENT_LOW_SLOPE, float high_slope = GRADIENT_HIGH_SLOPE);

    void publishMap();
    void publishMapUpdates();
    void publishCurrentCameraPose(cv::Mat orbCameraPose);
    void publishOctomap();
    void publishState(ORB_SLAM2::Tracking *tracking);
    void publishImage(ORB_SLAM2::Tracking *tracking);
    void publishProjectedMap();
    void publishGradientMap();
    void publishCamTrajectory(bool finished_loop_closure);

    bool checkLoopClosure();
    geometry_msgs::TransformStamped getTransformBaseInMap(cv::Mat orb_Pmap_camera, ros::Time frame_time);
    geometry_msgs::PoseStamped getPoseBaseInMap(cv::Mat orb_Pmap_camera, ros::Time frame_time);


    ORB_SLAM2::FrameDrawer drawer_;

    /* Important: 'nh_' goes before the '*_pub_', because their
    ** construction relies on 'nh_'! */
    ros::NodeHandle   nh_;
    std::string name_of_node_;

    ros::Publisher    map_pub_, map_updates_pub_, image_pub_, odom_pub_,
                      state_pub_, state_desc_pub_, octomap_pub_,
                      projected_map_pub_, projected_morpho_map_pub_, gradient_map_pub_,
                      kf_pub_, kp_pub_, mp_pub_,
                      cam_pose_pub_,
                      trajectory_pub_,
                      loop_close_pub_, num_loop_closures_pub_;
    std::string       map_topic_, map_updates_topic_, image_topic_, 
                      state_topic_, state_disc_topic_, kp_topic_, kf_topic_, mp_topic_, 
                      loop_close_topic_, num_loop_closures_topic_, 
                      cam_pose_topic_, trajectory_topic_,
                      octomap_topic_, 
                      projected_map_topic_, projected_morpho_map_topic_,
                      gradient_map_topic_;

    ros::Rate         pub_rate_;
    std::thread       info_updater_thread_;

    bool              requires_subscriber_;
    bool              octomap_requires_subscriber_;
    bool              image_requires_subscriber_;

    geometry_msgs::PoseStamped cam_pose_;
    ros::Subscriber   clear_path_sub_;
    bool              clear_path_;
    void              clearCamTrajectoryCallback(const std_msgs::Bool::ConstPtr& msg);

    // -------- Mode
    ros::Subscriber   mode_sub_;
    void              localizationModeCallback(const std_msgs::Bool::ConstPtr& msg);
    void              checkMode();
    bool              localize_only;
    void              publishLoopState();
    void              publishNumLoopClosures();

    // -------- Feature Points
    void              publishKeypointsNb(const int &nb);
    void              publishUInt32Msg(const ros::Publisher &pub, const unsigned long &data);

    // -------- TF
    // initialization Transform listener
    boost::shared_ptr<tf2_ros::Buffer> tfBuffer_;
    boost::shared_ptr<tf2_ros::TransformListener> tfListener_;

    tf2::Transform TransformFromMat (cv::Mat position_mat);
    tf2::Transform TransformToTarget (tf2::Transform tf_in, std::string frame_in, std::string frame_target);
    sensor_msgs::PointCloud2 MapPointsToPointCloud (std::vector<ORB_SLAM2::MapPoint*> map_points);

    tf::TransformListener     tf_listener_;
    tf::TransformBroadcaster  camera_tf_pub_;
    tf::Vector3               camera_position_;
    float                     camera_height_;
    float                     camera_height_mult_;
    float                     camera_height_corrected_; // separate variable because there would be many * operations with transformPoint function

    int   lastBigMapChange_;
    bool  adjust_map_frame_;
    bool  align_map_to_cam_frame_;
    bool  perform_scale_correction_;
    float scaling_distance_;

    // ------ octomap
    octomap::OcTree octomap_;
    std::thread     octomap_worker_thread_;
    bool            octomap_tf_based_;
    bool            octomap_enabled_;
    bool            octomap_rebuild_;
    bool            clear_octomap_;
    float           octomap_rate_;
    bool            publish_octomap_;
    bool            publish_projected_map_;
    bool            publish_gradient_map_;

    // ------ PCL
    sensor_msgs::PointCloud2 all_map_points_;
    sensor_msgs::PointCloud2 reference_map_points_;
    octomap::Pointcloud pointcloud_map_points_;
    std::mutex          pointcloud_map_points_mutex_;
    int                 pointcloud_chunks_stashed_;
    float               map_scale_;

    // params for z-plane-based occupancy grid approach
    // pcl::PassThrough<sensor_msgs::PointCloud2> pass_x_;
    // pcl::PassThrough<sensor_msgs::PointCloud2> pass_y_;
    double  pointcloud_min_x_, pointcloud_max_x_;
    double  pointcloud_min_y_, pointcloud_max_y_;
    double  projection_min_height_;
    double  projection_max_height_;

    // params for morphological operations
    int  erode_se_size_;
    int  erode_nb_;
    int  open_se_size_;
    int  open_nb_;
    int  close_se_size_;
    int  close_nb_;
    int  erode2_nb_;
    int  erode2_se_size_;

    // params for gradient-based approach
    float   gradient_max_height_;
    float   gradient_low_slope_;
    float   gradient_high_slope_;
    int     gradient_nb_erosions_;

    // params for frames
    std::string map_frame_;
    std::string camera_frame_;
    std::string map_frame_adjusted_;
    std::string base_frame_;

    // time from header of current frame image
    ros::Time               current_frame_time_;

    // state republish rate
    orb_slam2_ros::ORBState orb_state_;
    ros::Time               last_state_publish_time_;
    float                   orb_state_republish_rate_;

    // loop closing republish
    bool                    loop_close_state_;
    int                     num_loop_closures_;
    ros::Time               last_loop_close_publish_time_;
    float                   loop_close_republish_rate_;

};

#endif //ORB_SLAM2_ROSPUBLISHER_H

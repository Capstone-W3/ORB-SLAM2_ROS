/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>

#include "System.h"
#include <include/ROSPublisher.h>
#include <include/ROSSystemBuilder.h>
#include "utils.h"

using namespace std;

struct ImageGrabber
{
    ORB_SLAM2::System* mpSLAM;

    void GrabImage(const sensor_msgs::ImageConstPtr& msg)
    {
        // Copy the ros image message to cv::Mat.
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();

    if(argc > 1) {
        ROS_WARN ("Arguments supplied via command line are neglected.");
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    double freq;
    std::string image_topic, voc_file, camera_settings_file;

    // Set up a namespace for topics
    ros::NodeHandle nh("/orb_slam2");
    std::string name_of_node_ = ros::this_node::getName();

    nh.param<double>(name_of_node_ + "/topic/freq", freq, 100.0);
    nh.param<std::string>(name_of_node_ + "/topic/image_topic", image_topic, ROSPublisher::DEFAULT_IMAGE_TOPIC);
    nh.param<std::string>(name_of_node_ + "/voc_file", voc_file);
    nh.param<std::string>(name_of_node_ + "/camera_settings_file", camera_settings_file);

    cout << "Loading BoW vocabulary..." << endl;
    ORB_SLAM2::System SLAM( make_unique<ROSSystemBuilder>(
                            voc_file,
                            camera_settings_file,
                            ORB_SLAM2::System::MONOCULAR,
                            freq,
                            nh));

    ImageGrabber igb {&SLAM};
    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe(image_topic, 1, &ImageGrabber::GrabImage, &igb);

    SLAM.Start();
    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

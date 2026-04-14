/*******************************************************************************
* Copyright (c) 2023 Orbbec 3D Technology, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include "orbbec_camera/d2c_viewer.h"

namespace orbbec_camera {
D2CViewer::D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  rgb_sub_.subscribe(nh_, "color/image_raw", 1);
  depth_sub_.subscribe(nh_, "depth/image_raw", 1);
  sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(10), rgb_sub_,
                                                                        depth_sub_);
  sync_->registerCallback(boost::bind(&D2CViewer::messageCallback, this, _1, _2));
  d2c_viewer_pub_ = nh_.advertise<sensor_msgs::Image>("depth_to_color/image_raw", 1);
  d2c_overlay_pub_ = nh_.advertise<sensor_msgs::Image>("depth_to_color_overlay/image_raw", 1);
}
D2CViewer::~D2CViewer() = default;

void D2CViewer::messageCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                const sensor_msgs::ImageConstPtr& depth_msg) {
  if (rgb_msg->width != depth_msg->width || rgb_msg->height != depth_msg->height) {
    ROS_ERROR("rgb and depth image size not match(%d, %d) vs (%d, %d)", rgb_msg->width,
              rgb_msg->height, depth_msg->width, depth_msg->height);
    return;
  }
  auto depth_img_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  auto depth_to_color_msg =
      cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::TYPE_16UC1,
                         depth_img_ptr->image)
          .toImageMsg();
  d2c_viewer_pub_.publish(depth_to_color_msg);

  auto rgb_img_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
  cv::Mat valid_mask = depth_img_ptr->image > 0;
  if (cv::countNonZero(valid_mask) == 0) {
    return;
  }

  cv::Mat depth_normalized;
  cv::normalize(depth_img_ptr->image, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1,
                valid_mask);
  cv::Mat depth_color_bgr;
  cv::applyColorMap(depth_normalized, depth_color_bgr, cv::COLORMAP_JET);
  cv::Mat depth_color_rgb;
  cv::cvtColor(depth_color_bgr, depth_color_rgb, cv::COLOR_BGR2RGB);

  cv::Mat overlay;
  cv::addWeighted(rgb_img_ptr->image, 0.65, depth_color_rgb, 0.35, 0.0, overlay);
  auto overlay_msg = cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::RGB8,
                                        overlay)
                         .toImageMsg();
  d2c_overlay_pub_.publish(overlay_msg);
}

}  // namespace orbbec_camera

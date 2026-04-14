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

#pragma once
#include <array>
#include <functional>
#include <boost/optional.hpp>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/core.hpp>
#include <sensor_msgs/Image.h>

#include "types.h"
#include "utils.h"

namespace orbbec_camera {
class D2CViewer {
 public:
  using CameraParamsProvider = std::function<boost::optional<OBCameraParam>()>;

  D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
  D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
            CameraParamsProvider camera_params_provider);
  ~D2CViewer();
  void messageCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                       const sensor_msgs::ImageConstPtr& depth_msg);

 private:
  bool ensureCameraParams();
  void buildDepthToColorTransform(const OBCameraParam& camera_param);
  bool alignDepthToColor(const cv::Mat& depth_image, const cv::Mat& rgb_image,
                         cv::Mat& aligned_depth) const;

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  CameraParamsProvider camera_params_provider_;
  boost::optional<OBCameraParam> camera_params_;
  int depth_width_ = 0;
  int depth_height_ = 0;
  int color_width_ = 0;
  int color_height_ = 0;
  float depth_fx_ = 0.0f;
  float depth_fy_ = 0.0f;
  float depth_cx_ = 0.0f;
  float depth_cy_ = 0.0f;
  float color_fx_ = 0.0f;
  float color_fy_ = 0.0f;
  float color_cx_ = 0.0f;
  float color_cy_ = 0.0f;
  std::array<float, 9> depth_to_color_rotation_{};
  std::array<float, 3> depth_to_color_translation_{};
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  using MySyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;
  ros::Publisher d2c_viewer_pub_;
  ros::Publisher d2c_overlay_pub_;
};
}  // namespace orbbec_camera

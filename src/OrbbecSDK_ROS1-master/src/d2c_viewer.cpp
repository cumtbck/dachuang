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

#include <cmath>
#include <limits>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "orbbec_camera/d2c_viewer.h"

namespace orbbec_camera {
D2CViewer::D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : D2CViewer(nh, nh_private, CameraParamsProvider()) {}

D2CViewer::D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
                     CameraParamsProvider camera_params_provider)
    : nh_(nh),
      nh_private_(nh_private),
      camera_params_provider_(std::move(camera_params_provider)) {
  rgb_sub_.subscribe(nh_, "color/image_raw", 1);
  depth_sub_.subscribe(nh_, "depth/image_raw", 1);
  sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(10), rgb_sub_,
                                                                        depth_sub_);
  sync_->registerCallback(boost::bind(&D2CViewer::messageCallback, this, _1, _2));
  d2c_viewer_pub_ = nh_.advertise<sensor_msgs::Image>("depth_to_color/image_raw", 1);
  d2c_overlay_pub_ = nh_.advertise<sensor_msgs::Image>("depth_to_color_overlay/image_raw", 1);
}
D2CViewer::~D2CViewer() = default;

bool D2CViewer::ensureCameraParams() {
  if (camera_params_) {
    return true;
  }
  if (!camera_params_provider_) {
    ROS_ERROR_STREAM("D2CViewer camera parameter provider is not configured");
    return false;
  }

  auto camera_param = camera_params_provider_();
  if (!camera_param) {
    return false;
  }

  buildDepthToColorTransform(*camera_param);
  camera_params_ = camera_param;
  return true;
}

void D2CViewer::buildDepthToColorTransform(const OBCameraParam& camera_param) {
  depth_width_ = camera_param.depthIntrinsic.width;
  depth_height_ = camera_param.depthIntrinsic.height;
  color_width_ = camera_param.rgbIntrinsic.width;
  color_height_ = camera_param.rgbIntrinsic.height;

  depth_fx_ = camera_param.depthIntrinsic.fx;
  depth_fy_ = camera_param.depthIntrinsic.fy;
  depth_cx_ = camera_param.depthIntrinsic.cx;
  depth_cy_ = camera_param.depthIntrinsic.cy;
  color_fx_ = camera_param.rgbIntrinsic.fx;
  color_fy_ = camera_param.rgbIntrinsic.fy;
  color_cx_ = camera_param.rgbIntrinsic.cx;
  color_cy_ = camera_param.rgbIntrinsic.cy;

  Eigen::Matrix3f color_to_depth_rotation;
  color_to_depth_rotation << camera_param.transform.rot[0], camera_param.transform.rot[3],
      camera_param.transform.rot[6], camera_param.transform.rot[1], camera_param.transform.rot[4],
      camera_param.transform.rot[7], camera_param.transform.rot[2], camera_param.transform.rot[5],
      camera_param.transform.rot[8];
  Eigen::Vector3f color_to_depth_translation(camera_param.transform.trans[0],
                                             camera_param.transform.trans[1],
                                             camera_param.transform.trans[2]);

  const Eigen::Matrix3f depth_to_color_rotation = color_to_depth_rotation.transpose();
  const Eigen::Vector3f depth_to_color_translation =
      -depth_to_color_rotation * color_to_depth_translation;

  depth_to_color_rotation_[0] = depth_to_color_rotation(0, 0);
  depth_to_color_rotation_[1] = depth_to_color_rotation(0, 1);
  depth_to_color_rotation_[2] = depth_to_color_rotation(0, 2);
  depth_to_color_rotation_[3] = depth_to_color_rotation(1, 0);
  depth_to_color_rotation_[4] = depth_to_color_rotation(1, 1);
  depth_to_color_rotation_[5] = depth_to_color_rotation(1, 2);
  depth_to_color_rotation_[6] = depth_to_color_rotation(2, 0);
  depth_to_color_rotation_[7] = depth_to_color_rotation(2, 1);
  depth_to_color_rotation_[8] = depth_to_color_rotation(2, 2);

  depth_to_color_translation_[0] = depth_to_color_translation.x();
  depth_to_color_translation_[1] = depth_to_color_translation.y();
  depth_to_color_translation_[2] = depth_to_color_translation.z();
}

bool D2CViewer::alignDepthToColor(const cv::Mat& depth_image, const cv::Mat& rgb_image,
                                  cv::Mat& aligned_depth) const {
  if (!camera_params_ || depth_image.empty() || rgb_image.empty()) {
    return false;
  }
  if (depth_width_ <= 0 || depth_height_ <= 0 || color_width_ <= 0 || color_height_ <= 0) {
    return false;
  }
  if (depth_fx_ <= 0.0f || depth_fy_ <= 0.0f || color_fx_ <= 0.0f || color_fy_ <= 0.0f) {
    return false;
  }

  const float depth_width_scale = static_cast<float>(depth_image.cols) /
                                  static_cast<float>(depth_width_);
  const float depth_height_scale = static_cast<float>(depth_image.rows) /
                                   static_cast<float>(depth_height_);
  const float color_width_scale = static_cast<float>(rgb_image.cols) /
                                  static_cast<float>(color_width_);
  const float color_height_scale = static_cast<float>(rgb_image.rows) /
                                   static_cast<float>(color_height_);

  const float depth_fx = depth_fx_ * depth_width_scale;
  const float depth_fy = depth_fy_ * depth_height_scale;
  const float depth_cx = depth_cx_ * depth_width_scale;
  const float depth_cy = depth_cy_ * depth_height_scale;
  const float color_fx = color_fx_ * color_width_scale;
  const float color_fy = color_fy_ * color_height_scale;
  const float color_cx = color_cx_ * color_width_scale;
  const float color_cy = color_cy_ * color_height_scale;

  aligned_depth = cv::Mat::zeros(rgb_image.rows, rgb_image.cols, CV_16UC1);
  cv::Mat z_buffer(rgb_image.rows, rgb_image.cols, CV_32FC1,
                   cv::Scalar(std::numeric_limits<float>::max()));

  const float max_depth_value = static_cast<float>(std::numeric_limits<uint16_t>::max());
  for (int y = 0; y < depth_image.rows; ++y) {
    const auto* depth_row = depth_image.ptr<uint16_t>(y);
    for (int x = 0; x < depth_image.cols; ++x) {
      const uint16_t depth_value = depth_row[x];
      if (depth_value == 0) {
        continue;
      }

      const float z = static_cast<float>(depth_value);
      const float x_depth = (static_cast<float>(x) - depth_cx) * z / depth_fx;
      const float y_depth = (static_cast<float>(y) - depth_cy) * z / depth_fy;

      const float x_color = depth_to_color_rotation_[0] * x_depth +
                            depth_to_color_rotation_[1] * y_depth +
                            depth_to_color_rotation_[2] * z + depth_to_color_translation_[0];
      const float y_color = depth_to_color_rotation_[3] * x_depth +
                            depth_to_color_rotation_[4] * y_depth +
                            depth_to_color_rotation_[5] * z + depth_to_color_translation_[1];
      const float z_color = depth_to_color_rotation_[6] * x_depth +
                            depth_to_color_rotation_[7] * y_depth +
                            depth_to_color_rotation_[8] * z + depth_to_color_translation_[2];

      if (!std::isfinite(z_color) || z_color <= 0.0f || z_color > max_depth_value) {
        continue;
      }

      const float u = color_fx * x_color / z_color + color_cx;
      const float v = color_fy * y_color / z_color + color_cy;
      const int u_index = static_cast<int>(std::lround(u));
      const int v_index = static_cast<int>(std::lround(v));
      if (u_index < 0 || u_index >= rgb_image.cols || v_index < 0 || v_index >= rgb_image.rows) {
        continue;
      }

      float* z_row = z_buffer.ptr<float>(v_index);
      uint16_t* aligned_row = aligned_depth.ptr<uint16_t>(v_index);
      if (z_color < z_row[u_index]) {
        z_row[u_index] = z_color;
        aligned_row[u_index] = static_cast<uint16_t>(std::lround(z_color));
      }
    }
  }

  return true;
}

void D2CViewer::messageCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                const sensor_msgs::ImageConstPtr& depth_msg) {
  if (!ensureCameraParams()) {
    ROS_WARN_STREAM_THROTTLE(5.0, "Waiting for camera parameters before publishing depth_to_color");
    return;
  }

  auto depth_img_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  auto rgb_img_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);

  cv::Mat aligned_depth;
  if (!alignDepthToColor(depth_img_ptr->image, rgb_img_ptr->image, aligned_depth)) {
    ROS_WARN_STREAM_THROTTLE(5.0, "Failed to align depth to color frame");
    return;
  }

  auto depth_to_color_msg =
      cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::TYPE_16UC1, aligned_depth)
          .toImageMsg();
  d2c_viewer_pub_.publish(depth_to_color_msg);

  cv::Mat valid_mask = aligned_depth > 0;
  if (cv::countNonZero(valid_mask) == 0) {
    return;
  }

  cv::Mat depth_normalized;
  cv::normalize(aligned_depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1, valid_mask);
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

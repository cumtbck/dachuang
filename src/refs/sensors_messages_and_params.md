# 传感器消息类型与可配置参数汇总

本文档汇总仓库中三个传感器驱动（MS200 激光雷达、Wheeltec 毫米波雷达、Orbbec 深度相机）发布的消息类型与可以通过启动参数/rosparam 设置的参数及其含义。

**说明**：话题名与参数名可能在对应 launch 文件中被 remap/替换，文中写出驱动默认的 topic/param 名称与含义。

**一、MS200 激光雷达（oradar_scan）**
- 节点／启动：由 `oradar_scan` 节点启动（包 `oradar_lidar`），常见 launch：`ms200_scan.launch` / `ms200_scan_view.launch`。

- 默认话题：`/scan`（可通过 `scan_topic` 参数修改）
- 发布的消息类型：`sensor_msgs/LaserScan`
  - 重要字段及含义：
    - `header`：时间戳与 `frame_id`（例如 launch 中 `frame_id`，默认示例为 `laser` 或 `laser_frame`）
    - `angle_min`（rad）：扫描起始角
    - `angle_max`（rad）：扫描终止角
    - `angle_increment`（rad）：相邻测量点角度增量
    - `scan_time`（s）：一次扫描时间
    - `time_increment`（s）：点与点之间的时间间隔
    - `range_min` / `range_max`（m）：有效测距范围
    - `ranges[]`（m）：每个采样角度对应的距离值（无效或超限点可能为 NaN 或 0）
    - `intensities[]`：回波强度（设备相关）

- 驱动可配置参数（launch 中或 rosparam）：
  - `device_model`（string）: 设备型号（例如 "MS200"）
  - `frame_id`（string）: 消息 `header.frame_id` 的值（用于 RViz/TF 绑定）
  - `scan_topic`（string）: 激光扫描发布的话题名（默认 `/scan`）
  - `port_name`（string）: 串口设备路径（例如 `/dev/oradar` 或 `/dev/ttyACM0`）
  - `baudrate`（int）: 串口波特率（例如 230400）
  - `angle_min`（double，度/驱动内转换为弧度）: 采集的最小角度（单位度，驱动中会转换）
  - `angle_max`（double，度）: 采集的最大角度
  - `range_min`（double，m）: 有效最小测距
  - `range_max`（double，m）: 有效最大测距
  - `clockwise`（bool）: 点云角度方向（true 顺时针，false 逆时针）
  - `motor_speed`（int）: 电机转速（驱动示例范围 5~15Hz，默认 10）

- RViz 可视化：直接使用 `LaserScan` 显示（Topic = `/scan`），或将其转换为点云查看。

**二、Wheeltec 毫米波雷达（wheeltec_radar_node / turn_on_radar）**
- 节点／启动：由 `wheeltec_radar_node`（执行文件 `wheeltec_radar_node`）在 `wheeltec_radar` 包中启动，launch：`wheeltec_radar.launch`。

- 发布话题与消息类型（驱动源码与 msg 定义）：
  1. `/radarscan` : `wheeltec_radar/RadarDetectionArray`
     - `RadarDetectionArray.msg`:
       - `std_msgs/Header header`
       - `wheeltec_radar/RadarDetection[] detections`
     - `RadarDetection.msg` 字段：
       - `uint16 detection_id`：雷达给出的检测 id（若无则为序号）
       - `geometry_msgs/Point position`：目标位置（仅 x,y 有效，z 为 0）
       - `geometry_msgs/Vector3 velocity`：速度分量（x,y 有效）
       - `float64 amplitude`：检测幅值（dB，设备相关）
  2. `/radartrack` : `wheeltec_radar/RadarTrackArray`
     - `RadarTrackArray.msg`:
       - `std_msgs/Header header`
       - `wheeltec_radar/RadarTrack[] tracks`
     - `RadarTrack.msg` 字段：
       - `uint16 track_id`：轨迹 id
       - `geometry_msgs/Polygon track_shape`：目标的二维多边形表示（一组 `Point32`）
       - `geometry_msgs/Vector3 linear_velocity`：线速度（x,y 有效）
       - `geometry_msgs/Vector3 linear_acceleration`：线加速度（x 有效）
  3. `/radarpointcloud` : `sensor_msgs/PointCloud2`
     - 驱动代码中将 `RadarDetectionArray` 转换为 `PointCloud2` 并发布，以便直接在 RViz 中使用点云显示。

- 驱动可配置参数（节点私有参数，launch/nh.param 读取）：
  - `usart_port_name`（string）: 串口设备名称（例如 `/dev/wheeltec_mmwave_radar` 或 `/dev/ttyUSB0`）
  - `serial_baud_rate`（int）: 串口波特率（默认示例 115200）

- 字段与含义注意：
  - `position` 单位/坐标系：源码中对串口原始数据做了坐标与单位转换，并以 ROS 右手坐标系发布，`z` 被置为 0；发布的 `header.frame_id` 为 "radar"。请以驱动源码为准确认单位（源码通过算术转换得到米/厘米等）。
  - `track_shape` 在源码中用 `geometry_msgs/Polygon` 表示小方框，用于表示目标尺寸/边界。

- RViz 可视化建议：
  - 直接显示 `/radarpointcloud`（`PointCloud2`）即可看到检测点；
  - 若要展示 `RadarTrackArray`（多边形、ID、速度箭头），建议写一个转换节点将 `RadarTrackArray` → `visualization_msgs/MarkerArray`（画多边形和箭头），或者将 `RadarDetectionArray` → `PointCloud2`（已实现）并将速度信息用 `Marker` 表示。

**三、Orbbec 深度相机（orbbec_camera / orbbec_camera_node）**
- 节点／启动：`orbbec_camera_node`（包 `orbbec_camera`），仓库提供多种 launch（例如 `dabai_dcw2.launch`、`dabai.launch`、`astra.launch` 等）。

- 常见（默认/可选）话题：
  - `/<camera>/color/image_raw` : `sensor_msgs/Image`（彩色图像）
  - `/<camera>/color/camera_info` : `sensor_msgs/CameraInfo`（彩色相机内参）
  - `/<camera>/depth/image_raw` : `sensor_msgs/Image`（深度图，编码多为 16UC1 或 32FC1，可依据驱动）
  - `/<camera>/depth/camera_info` : `sensor_msgs/CameraInfo`（深度相机内参）
  - `/<camera>/depth/points` : `sensor_msgs/PointCloud2`（点云，需 `enable_point_cloud=true`）
  - `/<camera>/depth_registered/points` : `sensor_msgs/PointCloud2`（彩色对齐点云，需 `enable_colored_point_cloud=true`）
  - `/camera/accel/sample`, `/camera/gyro/sample` 等：IMU/加速度/陀螺相关（依驱动与参数启用）

- 重要消息字段/含义：
  - `sensor_msgs/Image`：`header`、`height`、`width`、`encoding`、`step`、`data`（彩色为 RGB/BGR，深度为深度编码）
  - `sensor_msgs/CameraInfo`：`K`（内参矩阵）、`D`（畸变系数）、`R`（校正矩阵）、`P`（投影矩阵）——用于像素↔三维点投影
  - `sensor_msgs/PointCloud2`：点云字段通常包含 `x,y,z`（以米为单位），可能包含 `rgb` 或 `intensity`

- 驱动可配置参数（launch args / rosparam，摘自 README 与 launch）
  - `camera_name`（string）: 相机名称空间前缀
  - `depth_registration`（bool）: 是否启用深度对齐到彩色（硬件/软件）
  - `serial_number`（string）: 设备序列号（多相机时用）
  - `usb_port`（string）: USB 端口标识（多相机时用）
  - `device_num`（int）: 设备编号/数量
  - `vendor_id` / `product_id`（hex/string）: USB 供应商/产品 id
  - `enable_point_cloud`（bool）: 是否发布 `depth/points`
  - `enable_colored_point_cloud`（bool）: 是否发布带颜色的对齐点云 `depth_registered/points`
  - `enable_d2c_viewer`（bool）: 发布 D2C 叠加图像（用于调试）
  - `enable_pipeline`（bool）: 是否启用处理流水线
  - `enable_soft_filter`（bool）: 软件深度滤波开关
  - `connection_delay`（int，ms）: 设备重连延迟
  - 彩色分辨率/帧率：`color_width`、`color_height`、`color_fps`
  - 彩色开关与格式：`enable_color`（bool）、`color_format`（例如 MJPG）、`flip_color`（bool）、`enable_color_auto_exposure`（bool）
  - 深度分辨率/帧率：`depth_width`、`depth_height`、`depth_fps`、`depth_format`（例如 Y11/Y10 等）、`flip_depth`、`enable_depth`
  - IR（红外）设置：`ir_width`、`ir_height`、`ir_fps`、`enable_ir`、`ir_format`、`flip_ir`、`enable_ir_auto_exposure`
  - `publish_tf`（bool）: 是否发布相机的 TF 变换（需在 RViz 中正确显示相机坐标）
  - `tf_publish_rate`（double）: TF 发布频率
  - `ir_info_uri` / `color_info_uri`：外部 camera info 文件路径（可选）
  - `log_level`（string）: 驱动日志等级（`none`/`info`/`debug`/`warn`/`fatal`）
  - 还有 IMU/加速度相关开关：`enable_accel`、`enable_gyro`、`enable_sync_output_accel_gyro` 等（驱动 README 中列出）

- RViz 可视化：
  - 使用 `Image` 或 `Camera` 显示彩色图像；`Camera` 插件配合 `CameraInfo` 可在视图中显示相机投影与深度叠加。
  - 使用 `PointCloud2` 显示深度点云（`depth/points` 或 `depth_registered/points`），可用 `rgb` 字段做着色。
  - 若发布 `sensor_msgs/Imu`，可使用 RViz 的 IMU 显示。

---

如果需要，我可以：
- 生成一个 `rosmsg show` 风格的每个消息完整字段列表示例；或
- 在仓库中添加一个示例节点，将 `RadarTrackArray` 转为 `visualization_msgs/MarkerArray`（并将文件添加到 `wheeltec_radar` 包），以便直接在 RViz 中显示轨迹多边形与速度箭头。

文档已保存为： [docs/sensors_messages_and_params.md](docs/sensors_messages_and_params.md)

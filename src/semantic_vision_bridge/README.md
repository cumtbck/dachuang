# semantic_vision_bridge

Jetson-side ROS package for sending RGB frames to Zynq, receiving detection results, converting them into semantic obstacle point clouds, and enforcing traffic light stop behavior.

## Nodes

- `rgb_sender.py`: JPEG-encodes `/camera/color/image_raw` and streams to Zynq over UDP with timestamps.
- `semantic_solver.py`: Receives detection JSON, matches depth frames, projects to 3D, publishes `/semantic_obstacles` and `/traffic_light_status`.
- `traffic_manager.py`: Gates `/cmd_vel_nav` into `/cmd_vel` when red light or safety stop is active.

## Launch

```bash
roslaunch semantic_vision_bridge semantic_bridge.launch zynq_ip:=192.168.1.10 zynq_port:=5000 listen_port:=5001
```

Adjust parameters in `config/rgb_sender.yaml`, `config/semantic_solver.yaml`, and `config/traffic_manager.yaml` for topics and thresholds.

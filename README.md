# Semantic Mapping and Navigation using a TurtleBot 4

Final Year Project — BSc Computer Science & Software Engineering  
Maynooth University | Student: Olatilewa Sam Ishola | Supervisor: Prof. John McDonald

## Overview

A ROS2-based semantic mapping and navigation framework that enables a TurtleBot 4 
to detect, localise, and autonomously navigate toward objects in an unknown indoor 
environment. The system fuses LiDAR-based SLAM with YOLOv8 object detection and 
stereo depth data to construct an object-level semantic map.

## System Architecture

The pipeline consists of five ROS2 nodes:

- **yolo_onnx_node** — YOLOv8n inference on RGB camera stream
- **semantic_projection** — back-projects 2D detections into 3D map coordinates
- **semantic_registry** — maintains stable object tracks on the occupancy map
- **semantic_marker_publisher** — visualises objects in RViz + sends NAV2 goals
- **yolo_viz** — debug bounding box visualiser

## Packages

| Package | Description |
|---|---|
| `semnav_perception` | YOLO detection and depth projection nodes |
| `semnav_mapping` | Registry, marker publisher, and navigation interface |
| `semnav_msgs` | Custom SemanticObject ROS2 message definition |
| `semnav_camera` | Camera launch and configuration files |

## Hardware

- Clearpath TurtleBot 4
- Luxonis OAK-D Pro RGB-D Camera
- 2D LiDAR (onboard)

## Dependencies

- ROS2 Jazzy
- SLAM Toolbox
- NAV2
- ONNX Runtime
- OpenCV
- Python 3.10+

## Running the System

Launch SLAM on the robot (SSH):
```bash
ros2 launch turtlebot4_navigation slam.launch.py
ros2 launch turtlebot4_navigation nav2.launch.py
```

Launch the full semantic pipeline:
```bash
colcon build --packages-select semnav_perception semnav_mapping
source install/setup.bash
ros2 launch semnav_mapping semnav.launch.py
```

Send a navigation goal:
```bash
ros2 topic pub --once /semnav/navigate_to_class std_msgs/msg/String "data: 'backpack'"
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `score_threshold` | 0.15 | YOLO confidence threshold |
| `iou_threshold` | 0.45 | NMS IoU threshold |
| `allowed_classes` | `['chair','person','backpack']` | Classes to track |
| `merge_radius` | 0.5m | Registry merge distance |
| `stale_seconds` | 5.0 | Track expiry timeout |
| `standoff_m` | 0.6m | NAV2 goal standoff distance |

## Author

Olatilewa Sam Ishola — Maynooth University 2026
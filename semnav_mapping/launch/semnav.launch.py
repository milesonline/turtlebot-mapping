"""
semnav.launch.py  —  master launch file
Starts: yolo, projection, registry, marker_publisher

Usage:
  ros2 launch semnav_mapping semnav.launch.py
  ros2 launch semnav_mapping semnav.launch.py score_threshold:=0.15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Launch arguments 
    score_threshold_arg = DeclareLaunchArgument(
        'score_threshold', default_value='0.15',
        description='YOLO confidence threshold'
    )
    map_frame_arg = DeclareLaunchArgument(
        'map_frame', default_value='map',
        description='Map frame ID'
    )

    score_threshold = LaunchConfiguration('score_threshold')
    map_frame       = LaunchConfiguration('map_frame')

    # Nodes 

    yolo_node = Node(
        package='semnav_perception',
        executable='yolo_onnx',
        name='yolo_onnx_node',
        output='screen',
        parameters=[{
            'score_threshold': score_threshold,
            'iou_threshold':   0.45,
        }]
    )

    projection_node = Node(
        package='semnav_perception',
        executable='semantic_projection',
        name='semantic_projection',
        output='screen',
        parameters=[{
            'map_frame':     map_frame,
            'use_latest_tf': True,
            'min_depth_m':   0.2,
            'max_depth_m':   6.0,
        }]
    )

    registry_node = Node(
        package='semnav_mapping',
        executable='semantic_registry',
        name='semantic_registry',
        output='screen',
        parameters=[{
            'mode':            'map',
            'merge_radius':    0.5,
            'stale_seconds':   5.0,
            'publish_hz':      2.0,
            'allowed_classes': ['chair', 'person', 'backpack'],
        }]
    )

    marker_node = Node(
        package='semnav_mapping',
        executable='semantic_marker_publisher',
        name='semantic_marker_publisher',
        output='screen',
        parameters=[{
            'map_frame':  map_frame,
            'standoff_m': 0.6,
        }]
    )

    return LaunchDescription([
        score_threshold_arg,
        map_frame_arg,
        yolo_node,
        projection_node,
        registry_node,
        marker_node,
    ])
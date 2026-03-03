from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = LaunchConfiguration('image_topic')
    model_path = LaunchConfiguration('model_path')
    score_threshold = LaunchConfiguration('score_threshold')
    allowed_classes = LaunchConfiguration('allowed_classes')

    return LaunchDescription([
        DeclareLaunchArgument(
        'image_topic',
        default_value='/oakd/rgb/preview/image_raw'
        ),
        DeclareLaunchArgument(
        'model_path',
        default_value='~/ros_ws/src/semnav_perception/models/yolov8n.onnx'
        ),
        DeclareLaunchArgument(
        'score_threshold',
        default_value='0.30'
        ),
        DeclareLaunchArgument(
        'allowed_classes',
        default_value="['chair','couch','person','dining table']"
        ),

    # YOLO detections -> /semnav/detections (vision_msgs/Detection2DArray)
    Node(
        package='semnav_perception',
        executable='yolo_onnx',
        name='yolo_onnx',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'model_path': model_path,
            'score_threshold': score_threshold,
        }],
    ),

    # Detection2DArray -> SemanticObject raw observations (/semnav/objects_raw)
    Node(
        package='semnav_perception',
        executable='semantic_projection',
        name='semantic_projection',
        output='screen',
    ),

# Registry: /semnav/objects_raw -> /semnav/objects (filtered + stable)
    Node(
        package='semnav_mapping',
        executable='semantic_registry',
        name='semantic_registry',
        output='screen',
        parameters=[{
            'mode': 'image',
            'allowed_classes': allowed_classes,
            'merge_radius': 60.0,
            'stale_seconds': 2.0,
            'publish_hz': 2.0,
        }],
    ),
])
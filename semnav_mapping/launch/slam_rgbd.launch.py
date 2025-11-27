from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    start_rgbd_odom = LaunchConfiguration('start_rgbd_odom', default='true')


    cfg = LaunchConfiguration(
        'config',
        default=os.path.join(
            get_package_share_directory('semnav_mapping'),
            'config',
            'rtabmap_rgbd.yaml'
        )
    )
    # Default camera topics (DepthAI TB4) override via args if needed
    rgb      = LaunchConfiguration('rgb', default='/oakd/rgb/image_raw')
    depth    = LaunchConfiguration('depth', default='/oakd/depth/image_raw')
    rgb_info = LaunchConfiguration('rgb_info', default='/oakd/rgb/camera_info')
    depth_info = LaunchConfiguration('depth_info', default='/oakd/depth/camera_info')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_rgbd_odom', default_value='true'),
        DeclareLaunchArgument('config', default_value=cfg),
        DeclareLaunchArgument('rgb', default_value=rgb),
        DeclareLaunchArgument('depth', default_value=depth),
        DeclareLaunchArgument('rgb_info', default_value=rgb_info),
        DeclareLaunchArgument('depth_info', default_value=depth_info),

        #RGBD ODOM (OPTIONAL)
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('rgbd/image', rgb),
                ('depth/image', depth),
                ('rgb/camera_info', rgb_info),
                ('depth/camera_info', depth_info),
            ],
            condition=IfCondition(start_rgbd_odom)
        ),

        #VSLAM
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}, cfg],
            remappings=[
                ('rgb/image', rgb),
                ('depth/image', depth),
                ('rgb/camera_info', rgb_info),
                ('depth/camera_info', depth_info),
                ('odom', '/odom'), # wheel odom
            ]
        )
    ])

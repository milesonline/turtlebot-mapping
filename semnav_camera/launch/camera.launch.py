from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    config_path = LaunchConfiguration(
        'config',
        default=os.path.join(
            get_package_share_directory('semnav_camera'),
            'config',
            'oakd_tb4.yaml'
        )
    )

    start_camera = LaunchConfiguration('start_camera', default='false')
    start_sim_cam = LaunchConfiguration('start_sim_cam', default ='false')


    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('config', default_value=config_path),
        DeclareLaunchArgument('start_camera', default_value='false'),
        DeclareLaunchArgument('start_sim_cam', default_value='false'),
        DeclareLaunchArgument('start_image_sub', default_value='false'),
        DeclareLaunchArgument('start_rviz', default_value='false'),

        # Real TB4 camera(depthai)
        Node(
            package="depthai_ros_driver",
            executable="camera_node",
            name="oakd",
            output="screen",
            parameters=[
                {'use_sim_time': use_sim_time},
                config_path
            ],
            condition=IfCondition(start_camera)
        ),

        #Simulated Camera (dummy image publisher)
        Node(
            package='image_tools',
            executable='cam2image',
            name='sim_cam',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'burger_mode': True}],
            remappings=[('image', '/camera/image_raw')],
            condition=IfCondition(start_sim_cam)
        ),

        # Static transform for simulation
        # Provides a fixed TF link between the robot base and the simulated camera frame.
        # This ensures that /camera_frame is correctly positioned relative to /base_link
        # during simulation (since the real robot publishes its own transforms).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='sim_cam_tf',
            arguments=['0', '0', '0.3', '0', '0', '0', 'base_link', 'camera_frame'],
            condition=IfCondition(start_sim_cam)
        ),

        # Start image subscriber
        Node(
            package='semnav_camera',
            executable='image_sub',
            name='image_sub',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('start_image_sub'))
        ),

        # Start rviz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(
                get_package_share_directory('semnav_camera'),
                'rviz',
                'semnav_sim.rviz'
            )],
            condition =IfCondition(LaunchConfiguration('start_rviz'))
        )
    ])
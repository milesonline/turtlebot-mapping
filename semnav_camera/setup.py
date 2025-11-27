from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'semnav_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/semnav_camera']),
        ('share/semnav_camera', ['package.xml']),
        ('share/semnav_camera/launch', glob('launch/*.launch.py')),
        ('share/semnav_camera/config', glob('config/*.yaml')),
        ('share/semnav_camera/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sam_i',
    maintainer_email='sam_i@todo.todo',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_sub = semnav_camera.image_sub:main',
        ],
    },
)

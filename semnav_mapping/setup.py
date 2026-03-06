from setuptools import find_packages, setup
from glob import glob

package_name = 'semnav_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/semnav_mapping']),
        ('share/semnav_mapping', ['package.xml']),
        ('share/semnav_mapping/launch', glob('launch/*.launch.py')),
        ('share/semnav_mapping/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sam_i',
    maintainer_email='olatilewa.ishola.2021@mumail.ie',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'semantic_registry = semnav_mapping.semantic_registry_node:main',
            'semantic_marker_publisher = semnav_mapping.semantic_marker_publisher:main',
        ],
    },
)

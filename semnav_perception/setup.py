from setuptools import find_packages, setup

package_name = 'semnav_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'yolo_stub = semnav_perception.yolo_stub_node:main',
            'yolo_onnx = semnav_perception.yolo_onnx_node:main',
            'yolo_viz = semnav_perception.yolo_viz_node:main',
            'semantic_projection = semnav_perception.semantic_projection_node:main',
        ],
    },
)

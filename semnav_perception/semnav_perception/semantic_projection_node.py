import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# custom msg
from semnav_msgs.msg import SemanticObject


class SemanticProjectionNode(Node):
    def __init__(self):
        super().__init__("semantic_projection")

        # Params (make it swappable tomorrow)
        self.declare_parameter("detections_topic", "/semnav/detections")
        self.declare_parameter("depth_topic", "/oakd/stereo/image_raw")     
        self.declare_parameter("camera_info_topic", "/oakd/rgb/camera_info")
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("depth_window", 3)       # median window size (odd int)
        self.declare_parameter("max_depth_m", 6.0)      # reject far points
        self.declare_parameter("min_depth_m", 0.2)      # reject near/invalid
        self.declare_parameter("publish_topic", "/semnav/objects_raw")

        det_topic = self.get_parameter("detections_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cam_info_topic = self.get_parameter("camera_info_topic").value
        self.map_frame = self.get_parameter("map_frame").value

        self.depth_window = int(self.get_parameter("depth_window").value)
        if self.depth_window < 1 or self.depth_window % 2 == 0:
            self.depth_window = 3

        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        pub_topic = self.get_parameter("publish_topic").value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Cached inputs
        self._last_depth: Optional[Image] = None
        self._last_info: Optional[CameraInfo] = None

        # IO
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, self.cam_info_topic, self.on_info, 10)
        self.create_subscription(Detection2DArray, det_topic, self.on_detections, 10)

        self.pub = self.create_publisher(SemanticObject, pub_topic, 10)

        self.get_logger().info(
            f"SemanticProjection started. det={det_topic}, depth={self.depth_topic}, info={self.cam_info_topic}, map={self.map_frame}"
        )

    def on_depth(self, msg: Image):
        self._last_depth = msg

    def on_info(self, msg: CameraInfo):
        self._last_info = msg

    def on_detections(self, dets: Detection2DArray):
        if self._last_depth is None or self._last_info is None:
            return

        # Camera intrinsics
        K = self._last_info.k  # row-major 3x3
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        if fx == 0.0 or fy == 0.0:
            return

        depth_msg = self._last_depth

        for det in dets.detections:
            if not det.results:
                continue

            # Pixel center of bbox
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

            z = self.depth_at_pixel_m(depth_msg, u, v, self.depth_window)
            if z is None or not (self.min_depth_m <= z <= self.max_depth_m):
                continue

            # Back-project to camera frame (optical frame coords)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Transform camera->map at (best) detection stamp time
            cam_frame = dets.header.frame_id or depth_msg.header.frame_id
            if not cam_frame:
                continue

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    cam_frame,
                    rclpy.time.Time(),  # align to detection time
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue

            point = PointStamped()
            point.header.frame_id = cam_frame
            point.headeer.stamp = dets.header.stamp

            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                point_map = self.tf_buffer.transform(
                    point,
                    self.map_frame,
                    timeout=rclpy.duration.DUration(seconds=0.2)
                )
            except Exception:
                continue
            
            mx = point+map.point.x
            my = point+map.point.y



            # Choose top hypothesis
            hyp0 = det.results[0].hypothesis
            obj = SemanticObject()
            obj.class_id = str(hyp0.class_id)
            obj.x = float(mx)
            obj.y = float(my)
            obj.confidence = float(hyp0.score)
            obj.stamp = rclypy.time.Time()

            self.pub.publish(obj)

    def depth_at_pixel_m(self, msg: Image, u: float, v: float, win: int) -> Optional[float]:
        # Bounds
        uu = int(round(u))
        vv = int(round(v))
        if uu < 0 or vv < 0 or uu >= msg.width or vv >= msg.height:
            return None

        enc = (msg.encoding or "").lower()

        # Make window (median for stability)
        r = win // 2
        u0, u1 = max(0, uu - r), min(msg.width - 1, uu + r)
        v0, v1 = max(0, vv - r), min(msg.height - 1, vv + r)

        if "32fc1" in enc:
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            patch = arr[v0 : v1 + 1, u0 : u1 + 1].flatten()
            patch = patch[np.isfinite(patch)]
            if patch.size == 0:
                return None
            z = float(np.median(patch))  # already meters
            return z if z > 0 else None

        if "16uc1" in enc:
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            patch = arr[v0 : v1 + 1, u0 : u1 + 1].flatten()
            patch = patch[patch > 0]
            if patch.size == 0:
                return None
            z_mm = float(np.median(patch))
            return z_mm / 1000.0

        # Unsupported
        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
        return None

    

def main(args=None):
    rclpy.init(args=args)
    node = SemanticProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
import math
from typing import Optional

import numpy as np

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs  # CRITICAL: registers PointStamped with tf2's transform machinery

from semnav_msgs.msg import SemanticObject


class SemanticProjectionNode(Node):
    def __init__(self):
        super().__init__("semantic_projection")

        #Parameters 
        self.declare_parameter("detections_topic",  "/semnav/detections")
        self.declare_parameter("depth_topic",        "/oakd/stereo/image_raw")
        self.declare_parameter("camera_info_topic",  "/oakd/stereo/camera_info")   # FIX: was rgb info — use stereo info to match depth pixels
        self.declare_parameter("map_frame",          "map")
        self.declare_parameter("depth_window",       3)
        self.declare_parameter("max_depth_m",        6.0)
        self.declare_parameter("min_depth_m",        0.2)
        self.declare_parameter("publish_topic",      "/semnav/objects_raw")
        self.declare_parameter("use_latest_tf",      True)   # FIX: bypass timestamp sync issues

        det_topic       = self.get_parameter("detections_topic").value
        self.depth_topic     = self.get_parameter("depth_topic").value
        self.cam_info_topic  = self.get_parameter("camera_info_topic").value
        self.map_frame       = self.get_parameter("map_frame").value
        pub_topic            = self.get_parameter("publish_topic").value
        self.use_latest_tf   = bool(self.get_parameter("use_latest_tf").value)

        self.depth_window = int(self.get_parameter("depth_window").value)
        if self.depth_window < 1 or self.depth_window % 2 == 0:
            self.depth_window = 3

        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)

        #TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #  Cached inputs 
        self._last_depth: Optional[Image]      = None
        self._last_info:  Optional[CameraInfo] = None

        #  QoS matched to OAK-D publisher profiles 
        # depth image = RELIABLE, camera_info = BEST_EFFORT
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscriptions 
        self.create_subscription(Image,           self.depth_topic,    self.on_depth,  depth_qos)
        self.create_subscription(CameraInfo,      self.cam_info_topic, self.on_info,   info_qos)
        self.create_subscription(Detection2DArray, det_topic,          self.on_detections, 10)

        # Publisher 
        self.pub = self.create_publisher(SemanticObject, pub_topic, 10)

        # Diagnostics counters 
        self._det_count      = 0
        self._published      = 0
        self._tf_failures    = 0
        self._depth_failures = 0

        # Periodic status timer (every 5 s)
        self.create_timer(5.0, self._log_status)

        self.get_logger().info(
            f"SemanticProjection ready.\n"
            f"  detections : {det_topic}\n"
            f"  depth      : {self.depth_topic}\n"
            f"  cam_info   : {self.cam_info_topic}\n"
            f"  map_frame  : {self.map_frame}\n"
            f"  use_latest_tf: {self.use_latest_tf}"
        )

    # Callbacks 

    def on_depth(self, msg: Image):
        self._last_depth = msg

    def on_info(self, msg: CameraInfo):
        self._last_info = msg

    def on_detections(self, dets: Detection2DArray):

        # wait for depth + info 
        if self._last_depth is None:
            self.get_logger().warn(
                f"No depth frame received yet on '{self.depth_topic}' — check topic name and QoS.",
                throttle_duration_sec=3.0,
            )
            return
        if self._last_info is None:
            self.get_logger().warn(
                f"No CameraInfo received yet on '{self.cam_info_topic}' — check topic name.",
                throttle_duration_sec=3.0,
            )
            return

        # Camera intrinsics 
        K  = self._last_info.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn("CameraInfo K matrix is all zeros — intrinsics not ready.", throttle_duration_sec=3.0)
            return

        depth_msg  = self._last_depth
        cam_frame  = dets.header.frame_id or depth_msg.header.frame_id

        # Diagnostic: log every batch 
        n = len(dets.detections)
        self._det_count += n
        self.get_logger().info(
            f"Received {n} detection(s) | cam_frame='{cam_frame}' | "
            f"depth_enc='{depth_msg.encoding}' | "
            f"depth_frame='{depth_msg.header.frame_id}'",
            throttle_duration_sec=2.0,
        )

        if not cam_frame:
            self.get_logger().warn("Detection header.frame_id is empty and depth frame_id is also empty — cannot transform.")
            return

        # Check TF reachability once per batch
        try:
            self.tf_buffer.lookup_transform(
                self.map_frame,
                cam_frame,
                rclpy.time.Time(),   # latest available
                timeout=rclpy.duration.Duration(seconds=0.3),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF not available: map ← '{cam_frame}': {e}",
                throttle_duration_sec=2.0,
            )
            self._tf_failures += 1
            return

        # Per-detection projection 
        for det in dets.detections:
            if not det.results:
                continue

            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

            z = self.depth_at_pixel_m(depth_msg, u, v, self.depth_window)

            if z is None:
                self._depth_failures += 1
                self.get_logger().debug(f"depth_at_pixel returned None for u={u:.1f} v={v:.1f}")
                continue

            if not (self.min_depth_m <= z <= self.max_depth_m):
                self.get_logger().debug(
                    f"Depth {z:.3f}m out of range [{self.min_depth_m}, {self.max_depth_m}] "
                    f"for u={u:.1f} v={v:.1f}"
                )
                self._depth_failures += 1
                continue

            # Back-project to camera optical frame
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            z_cam = z

            point = PointStamped()
            point.header.frame_id = cam_frame

            if self.use_latest_tf:
                # Use time=0 (latest transform) — avoids stamp sync issues with SLAM
                point.header.stamp = rclpy.time.Time().to_msg()
            else:
                point.header.stamp = dets.header.stamp

            point.point.x = x_cam
            point.point.y = y_cam
            point.point.z = z_cam

            try:
                point_map = self.tf_buffer.transform(
                    point,
                    self.map_frame,
                    timeout=rclpy.duration.Duration(seconds=0.3),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(
                    f"TF transform failed: {e}",
                    throttle_duration_sec=2.0,
                )
                self._tf_failures += 1
                continue
            except Exception as e:
                self.get_logger().error(f"Unexpected TF error: {e}")
                continue

            mx = point_map.point.x
            my = point_map.point.y

            self.get_logger().debug(
                f"Projected ({u:.0f},{v:.0f}) z={z:.2f}m → map ({mx:.2f}, {my:.2f})"
            )

            hyp0 = det.results[0].hypothesis
            obj           = SemanticObject()
            obj.class_id  = str(hyp0.class_id)
            obj.x         = float(mx)
            obj.y         = float(my)
            obj.confidence = float(hyp0.score)
            obj.stamp     = dets.header.stamp

            self.pub.publish(obj)
            self._published += 1

    # Depth extraction 

    def depth_at_pixel_m(self, msg: Image, u: float, v: float, win: int) -> Optional[float]:
        uu = int(round(u))
        vv = int(round(v))
        if uu < 0 or vv < 0 or uu >= msg.width or vv >= msg.height:
            return None

        enc = (msg.encoding or "").lower()
        r   = win // 2
        u0, u1 = max(0, uu - r), min(msg.width  - 1, uu + r)
        v0, v1 = max(0, vv - r), min(msg.height - 1, vv + r)

        if "32fc1" in enc:
            arr   = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            patch = arr[v0:v1+1, u0:u1+1].flatten()
            patch = patch[np.isfinite(patch) & (patch > 0)]
            if patch.size == 0:
                return None
            return float(np.median(patch))

        if "16uc1" in enc:
            arr   = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            patch = arr[v0:v1+1, u0:u1+1].flatten()
            patch = patch[patch > 0]
            if patch.size == 0:
                return None
            return float(np.median(patch)) / 1000.0

        self.get_logger().warn(
            f"Unsupported depth encoding: '{msg.encoding}' — expected 32FC1 or 16UC1",
            throttle_duration_sec=5.0,
        )
        return None

    #Periodic status 

    def _log_status(self):
        self.get_logger().info(
            f"[STATUS] detections_rx={self._det_count} | "
            f"published={self._published} | "
            f"tf_failures={self._tf_failures} | "
            f"depth_failures={self._depth_failures} | "
            f"depth_ready={self._last_depth is not None} | "
            f"info_ready={self._last_info is not None}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = SemanticProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
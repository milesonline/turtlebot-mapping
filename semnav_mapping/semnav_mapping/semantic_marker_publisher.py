import math
import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point, Vector3
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

from semnav_msgs.msg import SemanticObject


_PALETTE = [
    (1.0, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.4, 1.0),
    (1.0, 0.8, 0.0),
    (0.8, 0.2, 1.0),
    (0.0, 0.9, 0.9),
    (1.0, 0.5, 0.0),
    (0.0, 0.8, 0.5),
]

def class_color(class_id: str) -> ColorRGBA:
    r, g, b = _PALETTE[hash(class_id) % len(_PALETTE)]
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, 0.85
    return c


class SemanticMarkerPublisher(Node):
    def __init__(self):
        super().__init__("semantic_marker_publisher")

        self.declare_parameter("objects_topic",  "/semnav/objects")
        self.declare_parameter("marker_topic",   "/semnav/markers")
        self.declare_parameter("map_frame",      "map")
        self.declare_parameter("marker_z",        0.3)
        self.declare_parameter("sphere_scale",    0.25)
        self.declare_parameter("nav_goal_topic",  "/semnav/navigate_to_class")
        self.declare_parameter("standoff_m",      0.6)

        objects_topic     = self.get_parameter("objects_topic").value
        marker_topic      = self.get_parameter("marker_topic").value
        self.map_frame    = self.get_parameter("map_frame").value
        self.marker_z     = float(self.get_parameter("marker_z").value)
        self.sphere_scale = float(self.get_parameter("sphere_scale").value)
        nav_goal_topic    = self.get_parameter("nav_goal_topic").value
        self.standoff_m   = float(self.get_parameter("standoff_m").value)

        self._objects: dict    = {}
        self._marker_ids: dict = {}
        self._next_id          = 0
        self._last_seen: dict  = {}
        self._stale_seconds    = 3.0

        # TF for robot position lookup
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(SemanticObject, objects_topic, self._on_object, 10)
        self.create_subscription(String, nav_goal_topic, self._on_nav_request, 10)

        self.marker_pub  = self.create_publisher(MarkerArray, marker_topic, 10)
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_timer(0.5, self._publish_markers)
        self.create_timer(1.0, self._cleanup_stale)

        self.get_logger().info(
            f"SemanticMarkerPublisher ready.\n"
            f"  objects  : {objects_topic}\n"
            f"  markers  : {marker_topic}\n"
            f"  nav cmd  : {nav_goal_topic}\n"
            f"  frame    : {self.map_frame}\n"
            f"  standoff : {self.standoff_m}m"
        )

    # ── Object callback 

    def _on_object(self, msg: SemanticObject):
        cls = msg.class_id
        now = self.get_clock().now().nanoseconds * 1e-9

        if cls not in self._objects:
            self._objects[cls] = []

        matched = False
        for i, (ox, oy, oconf) in enumerate(self._objects[cls]):
            if math.hypot(msg.x - ox, msg.y - oy) < 0.3:
                self._objects[cls][i] = (msg.x, msg.y, msg.confidence)
                matched = True
                break
        if not matched:
            self._objects[cls].append((msg.x, msg.y, msg.confidence))

        self._last_seen[cls] = now

    # ── Marker publisher 

    def _publish_markers(self):
        array = MarkerArray()
        now   = self.get_clock().now().to_msg()

        for cls, positions in self._objects.items():
            color = class_color(cls)
            for idx, (x, y, conf) in enumerate(positions):
                key = (cls, idx)
                if key not in self._marker_ids:
                    self._marker_ids[key] = self._next_id
                    self._next_id += 1
                mid = self._marker_ids[key]

                sphere = Marker()
                sphere.header.frame_id    = self.map_frame
                sphere.header.stamp       = now
                sphere.ns                 = "semnav_objects"
                sphere.id                 = mid
                sphere.type               = Marker.SPHERE
                sphere.action             = Marker.ADD
                sphere.pose.position.x    = float(x)
                sphere.pose.position.y    = float(y)
                sphere.pose.position.z    = self.marker_z
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = sphere.scale.y = sphere.scale.z = self.sphere_scale
                sphere.color              = color
                sphere.lifetime           = rclpy.duration.Duration(seconds=1.5).to_msg()
                array.markers.append(sphere)

                label = Marker()
                label.header.frame_id    = self.map_frame
                label.header.stamp       = now
                label.ns                 = "semnav_labels"
                label.id                 = mid + 10000
                label.type               = Marker.TEXT_VIEW_FACING
                label.action             = Marker.ADD
                label.pose.position.x    = float(x)
                label.pose.position.y    = float(y)
                label.pose.position.z    = self.marker_z + self.sphere_scale
                label.pose.orientation.w = 1.0
                label.scale.z            = 0.18
                label.color.r = label.color.g = label.color.b = label.color.a = 1.0
                label.text               = f"{cls}\n{conf:.2f}"
                label.lifetime           = rclpy.duration.Duration(seconds=1.5).to_msg()
                array.markers.append(label)

        self.marker_pub.publish(array)

    # ── Stale cleanup 

    def _cleanup_stale(self):
        now   = self.get_clock().now().nanoseconds * 1e-9
        stale = [cls for cls, t in self._last_seen.items()
                 if (now - t) > self._stale_seconds]
        for cls in stale:
            self._objects.pop(cls, None)
            self._last_seen.pop(cls, None)
            self.get_logger().info(f"Removed stale class: '{cls}'")

    # ── Robot position from TF 

    def _get_robot_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f"Could not get robot position from TF: {e}")
            return 0.0, 0.0

    # ── NAV2 goal sender 

    def _on_nav_request(self, msg: String):
        target_class = msg.data.strip()
        self.get_logger().info(f"Navigation request for class: '{target_class}'")

        if target_class not in self._objects or not self._objects[target_class]:
            self.get_logger().warn(
                f"No tracked objects of class '{target_class}' — make sure object "
                f"is visible in camera before sending nav goal."
            )
            return

        best          = max(self._objects[target_class], key=lambda t: t[2])
        tx, ty, tconf = best

        # Standoff: navigate to a point standoff_m short of the object
        # This keeps the NAV2 goal in free space, not on/inside the object
        rx, ry = self._get_robot_position()
        dx     = tx - rx
        dy     = ty - ry
        dist   = math.hypot(dx, dy)

        if dist > self.standoff_m:
            ratio = (dist - self.standoff_m) / dist
            gx    = rx + dx * ratio
            gy    = ry + dy * ratio
        else:
            gx, gy = tx, ty  # already within standoff, go exactly there

        # Orient robot to face toward the object at the goal
        yaw    = math.atan2(dy, dx)
        quat_z = math.sin(yaw / 2.0)
        quat_w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f"Object=({tx:.2f},{ty:.2f}) | "
            f"Robot=({rx:.2f},{ry:.2f}) | "
            f"Goal=({gx:.2f},{gy:.2f}) | "
            f"dist={dist:.2f}m conf={tconf:.2f}"
        )

        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("NAV2 navigate_to_pose action server not available!")
            return

        goal                         = NavigateToPose.Goal()
        goal.pose.header.frame_id    = self.map_frame
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(gx)
        goal.pose.pose.position.y    = float(gy)
        goal.pose.pose.position.z    = 0.0
        goal.pose.pose.orientation.z = quat_z
        goal.pose.pose.orientation.w = quat_w

        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback
        )
        future.add_done_callback(self._nav_goal_response)

    def _nav_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("NAV2 goal was REJECTED.")
            return
        self.get_logger().info("NAV2 goal ACCEPTED — robot is navigating.")
        handle.get_result_async().add_done_callback(self._nav_result)

    def _nav_result(self, future):
        code = future.result().result.error_code
        if code == 0:
            self.get_logger().info("NAV2 navigation SUCCEEDED.")
        else:
            self.get_logger().warn(f"NAV2 navigation finished with error_code={code}")

    def _nav_feedback(self, feedback):
        dist = feedback.feedback.distance_remaining
        self.get_logger().info(
            f"NAV2 feedback: {dist:.2f}m remaining",
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
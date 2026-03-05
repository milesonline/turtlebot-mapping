import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from semnav_msgs.msg import SemanticObject

def dist2(a_x, a_y, b_x, b_y):
    dx = a_x - b_x
    dy = a_y - b_y
    return dx * dx + dy * dy

class SemanticRegistryNode(Node):
    """
    Maintains a stable object-level registry.
    
    Sub:  /semnav/objects_raw  (SemanticObject)
    Pub:  /semnav/objects      (SemanticObject)  -= publishes current tracks periodically
    
    mode=image  -> x/y treated as pixels
    mode=map    -> x/y treated as meters
    """

    def __init__(self):
        super().__init__('semantic_registry')

        self.declare_parameter('mode', 'image')
        self.declare_parameter('merge_radius', 60.0)
        self.declare_parameter('stale_seconds', 2.0)
        self.declare_parameter('publish_hz', 2.0)
        self.declare_parameter('allowed_classes', []) #empty = allow all
        allowed = self.get_parameter('allowed_classes').value or []
        self.allowed_classes = set(allowed)

        self.mode = self.get_parameter('mode').value
        self.merge_radius = float(self.get_parameter('merge_radius').value)
        self.stale_seconds = float(self.get_parameter('stale_seconds').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.merge_radius2 = self.merge_radius * self.merge_radius

        # tracks: dict[class_id] -> list[track]
        # track = {'x', 'y', 'confidence', 'stamp', 'last_seen_ns'}
        self.tracks = {}

        self.sub = self.create_subscription(
            SemanticObject,
            '/semnav/objects_raw',
            self.cb,
            10
        )

        self.pub = self.create_publisher(SemanticObject, '/semnav/objects', 10)

        period = 1.0 / max(self.publish_hz, 0.1)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"SemanticRegistry started: mode={self.mode}, merge_radius={self.merge_radius}, "
            f"stale_seconds={self.stale_seconds}, publish_hz={self.publish_hz}"
        )

    def cb(self, msg: SemanticObject):
        now_ns = self.get_clock().now().nanoseconds
        cls = msg.class_id

        if self.allowed_classes and cls not in self.allowed_classes:
            return

        # Get list for class
        lst = self.tracks.setdefault(cls, [])

        # Find nearest track
        best_i = -1
        best_d2 = None
        for i, t in enumerate(lst):
            d2 = dist2(msg.x, msg.y, t['x'], t['y'])
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_d2 is not None and best_d2 <= self.merge_radius2:
            # Update existing track (simple EMA smoothing)
            t = lst[best_i]
            alpha = 0.3
            t['x'] = (1 - alpha) * t['x'] + alpha * msg.x
            t['y'] = (1 - alpha) * t['y'] + alpha * msg.y
            t['confidence'] = max(t['confidence'], msg.confidence)
            t['stamp'] = msg.stamp
            t['last_seen_ns'] = now_ns
        else:
            # New track
            lst.append({
                'x': float(msg.x),
                'y': float(msg.y),
                'confidence': float(msg.confidence),
                'stamp': msg.stamp,
                'last_seen_ns': now_ns,
            })

    def on_timer(self):
        now_ns = self.get_clock().now().nanoseconds
        stale_ns = int(self.stale_seconds * 1e9)

        # prune + publish
        total = 0
        for cls, lst in list(self.tracks.items()):
        # prune stale
            lst[:] = [t for t in lst if (now_ns - t['last_seen_ns']) <= stale_ns]
            if not lst:
                del self.tracks[cls]
                continue

            # publish each active track as SemanticObject
            for t in lst:
                out = SemanticObject()
                out.class_id = cls
                out.x = float(t['x'])
                out.y = float(t['y'])
                out.confidence = float(t['confidence'])
                out.stamp = t['stamp']
                self.pub.publish(out)
                total += 1

        # light log
        self.get_logger().info(f"Published {total} active objects")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticRegistryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from builtin_interfaces.msg import Time


class YoloStubNode(Node):
    """
    YOLO-style detection stub.

    - Subscribes to a camera image topic.
    - Publishes a fake Detection2DArray on /semnav/detections.
    This is just to verify the interface and message flow.
    """

    def __init__(self):
        super().__init__('yolo_stub_node')

        #Parameter: which image topic to listen to
        self.declare_parameter('image_topic', '/camera/image_raw')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # Subscribe to the camera image
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Publisher for YOLO-style detections
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/semnav/detections',
            10
        )

        self.get_logger().info(f"YOLO stub node listening on: {image_topic}")
        self.get_logger().info(f"Publishing fake detections on: /semnav/detections")

    def image_callback(self, msg: Image):
        """
        Called every time an image comes in.
        For now, we just publish a single fake 'chair' detection
        roughly in the centre of the image.
        """

        # Build a Detection2DArray message
        det_array = Detection2DArray()
        det_array.header = msg.header # keep same timestamp and frame_id

        # Create a single fake detection
        det = Detection2D()
        det.header = msg.header
        
        # Fake bounding box in the center of the image
        bbox = BoundingBox2D()
        bbox.center.position.x = msg.width / 2.0
        bbox.center.position.y = msg.height / 2.0
        bbox.size_x = msg.width / 4.0
        bbox.size_y = msg.height / 4.0
        det.bbox = bbox

        # Fake class hypothesis: "chair" with 0.9 confidence
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "chair"
        hyp.hypothesis.score = 0.9
        # pose field left default (no 3D yet)
        det.results.append(hyp)


        det_array.detections.append(det)

        # Publish and log very lightly
        self.detections_pub.publish(det_array)
        self.get_logger().debug("Published fake detection")


def main(args=None):
    rclpy.init(args=args)
    node = YoloStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

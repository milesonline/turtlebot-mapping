import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class YoloVizNode(Node):
    """ 
    Simple Visualiser:
    Subs to raw image and yolo detections
    overlays bounding boxes and labels using OpenCV
    publishes annotated image as /semnav/debug/image
    """


    def __init__(self):
        super().__init__("yolo_viz_node")

        #Parameters (can be overidden with --ros-args)
        self.declare_parameter("image_topic", "oakd/rgb/image_raw")
        self.declare_parameter("detections_topic", "/semnav/detections")
        self.declare_parameter("output_image_topic", '/semnav/debug/image')


        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        output_image_topic = (
            self.get_parameter("output_image_topic").get_parameter_value().string_value
        )

        # Latest detections
        self.latest_detections = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray, detections_topic, self.detections_callback, 10
        )

        # Publisher for annotated image 
        self.image_pub = self.create_publisher(
            Image, output_image_topic, 10
        )

        self.get_logger().info(
            f"YOLO visualizer listening to image: {image_topic}, "
            f"detections: {detections_topic}"
        )
        self.get_logger().info(
            f"Publishing annotated images on: {output_image_topic}"
        )

    
    def detections_callback(self, msg: Detection2DArray):
        # Just store the latest detections
        self.latest_detections = msg

    def image_callback(self, msg: Image):
        # Convert ROS Image -> OpenCV BGR image
        img = self.rosimg_to_numpy(msg)
        if img is None:
            return

        h, w, _ = img.shape

        # If we have no detections yet, just republish the original image
        if self.latest_detections is None or len(self.latest_detections.detections) == 0:
            annotated = img
        else:
            annotated = img.copy()
            # Draw each detection
            for det in self.latest_detections.detections:
                # Bounding box in image coordinates
                bbox = det.bbox
                cx = bbox.center.position.x
                cy = bbox.center.position.y
                bw = bbox.size_x
                bh = bbox.size_y

                # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
                x1 = int(cx - bw / 2.0)
                y1 = int(cy - bh / 2.0)
                x2 = int(cx + bw / 2.0)
                y2 = int(cy + bh / 2.0)

                # Clip to image bounds
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # Get label (class_id + score)
                label = ""
                if len(det.results) > 0:
                    hyp = det.results[0].hypothesis
                    label = f"{hyp.class_id} {hyp.score:.2f}"

                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background + text
                if label:
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    # Put the label above the top-left corner of the box
                    y_text = max(0, y1 - 5)
                    cv2.rectangle(
                        annotated,
                        (x1, y_text - text_height - baseline),
                        (x1 + text_width, y_text + baseline),
                        (0, 255, 0),
                        thickness=-1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # Convert back to ROS Image and publish
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = annotated.shape[0]
        out_msg.width = annotated.shape[1]
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = annotated.shape[1] * 3
        out_msg.data = annotated.tobytes()

        self.image_pub.publish(out_msg)

    def rosimg_to_numpy(self, msg: Image):
        """Convert ROS Image to OpenCV BGR numpy array."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "bgr8":
                img = data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == "rgb8":
                img = data.reshape((msg.height, msg.width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return None
            return img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = YoloVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

    

            
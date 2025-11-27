import os
import time
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class YoloOnnxNode(Node):
    """
    YOLO inference node

    Subs to RGB image topic.
    Runs yolo with onnxruntime
    publishes Detection2DArray on /semnav/detections
    """

    def __init__(self):
        super().__init__('yolo_onnx_node')

        #params
        default_model_path = os.path.expanduser(
            '~/ros_ws/src/semnav_perception/models/yolov8n.onnx'
        )
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('score_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.45)  

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.score_thresh = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.iou_thresh = self.get_parameter('iou_threshold').get_parameter_value().double_value

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file model from: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"Loading YOLOv8 ONNX model from: {model_path}")
        t0 = time.time()
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.get_logger().info(f"Model loaded in {time.time() - t0:2f} s")

        # Get input / output meta
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape # [1, 3, H, W]
        # Some models use dynamic dims (None), default to 640 if so
        try:
            self.input_h = int(input_shape[2]) if input_shape[2] not in (None, 'None') else 640
            self.input_w = int(input_shape[3]) if input_shape[3] not in (None, 'None') else 640
        except Exception:
            self.input_h = 640
            self.input_w = 640

        self.get_logger().info(f"YOLO input size: {self.input_w}x{self.input_h}")


        # Subscriber / publisher
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )   
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/semnav/detections',
            10
        )

        self.get_logger().info(f"YoloOnnxNode listening to: {image_topic}")
        self.get_logger().info("Publishing real detections on: /semnav/detections")

    def image_callback(self, msg: Image):
        # Convert ROS Image -> numpy RGB
        img = self.rosimg_to_numpy(msg)
        if img is None:
            return
        
        h, w, _ = img.shape

        # Preprocess: resize, normalize, CHW
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1)) # HWC -> CHW
        input_tensor = np.expand_dims(img_chw, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # Ultralytics YOLOv8 ONNX: (1, N, 84) = [x, y, w, h, obj_conf, 80 class scores]
        preds = outputs[0][0] # (N, 84)

        boxes_xywh = preds[:, 0:4]
        scores_obj = preds[:, 4]
        scores_cls = preds[:, 5:]

        class_ids = np.argmax(scores_cls, axis=1)
        class_scores = scores_cls[np.arange(scores_cls.shape[0]), class_ids]
        confidences = scores_obj * class_scores

        # Filter by confidence
        keep = confidences > self.score_thresh
        boxes_xywh = boxes_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        if boxes_xywh.size == 0:
            # No detections
            return
        
        #Convert model-space boxes back to original image scale 
        scale_x = w / float(self.input_w)
        scale_y = h / float(self.input_h)

        boxes_xywh[:, 0] *= scale_x
        boxes_xywh[:, 1] *= scale_y
        boxes_xywh[:, 2] *= scale_x
        boxes_xywh[:, 3] *= scale_y

        # NMS
        nms_indices = self.nms_xywh(boxes_xywh, confidences, self.iou_thresh)
        boxes_xywh = boxes_xywh[nms_indices]
        confidences = confidences[nms_indices]
        class_ids = class_ids[nms_indices]

        # Build Detection2DArray
        det_array = Detection2DArray()
        det_array.header = msg.header

        for box, score, cls_id in zip(boxes_xywh, confidences, class_ids):
            det = Detection2D()
            det.header = msg.header

            # YOLO boxes are [cx, cy, w, h]
            cx, cy, bw, bh = box.tolist()

            bbox = BoundingBox2D()
            bbox.center.position.x = cx
            bbox.center.position.y = cy
            bbox.center.theta = 0.0
            bbox.size_x = bw
            bbox.size_y = bh
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else str(cls_id)
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            det_array.detections.append(det)

        self.detections_pub.publish(det_array)
        # self.get_logger().info(f"Published {len(det_array.detections)} detections")

    def rosimg_to_numpy(self, msg: Image):
        """Convert ROS Image to OpenCV BGR numpy array."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                img = data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img = data.reshape((msg.height, msg.width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return None
            return img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return None

    def nms_xywh(self, boxes, scores, iou_threshold):
        """Non-maximum suppression for boxes in [cx, cy, w, h] format."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        # Convert to [x1, y1, x2, y2]
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_int = np.maximum(0.0, xx2 - xx1)
            h_int = np.maximum(0.0, yy2 - yy1)
            inter = w_int * h_int

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=int)


def main(args=None):
    rclpy.init(args=args)
    node = YoloOnnxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
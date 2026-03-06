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
    YOLO inference node.
    Subscribes to RGB image topic, runs YOLOv8 with onnxruntime,
    publishes Detection2DArray on /semnav/detections.
    """

    def __init__(self):
        super().__init__('yolo_onnx_node')

        default_model_path = os.path.expanduser(
            '~/ros_ws/src/semnav_perception/models/yolov8n.onnx'
        )
        self.declare_parameter('image_topic',      '/oakd/rgb/image_raw')
        self.declare_parameter('model_path',       default_model_path)
        self.declare_parameter('score_threshold',  0.40)
        self.declare_parameter('iou_threshold',    0.45)

        image_topic       = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path        = self.get_parameter('model_path').get_parameter_value().string_value
        self.score_thresh = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.iou_thresh   = self.get_parameter('iou_threshold').get_parameter_value().double_value

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"Loading YOLOv8 ONNX model from: {model_path}")
        t0 = time.time()
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.get_logger().info(f"Model loaded in {time.time() - t0:.3f} s")

        self.input_name = self.session.get_inputs()[0].name
        input_shape     = self.session.get_inputs()[0].shape
        try:
            self.input_h = int(input_shape[2]) if input_shape[2] not in (None, 'None') else 640
            self.input_w = int(input_shape[3]) if input_shape[3] not in (None, 'None') else 640
        except Exception:
            self.input_h = self.input_w = 640

        self.get_logger().info(
            f"YOLO input size: {self.input_w}x{self.input_h} | "
            f"score_threshold={self.score_thresh} | iou_threshold={self.iou_thresh}"
        )

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray, '/semnav/detections', 10
        )

        self.get_logger().info(f"YoloOnnxNode listening to: {image_topic}")
        self.get_logger().info("Publishing detections on: /semnav/detections")

    # ── Image callback 

    def image_callback(self, msg: Image):
        img = self.rosimg_to_numpy(msg)
        if img is None:
            return

        h, w = img.shape[:2]

        # ── Preprocess 
        img_resized  = cv2.resize(img, (self.input_w, self.input_h))
        img_rgb      = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm     = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0)

        # ── Inference 
        raw = self.session.run(None, {self.input_name: input_tensor})[0]

        if raw.ndim != 3:
            self.get_logger().error(f"Unexpected YOLO output rank: {raw.ndim}")
            return

        # (1, 84, 8400) → (8400, 84)
        if raw.shape[1] == 84:
            preds = np.transpose(raw[0], (1, 0))
        elif raw.shape[2] == 84:
            preds = raw[0]
        else:
            self.get_logger().error(f"Unexpected YOLO output shape: {raw.shape}")
            return

        # ── Decode 
        boxes_xyxy  = preds[:, 0:4]          # model-space xyxy
        scores_cls  = preds[:, 4:]           # (N, 80)
        class_ids   = np.argmax(scores_cls, axis=1)
        confidences = scores_cls[np.arange(scores_cls.shape[0]), class_ids]

        # ── Confidence filter 
        keep = confidences >= self.score_thresh
        if not np.any(keep):
            self.get_logger().warn(
                f"No boxes above threshold {self.score_thresh:.2f}. "
                f"Max conf={float(confidences.max()):.3f}",
                throttle_duration_sec=2.0,
            )
            self._publish_empty(msg)
            return

        boxes_xyxy  = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids   = class_ids[keep]

        # ── Convert xyxy → xywh (cx, cy, w, h) in model space 
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        w_box = x2 - x1
        h_box = y2 - y1
        cx    = x1 + 0.5 * w_box
        cy    = y1 + 0.5 * h_box
        boxes_xywh = np.stack([cx, cy, w_box, h_box], axis=1)

        # ── Scale to original image size 
        scale_x = w / float(self.input_w)
        scale_y = h / float(self.input_h)
        boxes_xywh[:, 0] *= scale_x   # cx
        boxes_xywh[:, 1] *= scale_y   # cy
        boxes_xywh[:, 2] *= scale_x   # w
        boxes_xywh[:, 3] *= scale_y   # h

        # ── NMS 
        nms_idx    = self.nms_xywh(boxes_xywh, confidences, self.iou_thresh)
        boxes_xywh = boxes_xywh[nms_idx]    # FIX: iterate xywh, not xyxy
        confidences = confidences[nms_idx]
        class_ids  = class_ids[nms_idx]

        # ── Build Detection2DArray 
        det_array        = Detection2DArray()
        det_array.header = msg.header

        for box, score, cls_id in zip(boxes_xywh, confidences, class_ids):
            cx, cy, bw, bh = box.tolist()   # correctly cx/cy now

            det        = Detection2D()
            det.header = msg.header

            bbox                     = BoundingBox2D()
            bbox.center.position.x   = cx
            bbox.center.position.y   = cy
            bbox.center.theta        = 0.0
            bbox.size_x              = bw
            bbox.size_y              = bh
            det.bbox                 = bbox

            hyp                      = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id  = (
                COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else str(cls_id)
            )
            hyp.hypothesis.score     = float(score)
            det.results.append(hyp)
            det_array.detections.append(det)

        self.detections_pub.publish(det_array)
        self.get_logger().info(
            f"Published {len(det_array.detections)} detections",
            throttle_duration_sec=1.0,
        )

    # ── Helpers 

    def _publish_empty(self, msg: Image):
        det_array        = Detection2DArray()
        det_array.header = msg.header
        self.detections_pub.publish(det_array)

    def rosimg_to_numpy(self, msg: Image):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                return data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img = data.reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(
                    f"Unsupported image encoding: {msg.encoding}",
                    throttle_duration_sec=5.0,
                )
                return None
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return None

    def nms_xywh(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=int)

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1   = np.maximum(x1[i], x1[order[1:]])
            yy1   = np.maximum(y1[i], y1[order[1:]])
            xx2   = np.minimum(x2[i], x2[order[1:]])
            yy2   = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]

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
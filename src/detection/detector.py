"""
LibraryGuard - Detection Module
Person detection, object detection, and pose estimation using YOLOv8/Ultralytics.
"""

import numpy as np
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO
import torch
import logging

from ..utils.constants import Box, Point

logger = logging.getLogger(__name__)

# COCO class IDs for target objects
COCO_CLASSES = {
    'person': 0,
    'bottle': 39,
    'cup': 41,
    'cell phone': 67,
    'book': 73,
    'laptop': 63,
    'wine glass': 40,
    'bowl': 45,
}

TARGET_OBJECT_CLASSES = [39, 41, 67, 73, 63, 40, 45]  # bottle, cup, phone, book, laptop, wine glass, bowl


@dataclass
class BoundingBox:
    """Normalized bounding box [x1, y1, x2, y2] in range [0, 1].

    Interface-compatible with :class:`src.utils.constants.Box` (``center``
    and ``area`` are properties) so detection outputs flow directly into
    the tracking and behavior modules.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.confidence)

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates"""
        return (
            int(self.x1 * img_width),
            int(self.y1 * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height),
        )

    def to_xywh(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel xywh format"""
        x1, y1, x2, y2 = self.to_xyxy(img_width, img_height)
        return (x1, y1, x2 - x1, y2 - y1)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_box(self) -> Box:
        """Convert to the dependency-free :class:`src.utils.constants.Box`."""
        return Box(self.x1, self.y1, self.x2, self.y2)

    def contains_point(self, point: Point) -> bool:
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def distance_to_point(self, point: Point) -> float:
        dx = max(self.x1 - point.x, 0.0, point.x - self.x2)
        dy = max(self.y1 - point.y, 0.0, point.y - self.y2)
        return (dx * dx + dy * dy) ** 0.5

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another box"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class Keypoint:
    """Single keypoint with x, y, confidence"""
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    """17 COCO keypoints for a person"""
    keypoints: List[Keypoint] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    
    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def get_keypoint(self, idx: int) -> Optional[Keypoint]:
        if 0 <= idx < len(self.keypoints):
            kp = self.keypoints[idx]
            if kp.confidence > 0.3:
                return kp
        return None

    def get_wrist_positions(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get left and right wrist positions (normalized)"""
        left = self.get_keypoint(self.LEFT_WRIST)
        right = self.get_keypoint(self.RIGHT_WRIST)
        left_pos = (left.x, left.y) if left else None
        right_pos = (right.x, right.y) if right else None
        return left_pos, right_pos

    def get_shoulder_positions(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        left = self.get_keypoint(self.LEFT_SHOULDER)
        right = self.get_keypoint(self.RIGHT_SHOULDER)
        left_pos = (left.x, left.y) if left else None
        right_pos = (right.x, right.y) if right else None
        return left_pos, right_pos

    def get_head_position(self) -> Optional[Tuple[float, float]]:
        """Get approximate head center from nose/eyes"""
        nose = self.get_keypoint(self.NOSE)
        left_eye = self.get_keypoint(self.LEFT_EYE)
        right_eye = self.get_keypoint(self.RIGHT_EYE)
        
        points = [p for p in [nose, left_eye, right_eye] if p is not None]
        if not points:
            return None
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        return (x, y)

    def get_neck_position(self) -> Optional[Tuple[float, float]]:
        """Get neck position from shoulders"""
        left_shoulder = self.get_keypoint(self.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(self.RIGHT_SHOULDER)
        if left_shoulder and right_shoulder:
            return ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        return None

    def is_head_down(self, angle_threshold: float = 45.0) -> bool:
        """Check if head is tilted down (sleeping indicator)"""
        head = self.get_head_position()
        neck = self.get_neck_position()
        if not head or not neck:
            return False
        
        # Vector from neck to head
        dx = head[0] - neck[0]
        dy = head[1] - neck[1]
        
        # Angle from vertical (negative y is up in image coords)
        angle = np.degrees(np.arctan2(dx, -dy))
        return abs(angle) > angle_threshold


@dataclass
class DetectionResult:
    """Complete detection result for a frame"""
    frame_number: int
    timestamp: float
    persons: List[Pose] = field(default_factory=list)
    objects: List[BoundingBox] = field(default_factory=list)
    fps: float = 0.0
    inference_time: float = 0.0
    preprocess_time: float = 0.0
    postprocess_time: float = 0.0


class BaseDetector:
    """Base class for YOLO detectors"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 100,
        classes: Optional[List[int]] = None,
        half_precision: bool = False,
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.classes = classes
        self.half_precision = half_precision
        
        self.model: Optional[YOLO] = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device

    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            if self.half_precision and self.device != 'cpu':
                try:
                    self.model.model.half()
                except Exception:  # pragma: no cover - depends on ultralytics version
                    logger.debug("FP16 mode unavailable for %s; using FP32.", self.model_path)
            logger.info(f"Loaded model {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise

    def predict(self, frame: np.ndarray, **kwargs) -> Any:
        """Run inference on a single frame"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
            **kwargs
        )

    def predict_batch(self, frames: List[np.ndarray], **kwargs) -> List[Any]:
        """Run inference on a batch of frames"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return self.model.predict(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
            **kwargs
        )


class PersonDetector(BaseDetector):
    """Person detection using YOLOv8"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model_path', 'yolov8n.pt')
        kwargs.setdefault('classes', [0])  # person only
        super().__init__(**kwargs)

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect persons in frame"""
        results = self.predict(frame)
        return self._parse_results(results[0])

    def _parse_results(self, result) -> List[BoundingBox]:
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxyn[0].cpu().numpy()  # normalized
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                boxes.append(BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name='person',
                ))
        return boxes


class ObjectDetector(BaseDetector):
    """Object detection for target classes (bottle, cup, phone, etc.)"""
    
    def __init__(self, target_classes: Optional[List[int]] = None, **kwargs):
        kwargs.setdefault('model_path', 'yolov8n.pt')
        kwargs.setdefault('classes', target_classes or TARGET_OBJECT_CLASSES)
        super().__init__(**kwargs)
        self.target_classes = target_classes or TARGET_OBJECT_CLASSES

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect target objects in frame"""
        results = self.predict(frame)
        return self._parse_results(results[0])

    def _parse_results(self, result) -> List[BoundingBox]:
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxyn[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                class_name = self._get_class_name(cls_id)
                boxes.append(BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                ))
        return boxes

    def _get_class_name(self, class_id: int) -> str:
        for name, cid in COCO_CLASSES.items():
            if cid == class_id:
                return name
        return f'class_{class_id}'


class PoseDetector(BaseDetector):
    """Pose estimation using YOLOv8-pose"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model_path', 'yolov8n-pose.pt')
        kwargs.setdefault('classes', [0])  # person only
        super().__init__(**kwargs)

    def detect(self, frame: np.ndarray) -> List[Pose]:
        """Detect persons with pose keypoints"""
        results = self.predict(frame)
        return self._parse_results(results[0])

    def _parse_results(self, result) -> List[Pose]:
        poses = []
        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxyn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            keypoints_data = result.keypoints.xyn.cpu().numpy()  # normalized
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                bbox = BoundingBox(
                    x1=float(boxes[i, 0]),
                    y1=float(boxes[i, 1]),
                    x2=float(boxes[i, 2]),
                    y2=float(boxes[i, 3]),
                    confidence=float(confs[i]),
                    class_id=0,
                    class_name='person',
                )
                
                # Parse keypoints
                kpts = []
                for j in range(keypoints_data.shape[1]):
                    kpts.append(Keypoint(
                        x=float(keypoints_data[i, j, 0]),
                        y=float(keypoints_data[i, j, 1]),
                        confidence=float(keypoints_conf[i, j]),
                    ))
                
                pose = Pose(keypoints=kpts, bbox=bbox)
                poses.append(pose)
        
        return poses


def create_detector(
    detector_type: str,
    model_path: Optional[str] = None,
    **kwargs
) -> BaseDetector:
    """Factory function to create detectors"""
    detectors = {
        'person': PersonDetector,
        'object': ObjectDetector,
        'pose': PoseDetector,
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}. Choose from {list(detectors.keys())}")
    
    if model_path:
        kwargs['model_path'] = model_path
    
    return detectors[detector_type](**kwargs)


class MultiDetector:
    """Combined detector running person, object, and pose detection"""
    
    def __init__(
        self,
        person_model: str = 'yolov8n.pt',
        object_model: str = 'yolov8n.pt',
        pose_model: str = 'yolov8n-pose.pt',
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        half_precision: bool = False,
    ):
        self.person_detector = PersonDetector(
            model_path=person_model,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half_precision=half_precision,
        )
        self.object_detector = ObjectDetector(
            model_path=object_model,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half_precision=half_precision,
        )
        self.pose_detector = PoseDetector(
            model_path=pose_model,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half_precision=half_precision,
        )
        
        # Use pose detector for person detection too (it has both boxes and keypoints)
        self.use_pose_for_persons = True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run all detections on a frame"""
        import time

        start_time = time.time()

        # Run pose detection (includes person boxes + keypoints)
        persons = self.pose_detector.detect(frame)
        pose_time = time.time()

        # Run object detection
        objects = self.object_detector.detect(frame)
        object_time = time.time()

        # If not using pose for persons, run person detector
        if not self.use_pose_for_persons:
            persons = self.person_detector.detect(frame)

        total_time = time.time()

        return DetectionResult(
            frame_number=0,  # Set by caller
            timestamp=0.0,   # Set by caller
            persons=persons,
            objects=objects,
            inference_time=total_time - start_time,
            preprocess_time=pose_time - start_time,   # pose stage duration
            postprocess_time=object_time - pose_time,  # object stage duration
        )

    def warmup(self, frame_shape: Tuple[int, int] = (640, 640)):
        """Warm up models with dummy input"""
        dummy = np.zeros((frame_shape[1], frame_shape[0], 3), dtype=np.uint8)
        _ = self.pose_detector.detect(dummy)
        _ = self.object_detector.detect(dummy)
        logger.info("Models warmed up")


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Test model loading
    detector = MultiDetector()
    detector.warmup()
    print("All detectors loaded successfully!")
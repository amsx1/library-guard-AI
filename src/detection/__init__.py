"""
LibraryGuard - Detection Module
Person detection, object detection, and pose estimation using YOLOv8.
"""

from .detector import (
    PersonDetector,
    ObjectDetector,
    PoseDetector,
    DetectionResult,
    create_detector,
)

__all__ = [
    "PersonDetector",
    "ObjectDetector", 
    "PoseDetector",
    "DetectionResult",
    "create_detector",
]
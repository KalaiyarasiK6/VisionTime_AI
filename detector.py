"""
detector.py
-----------
YOLOv8-based Person Detector for VisionTime AI.

Responsible ONLY for:
- Loading the YOLOv8 model
- Selecting GPU (CUDA) if available, else CPU
- Reporting model / device status to the dashboard

Detection is restricted to the COCO "person" class so the system works
identically in any CCTV environment (airport, hospital, office, warehouse,
mall, factory, parking area, etc.) without any scene-specific logic.
"""

import torch
from ultralytics import YOLO

# COCO class id for "person"
PERSON_CLASS_ID = 0


class PersonDetector:
    """
    Wraps a YOLOv8 model configured to detect ONLY the 'person' class.
    Automatically uses GPU (CUDA) if available, otherwise falls back to CPU.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"  # FP16 inference on GPU for speed

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else None

    def status(self) -> dict:
        """Returns a dict describing current model/device status for the UI."""
        return {
            "device": self.device.upper(),
            "gpu_available": self.device == "cuda",
            "gpu_name": self.gpu_name,
            "model_path": self.model_path,
            "half_precision": self.half,
            "conf_threshold": self.conf_threshold,
        }
    def detect(self, frame):
        """
        Detect only persons in a video frame.

        Returns:
            results: YOLO Results object
        """

        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],   # Detect only persons
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        return results[0]

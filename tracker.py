"""
tracker.py
----------
ByteTrack-based multi-person tracker for VisionTime AI.

Uses Ultralytics' built-in ByteTrack integration (model.track) rather than a
hand-rolled tracker. This is the most stable, well-tested way to get
consistent person IDs frame-to-frame, and it directly improves ID-switch
problems compared to a naive IoU tracker.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from detector import PersonDetector, PERSON_CLASS_ID


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple        # (x1, y1, x2, y2) in original frame coordinates
    confidence: float
    centroid: tuple     # (cx, cy)


class PersonTracker:
    """
    Wraps YOLOv8 + ByteTrack (via ultralytics' .track API) to produce
    stable person IDs across frames.
    """

    def __init__(self, detector: PersonDetector, tracker_cfg: str = "bytetrack.yaml"):
        self.detector = detector
        self.tracker_cfg = tracker_cfg

    def reset(self):
        """Reset track memory (call once per new video)."""
        # Ultralytics keeps persistence internally keyed to the model object.
        # Re-creating .track with persist=False once clears state.
        try:
            self.detector.model.predictor = None
        except Exception:
            pass

    def update(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Run detection + tracking on a single frame.
        Returns a list of TrackedPerson with stable IDs.
        """
        results = self.detector.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            classes=[PERSON_CLASS_ID],
            conf=self.detector.conf_threshold,
            device=self.detector.device,
            half=self.detector.half,
            verbose=False,
        )

        people: List[TrackedPerson] = []
        if not results:
            return people

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return people

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for tid, box, conf in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            people.append(
                TrackedPerson(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    centroid=(float(cx), float(cy)),
                )
            )
        return people

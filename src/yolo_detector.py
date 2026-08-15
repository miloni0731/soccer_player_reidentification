from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Any

class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.3, device: str = 'cpu'):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        # Run YOLOv11 inference
        results = self.model(frame, conf=self.conf_threshold, device=self.device)[0]
        detections = []
        for box in results.boxes:
            # Only keep 'player' class (assume class 0 or use box.cls)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
            if hasattr(results, 'names'):
                label = results.names[cls]
            else:
                label = 'player'
            if label != 'player':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'confidence': conf,
                'class': label
            })
        return detections 
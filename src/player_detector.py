import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from src.utils import Config, apply_morphology, filter_contours, extract_features

class PlayerDetector:
    """
    Player detection using background subtraction and blob analysis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.bg_subtractor = None
        self.initialize_background_subtractor()
        
    def initialize_background_subtractor(self) -> None:
        """Initialize background subtractor"""
        history = self.config.get('detection.history_frames', 500)
        var_threshold = self.config.get('detection.var_threshold', 16)
        detect_shadows = self.config.get('detection.detect_shadows', True)
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        # Apply Gaussian blur to reduce noise
        kernel_size = self.config.get('detection.gaussian_blur_kernel', 5)
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return blurred
    
    def detect_foreground(self, frame: np.ndarray) -> np.ndarray:
        """Detect foreground objects using background subtraction"""
        preprocessed = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(preprocessed)
        
        # Apply morphological operations to clean up the mask
        kernel_size = self.config.get('detection.morphology_kernel', 3)
        cleaned_mask = apply_morphology(fg_mask, kernel_size)
        
        return cleaned_mask
    
    def find_player_contours(self, fg_mask: np.ndarray) -> List[np.ndarray]:
        """Find contours that likely represent players"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        min_area = self.config.get('detection.min_blob_area', 500)
        max_area = self.config.get('detection.max_blob_area', 5000)
        
        filtered_contours = filter_contours(contours, min_area, max_area)
        
        return filtered_contours
    
    def contour_to_bbox(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert contour to bounding box"""
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, w, h)
    
    def filter_overlapping_detections(self, bboxes: List[Tuple[int, int, int, int]], 
                                    threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping detections using Non-Maximum Suppression"""
        if not bboxes:
            return []
        
        # Convert to format expected by NMS
        boxes = []
        scores = []
        
        for bbox in bboxes:
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            scores.append(w * h)  # Use area as confidence score
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [bboxes[i] for i in indices]
        
        return []
    
    def validate_detection(self, bbox: Tuple[int, int, int, int], 
                          frame_shape: Tuple[int, int]) -> bool:
        """Validate if detection is reasonable"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Check if bbox is within frame bounds
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False
        
        # Check aspect ratio (players are typically taller than wide)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2.0 or aspect_ratio < 0.2:  # Too wide or too narrow
            return False
        
        # Check if detection is too close to edges (likely noise)
        edge_margin = 10
        if (x < edge_margin or y < edge_margin or 
            x + w > frame_w - edge_margin or y + h > frame_h - edge_margin):
            return False
        
        return True
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Main detection method that returns player detections
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries containing bbox and features
        """
        # Get foreground mask
        fg_mask = self.detect_foreground(frame)
        
        # Find player contours
        contours = self.find_player_contours(fg_mask)
        
        # Convert contours to bounding boxes
        bboxes = [self.contour_to_bbox(contour) for contour in contours]
        
        # Filter overlapping detections
        bboxes = self.filter_overlapping_detections(bboxes)
        
        # Validate detections and extract features
        detections = []
        for bbox in bboxes:
            if self.validate_detection(bbox, frame.shape):
                # Extract features for this detection
                features = extract_features(frame, bbox)
                
                detection = {
                    'bbox': bbox,
                    'features': features,
                    'confidence': self.calculate_confidence(bbox, fg_mask)
                }
                
                detections.append(detection)
        
        return detections
    
    def calculate_confidence(self, bbox: Tuple[int, int, int, int], 
                           fg_mask: np.ndarray) -> float:
        """Calculate confidence score for detection"""
        x, y, w, h = bbox
        
        # Extract region from foreground mask
        roi = fg_mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # Calculate ratio of foreground pixels
        foreground_ratio = np.sum(roi > 0) / roi.size
        
        # Factor in size (medium-sized detections are more likely to be players)
        area = w * h
        size_score = 1.0 - abs(area - 2000) / 2000  # Optimal around 2000 pixels
        size_score = max(0.0, min(1.0, size_score))
        
        # Combine scores
        confidence = 0.7 * foreground_ratio + 0.3 * size_score
        
        return confidence
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize detections on frame"""
        vis_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            x, y, w, h = bbox
            color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            text = f"Det {i}: {confidence:.2f}"
            cv2.putText(vis_frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame
    
    def get_detection_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about current detections"""
        if not detections:
            return {'count': 0, 'avg_confidence': 0.0, 'avg_area': 0.0}
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['bbox'][2] * d['bbox'][3] for d in detections]
        
        return {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences)
        }
    
    def reset_background_model(self) -> None:
        """Reset background subtractor (useful for scene changes)"""
        self.initialize_background_subtractor()
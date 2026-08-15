import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class OpticalFlowTracker:
    """
    Optical flow tracker using Lucas-Kanade and Kalman filtering for player movement prediction.
    """
    def __init__(self, config=None):
        self.prev_gray = None
        self.prev_points = None
        self.track_ids = []
        self.kalman_filters = {}
        
        # Load parameters from config if available
        if config:
            lk_params = config.get('tracking.lk_params', {})
            self.lk_params = {
                'winSize': tuple(lk_params.get('winSize', (15, 15))),
                'maxLevel': lk_params.get('maxLevel', 2),
                'criteria': tuple(lk_params.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)))
            }
        else:
            self.lk_params = {
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }
        
        self.config = config

    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create and initialize a new Kalman filter for tracking"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        return kalman

    def _get_or_create_kalman(self, track_id: str) -> cv2.KalmanFilter:
        """Get existing Kalman filter or create new one for track"""
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._create_kalman_filter()
        return self.kalman_filters[track_id]

    def _clean_old_filters(self, active_ids: List[str]) -> None:
        """Remove Kalman filters for tracks that are no longer active"""
        current_ids = set(active_ids)
        old_ids = set(self.kalman_filters.keys()) - current_ids
        for old_id in old_ids:
            del self.kalman_filters[old_id]

    def track(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Track player movements using optical flow and Kalman filtering
        
        Args:
            frame: Current video frame
            detections: List of player detections with features and track IDs
            
        Returns:
            List of updated detections with predicted positions
        """
        if not detections:
            self.prev_gray = None
            self.prev_points = None
            return []

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract points and track IDs
        curr_points = []
        track_ids = []
        for det in detections:
            if 'features' in det and 'centroid' in det['features'] and 'track_id' in det:
                curr_points.append(det['features']['centroid'])
                track_ids.append(det['track_id'])
        
        if not curr_points:
            self.prev_gray = None
            self.prev_points = None
            return detections

        curr_points = np.array(curr_points, dtype=np.float32)
        
        # Initialize tracking
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = gray
            self.prev_points = curr_points
            self.track_ids = track_ids
            return detections
            
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, curr_points, **self.lk_params
        )
        
        # Update detections with predictions
        updated_detections = []
        for i, (det, new_pt, st) in enumerate(zip(detections, new_points, status)):
            if st[0] == 1:  # Valid point
                # Get Kalman filter for this track
                track_id = det.get('track_id', str(i))
                kalman = self._get_or_create_kalman(track_id)
                
                # Correct Kalman filter with optical flow measurement
                measurement = np.array([[np.float32(new_pt[0])], 
                                      [np.float32(new_pt[1])]])
                kalman.correct(measurement)
                
                # Predict next position
                prediction = kalman.predict()
                predicted_pt = (float(prediction[0][0]), float(prediction[1][0]))
                
                # Update detection with prediction
                det['predicted_position'] = predicted_pt
                det['velocity'] = (float(prediction[2][0]), float(prediction[3][0]))
                det['optical_flow_status'] = 'tracked'
            else:
                # Mark as lost track
                det['optical_flow_status'] = 'lost'
                
            updated_detections.append(det)
        
        # Clean up old Kalman filters
        self._clean_old_filters(track_ids)
        
        # Update state for next frame
        self.prev_gray = gray
        self.prev_points = curr_points
        self.track_ids = track_ids
        
        return updated_detections

    def reset(self) -> None:
        """Reset tracker state"""
        self.prev_gray = None
        self.prev_points = None
        self.track_ids = []
        self.kalman_filters = {}
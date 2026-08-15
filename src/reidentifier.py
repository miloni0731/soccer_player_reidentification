import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.utils import calculate_iou, distance_between_points, calculate_histogram, compare_histograms

class PlayerReidentifier:
    """
    Player re-identification engine: assigns consistent IDs using color, position, and movement features.
    """
    def __init__(self, color_threshold=0.75, optical_flow_method='lucas_kanade', background_model='mog2', config=None):
        self.next_id = 1
        self.tracks = []  # List of dicts: {id, bbox, features, last_seen, ...}
        self.similarity_threshold = color_threshold
        self.max_frames_absent = 30
        self.min_track_length = 10
        
        if config is not None:
            self.similarity_threshold = config.get('reidentification.similarity_threshold', 0.75)
            self.max_frames_absent = config.get('reidentification.max_frames_absent', 30)
            self.min_track_length = config.get('reidentification.min_track_length', 10)
            
        self.config = config
        self.frame_count = 0

    def _predict_track_position(self, track: Dict[str, Any]) -> Tuple[float, float]:
        """Predict next position based on track history"""
        positions = track.get('positions', [])
        if len(positions) < 2:
            return track['features'].get('centroid', (0, 0))
            
        # Calculate velocity from last few positions
        num_pos = min(5, len(positions))
        recent_pos = positions[-num_pos:]
        if len(recent_pos) < 2:
            return recent_pos[-1]
            
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_pos)):
            dx = recent_pos[i][0] - recent_pos[i-1][0]
            dy = recent_pos[i][1] - recent_pos[i-1][1]
            velocities.append((dx, dy))
            
        avg_dx = np.mean([v[0] for v in velocities])
        avg_dy = np.mean([v[1] for v in velocities])
        
        # Predict next position
        last_pos = recent_pos[-1]
        predicted_x = last_pos[0] + avg_dx
        predicted_y = last_pos[1] + avg_dy
        
        return (predicted_x, predicted_y)

    def _calculate_movement_similarity(self, det: Dict[str, Any], track: Dict[str, Any]) -> float:
        """Calculate similarity based on movement patterns"""
        if 'velocity' not in det or 'velocity' not in track:
            return 0.0
            
        det_vel = det['velocity']
        track_vel = track['velocity']
        
        # Calculate angle between velocity vectors
        det_mag = np.sqrt(det_vel[0]**2 + det_vel[1]**2)
        track_mag = np.sqrt(track_vel[0]**2 + track_vel[1]**2)
        
        if det_mag == 0 or track_mag == 0:
            return 0.0
            
        dot_product = det_vel[0]*track_vel[0] + det_vel[1]*track_vel[1]
        cos_angle = dot_product / (det_mag * track_mag)
        
        # Convert to similarity score (1 for same direction, 0 for opposite)
        angle_sim = (cos_angle + 1) / 2
        
        # Also consider speed similarity
        speed_ratio = min(det_mag, track_mag) / max(det_mag, track_mag)
        
        return 0.7 * angle_sim + 0.3 * speed_ratio

    def _update_track_features(self, track: Dict[str, Any], detection: Dict[str, Any]) -> None:
        """Update track with new detection features"""
        track['bbox'] = detection['bbox']
        track['features'] = detection['features']
        track['last_seen'] = 0
        track['positions'].append(detection['features'].get('centroid', (0, 0)))
        
        # Keep only recent positions
        max_positions = 20
        if len(track['positions']) > max_positions:
            track['positions'] = track['positions'][-max_positions:]
            
        if 'velocity' in detection:
            track['velocity'] = detection['velocity']

    def _create_new_track(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new track from detection"""
        track_id = str(self.next_id)
        self.next_id += 1
        
        return {
            'id': track_id,
            'bbox': detection['bbox'],
            'features': detection['features'],
            'last_seen': 0,
            'positions': [detection['features'].get('centroid', (0, 0))],
            'velocity': detection.get('velocity', (0, 0)),
            'start_frame': self.frame_count,
            'confidence': detection.get('confidence', 1.0)
        }

    def calculate_similarity(self, det: Dict[str, Any], track: Dict[str, Any]) -> float:
        """Calculate multi-modal similarity between detection and track"""
        # Color similarity
        color_sim = 0.0
        if 'features' in det and 'features' in track:
            hist1 = det['features'].get('histogram')
            hist2 = track['features'].get('histogram')
            if hist1 is not None and hist2 is not None:
                color_sim = compare_histograms(hist1, hist2)

        # Position similarity
        pos_sim = 0.0
        if 'features' in det and 'features' in track:
            c1 = det['features'].get('centroid')
            c2 = track['features'].get('centroid')
            if c1 is not None and c2 is not None:
                dist = distance_between_points(c1, c2)
                pos_sim = 1.0 / (1.0 + dist / 50.0)

        # Movement similarity
        mov_sim = self._calculate_movement_similarity(det, track)

        # Get weights from config or use defaults
        color_weight = self.config.get('reidentification.color_weight', 0.4) if self.config else 0.4
        position_weight = self.config.get('reidentification.position_weight', 0.3) if self.config else 0.3
        movement_weight = self.config.get('reidentification.movement_weight', 0.3) if self.config else 0.3

        # Combine similarities
        total_sim = (color_weight * color_sim + 
                    position_weight * pos_sim + 
                    movement_weight * mov_sim)

        return total_sim

    def update_tracks(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracks with new detections"""
        self.frame_count += 1
        updated_tracks = []
        unmatched_detections = []

        # Predict track positions
        for track in self.tracks:
            track['predicted_position'] = self._predict_track_position(track)

        # Match detections to existing tracks
        detection_track_pairs = []
        for det in detections:
            best_track = None
            best_sim = self.similarity_threshold
            
            for track in self.tracks:
                if track.get('matched', False):
                    continue
                    
                sim = self.calculate_similarity(det, track)
                if sim > best_sim:
                    best_sim = sim
                    best_track = track

            if best_track is not None:
                detection_track_pairs.append((det, best_track))
                best_track['matched'] = True
            else:
                unmatched_detections.append(det)

        # Update matched tracks
        for det, track in detection_track_pairs:
            self._update_track_features(track, det)
            updated_tracks.append(track)

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_track = self._create_new_track(det)
            updated_tracks.append(new_track)

        # Update unmatched tracks
        for track in self.tracks:
            if not track.get('matched', False):
                track['last_seen'] += 1
                if track['last_seen'] < self.max_frames_absent:
                    # Only keep tracks that have been around for a while
                    if self.frame_count - track['start_frame'] >= self.min_track_length:
                        updated_tracks.append(track)

        # Clean up
        for track in updated_tracks:
            track.pop('matched', None)

        self.tracks = updated_tracks
        return self.tracks

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get list of currently active tracks"""
        return [t for t in self.tracks if t['last_seen'] == 0]

    def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track by ID"""
        for track in self.tracks:
            if str(track['id']) == str(track_id):
                return track
        return None

    def reset(self) -> None:
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
import unittest
import numpy as np
import cv2
from src.reidentifier import PlayerReidentifier
from src.utils import Config

class TestPlayerReidentifier(unittest.TestCase):
    def setUp(self):
        config = {
            'reidentification': {
                'similarity_threshold': 0.75,
                'max_frames_absent': 30,
                'color_weight': 0.4,
                'position_weight': 0.3,
                'movement_weight': 0.3,
                'min_track_length': 10
            }
        }
        self.reidentifier = PlayerReidentifier(config=config)

    def create_mock_detection(self, bbox, features=None, velocity=(0, 0), confidence=0.9):
        """Helper to create mock detection"""
        if features is None:
            # Create mock features
            features = {
                'histogram': np.random.rand(32),
                'centroid': (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2),
                'area': bbox[2] * bbox[3],
                'aspect_ratio': bbox[2] / bbox[3],
                'bbox': bbox
            }
        return {
            'bbox': bbox,
            'features': features,
            'velocity': velocity,
            'confidence': confidence
        }

    def test_new_track_creation(self):
        """Test that new tracks are created for new detections"""
        # Create a mock detection
        detection = self.create_mock_detection((100, 100, 50, 100))
        
        # Update with single detection
        tracks = self.reidentifier.update_tracks([detection], np.zeros((500, 500, 3)))
        
        self.assertEqual(len(tracks), 1)
        self.assertTrue('id' in tracks[0])
        self.assertEqual(tracks[0]['last_seen'], 0)
        self.assertEqual(len(tracks[0]['positions']), 1)

    def test_track_continuation(self):
        """Test that existing tracks are continued correctly"""
        # Initial detection
        det1 = self.create_mock_detection((100, 100, 50, 100))
        tracks = self.reidentifier.update_tracks([det1], np.zeros((500, 500, 3)))
        initial_track_id = tracks[0]['id']

        # Similar detection in next frame
        det2 = self.create_mock_detection((105, 102, 50, 100))  # Slightly moved
        tracks = self.reidentifier.update_tracks([det2], np.zeros((500, 500, 3)))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['id'], initial_track_id)
        self.assertEqual(len(tracks[0]['positions']), 2)

    def test_track_termination(self):
        """Test that tracks are terminated after being absent for too long"""
        # Initial detection
        det = self.create_mock_detection((100, 100, 50, 100))
        tracks = self.reidentifier.update_tracks([det], np.zeros((500, 500, 3)))
        
        # No detections for max_frames_absent + 1 frames
        for _ in range(31):  # max_frames_absent is 30
            tracks = self.reidentifier.update_tracks([], np.zeros((500, 500, 3)))
        
        self.assertEqual(len(tracks), 0)

    def test_multiple_tracks(self):
        """Test handling of multiple simultaneous tracks"""
        # Create two distant detections
        det1 = self.create_mock_detection((100, 100, 50, 100))
        det2 = self.create_mock_detection((300, 300, 50, 100))
        
        tracks = self.reidentifier.update_tracks([det1, det2], np.zeros((500, 500, 3)))
        
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0]['id'], tracks[1]['id'])

    def test_track_movement_similarity(self):
        """Test movement similarity calculation"""
        # Create detection with velocity
        det1 = self.create_mock_detection((100, 100, 50, 100), velocity=(5, 0))
        tracks = self.reidentifier.update_tracks([det1], np.zeros((500, 500, 3)))
        
        # Similar movement detection
        det2 = self.create_mock_detection((105, 100, 50, 100), velocity=(5, 0))
        sim = self.reidentifier._calculate_movement_similarity(det2, tracks[0])
        
        # Should have high similarity for same direction
        self.assertGreater(sim, 0.8)
        
        # Opposite direction
        det3 = self.create_mock_detection((95, 100, 50, 100), velocity=(-5, 0))
        sim = self.reidentifier._calculate_movement_similarity(det3, tracks[0])
        
        # Should have low similarity for opposite direction
        self.assertLess(sim, 0.3)

    def test_track_prediction(self):
        """Test track position prediction"""
        # Create sequence of detections with consistent movement
        positions = [(100, 100), (110, 100), (120, 100)]  # Moving right
        tracks = None
        
        for x, y in positions:
            det = self.create_mock_detection((x, y, 50, 100))
            tracks = self.reidentifier.update_tracks([det], np.zeros((500, 500, 3)))
        
        # Predict next position
        predicted = self.reidentifier._predict_track_position(tracks[0])
        
        # Should predict continued rightward movement
        self.assertGreater(predicted[0], positions[-1][0])
        self.assertAlmostEqual(predicted[1], positions[-1][1], delta=5)

    def test_reidentification_after_occlusion(self):
        """Test re-identification of player after temporary occlusion"""
        # Initial detection with specific features
        features = {
            'histogram': np.array([0.1] * 32),
            'centroid': (125, 150),
            'area': 5000,
            'aspect_ratio': 0.5,
            'bbox': (100, 100, 50, 100)
        }
        det1 = self.create_mock_detection((100, 100, 50, 100), features=features)
        tracks = self.reidentifier.update_tracks([det1], np.zeros((500, 500, 3)))
        initial_track_id = tracks[0]['id']
        
        # No detection for a few frames (occlusion)
        for _ in range(10):
            tracks = self.reidentifier.update_tracks([], np.zeros((500, 500, 3)))
        
        # Similar detection reappears
        det2 = self.create_mock_detection((120, 110, 50, 100), features=features)
        tracks = self.reidentifier.update_tracks([det2], np.zeros((500, 500, 3)))
        
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['id'], initial_track_id)

    def test_reset(self):
        """Test tracker reset functionality"""
        # Create some tracks
        det = self.create_mock_detection((100, 100, 50, 100))
        tracks = self.reidentifier.update_tracks([det], np.zeros((500, 500, 3)))
        
        # Reset tracker
        self.reidentifier.reset()
        
        # Check that state is cleared
        self.assertEqual(len(self.reidentifier.tracks), 0)
        self.assertEqual(self.reidentifier.next_id, 1)
        self.assertEqual(self.reidentifier.frame_count, 0)

if __name__ == '__main__':
    unittest.main()
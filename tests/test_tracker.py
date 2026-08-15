import unittest
import numpy as np
import cv2
from src.color_tracker import ColorTracker
from src.optical_flow import OpticalFlowTracker
from src.utils import Config

class TestColorTracker(unittest.TestCase):
    def setUp(self):
        config = {
            'tracking': {
                'color_bins': 32,
                'max_tracking_distance': 100
            }
        }
        self.tracker = ColorTracker(config)

    def test_create_color_profile(self):
        # Create a test image with a red rectangle
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 255), -1)  # Red rectangle
        bbox = (30, 30, 40, 40)  # x, y, w, h

        profile = self.tracker.create_color_profile(img, bbox)
        
        self.assertIsNotNone(profile)
        self.assertIn('histogram', profile)
        self.assertIn('dominant_colors', profile)
        self.assertIn('average_color', profile)
        self.assertIn('primary_color', profile)
        self.assertEqual(profile['bbox'], bbox)

    def test_empty_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (30, 30, 40, 40)

        profile = self.tracker.create_color_profile(img, bbox)
        
        self.assertEqual(profile, {})

    def test_invalid_bbox(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        invalid_bbox = (-10, -10, 20, 20)  # Outside image bounds

        profile = self.tracker.create_color_profile(img, invalid_bbox)
        
        self.assertEqual(profile, {})

    def test_extract_dominant_colors(self):
        # Create image with two distinct colors
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (50, 100), (255, 0, 0), -1)  # Blue half
        cv2.rectangle(img, (50, 0), (100, 100), (0, 255, 0), -1)  # Green half

        colors = self.tracker.extract_dominant_colors(img, k=2)
        
        self.assertEqual(len(colors), 2)
        self.assertTrue(any(c[0] > c[1] and c[0] > c[2] for c in colors))  # Blue present
        self.assertTrue(any(c[1] > c[0] and c[1] > c[2] for c in colors))  # Green present

    def test_identify_team_colors(self):
        # Create frame with players in two team colors
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Team 1 (Red)
        cv2.rectangle(frame, (30, 30), (70, 70), (0, 0, 255), -1)
        # Team 2 (Blue)
        cv2.rectangle(frame, (130, 30), (170, 70), (255, 0, 0), -1)

        detections = [
            {'bbox': (30, 30, 40, 40)},
            {'bbox': (130, 30, 40, 40)}
        ]

        team_colors = self.tracker.identify_team_colors(detections, frame)
        
        self.assertEqual(len(team_colors), 2)
        self.assertTrue(all(isinstance(color, tuple) for color in team_colors.values()))

class TestOpticalFlowTracker(unittest.TestCase):
    def setUp(self):
        config = {
            'tracking': {
                'optical_flow_threshold': 0.8,
                'prediction_window': 5,
                'lk_params': {
                    'winSize': [15, 15],
                    'maxLevel': 2,
                    'criteria': [3, 10, 0.03]
                }
            }
        }
        self.tracker = OpticalFlowTracker(config)

    def create_mock_detection(self, centroid, track_id="1"):
        return {
            'features': {'centroid': centroid},
            'track_id': track_id
        }

    def test_track_static_point(self):
        # Create two identical frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        
        # Add a point to track
        cv2.circle(frame1, (50, 50), 3, (255, 255, 255), -1)
        cv2.circle(frame2, (50, 50), 3, (255, 255, 255), -1)
        
        detection = self.create_mock_detection((50.0, 50.0))
        
        # First frame initializes tracking
        self.tracker.track([detection], frame1)
        # Second frame should track the static point
        updated = self.tracker.track([detection], frame2)
        
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]['optical_flow_status'], 'tracked')
        self.assertIn('predicted_position', updated[0])

    def test_track_moving_point(self):
        # Create frames with a moving point
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw point in different positions
        cv2.circle(frame1, (50, 50), 3, (255, 255, 255), -1)
        cv2.circle(frame2, (60, 50), 3, (255, 255, 255), -1)
        
        det1 = self.create_mock_detection((50.0, 50.0))
        det2 = self.create_mock_detection((60.0, 50.0))
        
        # Track moving point
        self.tracker.track([det1], frame1)
        updated = self.tracker.track([det2], frame2)
        
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]['optical_flow_status'], 'tracked')
        pred_x = updated[0]['predicted_position'][0]
        self.assertGreater(pred_x, 60.0)  # Should predict continued rightward motion

    def test_track_lost_point(self):
        # Test tracking when point disappears
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Only draw point in first frame
        cv2.circle(frame1, (50, 50), 3, (255, 255, 255), -1)
        
        det = self.create_mock_detection((50.0, 50.0))
        
        # Initialize tracking
        self.tracker.track([det], frame1)
        # Try to track with no detections
        updated = self.tracker.track([], frame2)
        
        self.assertEqual(len(updated), 0)

    def test_multiple_points(self):
        # Test tracking multiple points simultaneously
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw two points
        cv2.circle(frame1, (30, 30), 3, (255, 255, 255), -1)
        cv2.circle(frame1, (70, 70), 3, (255, 255, 255), -1)
        cv2.circle(frame2, (35, 30), 3, (255, 255, 255), -1)
        cv2.circle(frame2, (75, 70), 3, (255, 255, 255), -1)
        
        det1 = self.create_mock_detection((30.0, 30.0), "1")
        det2 = self.create_mock_detection((70.0, 70.0), "2")
        det3 = self.create_mock_detection((35.0, 30.0), "1")
        det4 = self.create_mock_detection((75.0, 70.0), "2")
        
        # Track points
        self.tracker.track([det1, det2], frame1)
        updated = self.tracker.track([det3, det4], frame2)
        
        self.assertEqual(len(updated), 2)
        self.assertTrue(all(d['optical_flow_status'] == 'tracked' for d in updated))

    def test_reset(self):
        # Test tracker reset functionality
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(frame, (50, 50), 3, (255, 255, 255), -1)
        
        det = self.create_mock_detection((50.0, 50.0))
        
        # Initialize tracking
        self.tracker.track([det], frame)
        
        # Reset tracker
        self.tracker.reset()
        
        # Check that state is cleared
        self.assertIsNone(self.tracker.prev_gray)
        self.assertIsNone(self.tracker.prev_points)
        self.assertEqual(len(self.tracker.track_ids), 0)
        self.assertEqual(len(self.tracker.kalman_filters), 0)

if __name__ == '__main__':
    unittest.main()
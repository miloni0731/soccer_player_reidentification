import unittest
import numpy as np
import cv2
from src.player_detector import PlayerDetector
from src.utils import Config

class TestPlayerDetector(unittest.TestCase):
    def setUp(self):
        config = {
            'detection': {
                'background_threshold': 50,
                'min_blob_area': 500,
                'max_blob_area': 5000,
                'gaussian_blur_kernel': 5,
                'morphology_kernel': 3,
                'history_frames': 500,
                'var_threshold': 16,
                'detect_shadows': True
            }
        }
        self.detector = PlayerDetector(config)

    def create_test_frame(self, shape=(200, 200, 3), player_positions=None):
        """Create a test frame with optional player positions"""
        frame = np.zeros(shape, dtype=np.uint8)
        if player_positions:
            for x, y, w, h in player_positions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
        return frame

    def test_preprocess_frame(self):
        """Test frame preprocessing"""
        frame = self.create_test_frame()
        # Add some noise
        noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
        
        processed = self.detector.preprocess_frame(noisy_frame)
        
        # Processed frame should have less noise (lower standard deviation)
        self.assertLess(np.std(processed), np.std(noisy_frame))

    def test_detect_foreground(self):
        """Test foreground detection"""
        # Create sequence of frames with moving object
        frame1 = self.create_test_frame()
        frame2 = self.create_test_frame(player_positions=[(50, 50, 20, 40)])
        
        # First frame initializes background model
        mask1 = self.detector.detect_foreground(frame1)
        # Second frame should detect the new object
        mask2 = self.detector.detect_foreground(frame2)
        
        # Check that something was detected in second frame
        self.assertGreater(np.sum(mask2), np.sum(mask1))

    def test_find_player_contours(self):
        """Test player contour detection"""
        # Create binary mask with player-sized objects
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Add valid player blob
        cv2.rectangle(mask, (50, 50), (70, 90), 255, -1)  # 20x40 pixels
        # Add too small blob
        cv2.rectangle(mask, (100, 100), (105, 105), 255, -1)  # 5x5 pixels
        
        contours = self.detector.find_player_contours(mask)
        
        # Should only detect the player-sized blob
        self.assertEqual(len(contours), 1)

    def test_filter_overlapping_detections(self):
        """Test overlapping detection filtering"""
        # Create overlapping bounding boxes
        bboxes = [
            (100, 100, 50, 100),  # Original box
            (110, 110, 50, 100),  # Overlapping box
            (200, 200, 50, 100)   # Non-overlapping box
        ]
        
        filtered = self.detector.filter_overlapping_detections(bboxes)
        
        # Should remove one of the overlapping boxes
        self.assertEqual(len(filtered), 2)

    def test_validate_detection(self):
        """Test detection validation"""
        frame_shape = (200, 200)
        
        # Test valid detection
        valid_bbox = (50, 50, 30, 60)  # Normal player size
        self.assertTrue(self.detector.validate_detection(valid_bbox, frame_shape))
        
        # Test invalid detections
        invalid_bboxes = [
            (-10, 50, 30, 60),  # Outside frame
            (50, -10, 30, 60),  # Outside frame
            (50, 50, 100, 10),  # Wrong aspect ratio (too wide)
            (50, 50, 10, 100),  # Wrong aspect ratio (too narrow)
            (0, 0, 30, 60),     # Too close to edge
        ]
        
        for bbox in invalid_bboxes:
            self.assertFalse(self.detector.validate_detection(bbox, frame_shape))

    def test_detect_players(self):
        """Test complete player detection pipeline"""
        # Create frame with two players
        player_positions = [
            (50, 50, 30, 60),   # Valid player
            (150, 50, 30, 60)   # Valid player
        ]
        frame = self.create_test_frame(player_positions=player_positions)
        
        # Run detection
        detections = self.detector.detect_players(frame)
        
        # Should detect both players
        self.assertEqual(len(detections), 2)
        
        # Check detection format
        for det in detections:
            self.assertIn('bbox', det)
            self.assertIn('features', det)
            self.assertIn('confidence', det)
            self.assertGreaterEqual(det['confidence'], 0.0)
            self.assertLessEqual(det['confidence'], 1.0)

    def test_calculate_confidence(self):
        """Test confidence calculation"""
        # Create foreground mask with different fill ratios
        mask = np.zeros((200, 200), dtype=np.uint8)
        
        # Full detection
        cv2.rectangle(mask, (50, 50), (80, 110), 255, -1)
        full_conf = self.detector.calculate_confidence((50, 50, 30, 60), mask)
        
        # Partial detection
        cv2.rectangle(mask, (150, 50), (165, 110), 255, -1)  # Half width
        partial_conf = self.detector.calculate_confidence((150, 50, 30, 60), mask)
        
        # Full detection should have higher confidence
        self.assertGreater(full_conf, partial_conf)

    def test_reset_background_model(self):
        """Test background model reset"""
        # Create sequence of frames
        frame1 = self.create_test_frame(player_positions=[(50, 50, 20, 40)])
        frame2 = self.create_test_frame()  # Empty frame
        
        # Initial detection
        mask1 = self.detector.detect_foreground(frame1)
        
        # Reset model
        self.detector.reset_background_model()
        
        # Detection after reset should be different
        mask2 = self.detector.detect_foreground(frame2)
        
        # Masks should be different
        self.assertFalse(np.array_equal(mask1, mask2))

if __name__ == '__main__':
    unittest.main()
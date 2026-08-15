import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class Config:
    """Configuration handler for the project"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "detection": {
                "background_threshold": 50,
                "min_blob_area": 500,
                "max_blob_area": 5000,
                "gaussian_blur_kernel": 5,
                "morphology_kernel": 3
            },
            "tracking": {
                "color_bins": 32,
                "optical_flow_threshold": 0.8,
                "prediction_window": 5,
                "max_tracking_distance": 100
            },
            "reidentification": {
                "similarity_threshold": 0.75,
                "max_frames_absent": 30,
                "color_weight": 0.4,
                "position_weight": 0.3,
                "movement_weight": 0.3
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

def calculate_histogram(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Calculate color histogram for an image region"""
    if len(image.shape) == 3:
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    else:
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
    
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compare two histograms using correlation"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union of two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0

def get_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get centroid of bounding box"""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def distance_between_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return euclidean(p1, p2)

def create_output_dirs(base_path: str) -> None:
    """Create necessary output directories"""
    dirs = ['frames', 'tracks', 'videos', 'reports']
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def draw_text(image: np.ndarray, text: str, position: Tuple[int, int], 
              color: Tuple[int, int, int] = (255, 255, 255), scale: float = 0.6) -> np.ndarray:
    """Draw text on image"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return image

def apply_morphology(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological operations to clean up binary image"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Remove noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Fill holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def filter_contours(contours: List, min_area: int = 500, max_area: int = 5000) -> List:
    """Filter contours based on area"""
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)
    return filtered

def extract_features(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Extract various features from image region"""
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {}
    
    # Create mask for better feature extraction
    mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
    
    features = {
        'histogram': calculate_histogram(roi, mask),
        'centroid': get_centroid(bbox),
        'area': w * h,
        'aspect_ratio': w / h if h > 0 else 0,
        'bbox': bbox
    }
    
    return features

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_video(video_path: str) -> cv2.VideoCapture:
    """Load video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap

def get_video_properties(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """Get video properties"""
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create video writer"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

class Logger:
    """Simple logging utility"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def info(self, message: str) -> None:
        if self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message: str) -> None:
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        if self.verbose:
            print(f"[ERROR] {message}")

def visualize_tracking_results(tracks: List[Dict], frame_shape: Tuple[int, int]) -> None:
    """Visualize tracking results using matplotlib"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tracks)))
    
    for i, track in enumerate(tracks):
        positions = track.get('positions', [])
        if len(positions) > 1:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            plt.plot(x_coords, y_coords, color=colors[i], 
                    label=f"Player {track.get('id', i)}", linewidth=2)
    
    plt.xlim(0, frame_shape[1])
    plt.ylim(frame_shape[0], 0)  # Invert y-axis for image coordinates
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Player Tracking Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
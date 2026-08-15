import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from src.utils import Config, calculate_histogram, compare_histograms

class ColorTracker:
    """
    Color-based player tracking using histogram analysis and team identification
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.team_colors = {}
        self.player_profiles = {}
        self.color_bins = config.get('tracking.color_bins', 32)
        
    def extract_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove very dark pixels (shadows, etc.)
        pixels = pixels[np.sum(pixels, axis=1) > 50]
        
        if len(pixels) < k:
            return [(0, 0, 0)] * k
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to list of tuples
        return [tuple(color) for color in colors]
    
    def create_color_profile(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Create comprehensive color profile for a player"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return {}
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask to focus on jersey area (middle portion)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 255
        
        # Calculate color histogram
        hist = calculate_histogram(hsv_roi, mask)
        
        # Extract dominant colors
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        dominant_colors = self.extract_dominant_colors(masked_roi)
        
        # Calculate average color
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        
        profile = {
            'histogram': hist,
            'dominant_colors': dominant_colors,
            'average_color': avg_color,
            'primary_color': dominant_colors[0] if dominant_colors else (0, 0, 0),
            'bbox': bbox
        }
        
        return profile
    
    def identify_team_colors(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """Identify team colors from initial detections"""
        if not detections:
            return {}
        
        # Extract all dominant colors from detections
        all_colors = []
        for detection in detections:
            bbox = detection['bbox']
            profile = self.create_color_profile(frame, bbox)
            if profile and 'dominant_colors' in profile:
                all_colors.extend(profile['dominant_colors'])
        
        if not all_colors:
            return {}
        
        # Cluster colors to identify teams
        color_array = np.array(all_colors)
        
        # Use K-means to find 2-3 main team colors
        n_teams = min(3, len(all_colors))
        kmeans = KMeans(n_clusters=n_teams, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(color_array)
        
        # Create team color dictionary
        team_colors = {}
        for i in range(n_teams):
            team_color = kmeans.cluster_centers_[i].astype(int)
            team_colors[f'team_{i}'] = tuple(team_color)
        
        return team_colors
    
    def assign_team(self, color_profile, team_refs=None):
        """Assign team based on dominant color profile. Optionally use reference colors for teams."""
        if color_profile is None or len(color_profile) == 0:
            return 'unknown'
        b, g, r = color_profile[0]
        color = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
        h = hsv[0]
        # Manchester United (red): hue < 10 or hue > 160
        # Manchester City (light blue): 90 < hue < 130
        if h < 10 or h > 160:
            return 'team1'  # red
        elif 90 < h < 130:
            return 'team2'  # blue
        else:
            return 'unknown'
    
    def assign_player_to_team(self, color_profile: Dict[str, Any]) -> str:
        """Assign player to team based on color profile"""
        if not self.team_colors or not color_profile:
            return 'unknown'
        
        primary_color = color_profile.get('primary_color', (0, 0, 0))
        
        # Find closest team color
        min_distance = float('inf')
        assigned_team = 'unknown'
        
        for team_name, team_color in self.team_colors.items():
            # Calculate color distance in RGB space
            distance = np.sqrt(np.sum((np.array(primary_color) - np.array(team_color))**2))
            
            if distance < min_distance:
                min_distance = distance
                assigned_team = team_name
        
        return assigned_team
    
    def calculate_color_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """Calculate similarity between two color profiles"""
        if not profile1 or not profile2:
            return 0.0
        
        # Compare histograms
        hist_similarity = compare_histograms(profile1['histogram'], profile2['histogram'])
        
        # Compare dominant colors
        colors1 = profile1['dominant_colors']
        colors2 = profile2['dominant_colors']
        
        color_similarity = 0.0
        if colors1 and colors2:
            # Find best matching colors
            max_color_sim = 0.0
            for c1 in colors1:
                for c2 in colors2:
                    # Calculate color distance
                    distance = np.sqrt(np.sum((np.array(c1) - np.array(c2))**2))
                    similarity = 1.0 / (1.0 + distance / 100.0)  # Normalize
                    max_color_sim = max(max_color_sim, similarity)
            color_similarity = max_color_sim
        
        # Combine similarities
        total_similarity = 0.7 * hist_similarity + 0.3 * color_similarity
        
        return max(0.0, min(1.0, total_similarity))
    
    def update_player_profile(self, player_id: str, new_profile: Dict[str, Any], 
                            learning_rate: float = 0.1) -> None:
        """Update player color profile with new observations"""
        if player_id not in self.player_profiles:
            self.player_profiles[player_id] = new_profile
            return
        
        old_profile = self.player_profiles[player_id]
        
        # Update histogram using weighted average
        old_hist = old_profile['histogram']
        new_hist = new_profile['histogram']
        
        updated_hist = (1 - learning_rate) * old_hist + learning_rate * new_hist
        
        # Update other features
        updated_profile = {
            'histogram': updated_hist,
            'dominant_colors': new_profile['dominant_colors'],  # Use latest
            'average_color': ((1 - learning_rate) * old_profile['average_color'] + 
                             learning_rate * new_profile['average_color']),
            'primary_color': new_profile['primary_color'],
            'bbox': new_profile['bbox']
        }
        
        self.player_profiles[player_id] = updated_profile
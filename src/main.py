import argparse
import os
import cv2
import logging
import time
from src.utils import Config, Logger, create_output_dirs, load_video, get_video_properties, create_video_writer, save_results, extract_features, distance_between_points
from src.yolo_detector import YOLODetector
from src.color_tracker import ColorTracker
import numpy as np

ID_ASSIGNMENT_FRAMES = 30  # Number of initial frames to assign IDs
BATCH_SIZE = 32

class SimpleTracker:
    def __init__(self, max_distance=80):
        self.next_id = 1
        self.tracks = []  # List of dicts: {id, bbox, features, last_seen}
        self.max_distance = max_distance
        self.max_frames_absent = 30

    def bbox_centroid(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def update(self, detections, frame_idx):
        updated_tracks = []
        assigned = set()
        for det in detections:
            if det.get('bbox') is None:
                continue
            det_centroid = self.bbox_centroid(det['bbox'])
            best_track = None
            best_dist = float('inf')
            for track in self.tracks:
                if track['last_seen'] < frame_idx - self.max_frames_absent:
                    continue
                track_centroid = self.bbox_centroid(track['bbox'])
                dist = distance_between_points(det_centroid, track_centroid)
                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_track = track
            if best_track and best_track['id'] not in assigned:
                det['id'] = best_track['id']
                best_track['bbox'] = det['bbox']
                best_track['last_seen'] = frame_idx
                updated_tracks.append(best_track)
                assigned.add(best_track['id'])
            else:
                det['id'] = self.next_id
                updated_tracks.append({'id': self.next_id, 'bbox': det['bbox'], 'last_seen': frame_idx})
                self.next_id += 1
        self.tracks = updated_tracks
        return detections

def setup_logging(log_level=logging.INFO):
    """Configure logging settings"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Soccer Player Re-Identification System (YOLOv11)")
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, default='config/settings.json', help='Config file path')
    parser.add_argument('--save_video', action='store_true', help='Save output video with visualizations')
    parser.add_argument('--save_json', action='store_true', help='Save detection/tracking results as JSON')
    parser.add_argument('--model', type=str, default='models/yolov11.pt', help='Path to YOLOv11 model')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    return parser.parse_args()

def validate_paths(args, logger):
    """Validate input and output paths"""
    if not os.path.exists(args.input):
        logger.error(f"Input video file not found: {args.input}")
        return False
        
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return False
        
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {args.output}: {str(e)}")
        return False
        
    return True

def main():
    # Parse arguments and setup logging
    args = parse_args()
    logger = setup_logging(getattr(logging, args.log_level))
    
    # Validate paths
    if not validate_paths(args, logger):
        return 1

    try:
        # Prepare output directories
        create_output_dirs(args.output)

        # Start timer for speed analysis
        pipeline_start_time = time.time()

        # Load config
        config = Config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # Load video with error handling
        cap = load_video(args.input)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {args.input}")
            return 1
            
        props = get_video_properties(cap)
        if props['frame_count'] == 0:
            logger.error(f"Video file appears to be empty: {args.input}")
            return 1
            
        logger.info(f"Loaded video: {args.input} ({props['frame_count']} frames, {props['fps']} FPS, {props['width']}x{props['height']})")

        # Prepare video writer if needed
        writer = None
        video_out_path = os.path.join(args.output, 'videos', 'output.mp4')
        frames_written = 0
        if args.save_video:
            # Ensure output directory exists
            video_dir = os.path.dirname(video_out_path)
            if not os.path.exists(video_dir):
                try:
                    os.makedirs(video_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create video directory {video_dir}: {e}")
                    return 1
            # Try multiple codecs for compatibility
            codecs_to_try = ['mp4v', 'avc1', 'XVID']
            for codec in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(video_out_path, fourcc, props['fps'], (props['width'], props['height']))
                if writer is not None and writer.isOpened():
                    logger.info(f"VideoWriter opened successfully with codec: {codec}")
                    break
                else:
                    logger.warning(f"VideoWriter failed to open with codec: {codec}")
                    writer = None
            if writer is None or not writer.isOpened():
                logger.error("Failed to open video writer with any supported codec!")
                return 1

        # Initialize modules
        try:
            detector = YOLODetector(args.model, device='cpu')
            color_tracker = ColorTracker(config)
            tracker = SimpleTracker()
        except Exception as e:
            logger.error(f"Failed to initialize modules: {str(e)}")
            return 1

        frame_idx = 0
        results = []
        batch_frames = []
        batch_indices = []
        
        # For consistent team+player IDs (stable across frames)
        team_player_id_map = {}  # global_id: (team, team_player_id)
        team_counters = {'team1': 1, 'team2': 1, 'unknown': 1}
        # For tracking last seen for stability
        last_seen = {}
        max_absent = 50  # frames to keep ID after missing

        # Main processing loop
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                logger.info(f"Processing frame {frame_idx}/{props['frame_count']}")

                batch_frames.append(frame)
                batch_indices.append(frame_idx)

                if len(batch_frames) == BATCH_SIZE or not ret:
                    # Batch inference
                    batch_start_time = time.time()
                    logger.info(f"Running batch inference for frames {batch_indices[0]}-{batch_indices[-1]}")
                    yolo_results = detector.model(batch_frames)  # returns list of results
                    batch_infer_time = time.time() - batch_start_time
                    logger.info(f"Batch inference time: {batch_infer_time:.2f} seconds for {len(batch_frames)} frames (avg {batch_infer_time/len(batch_frames):.3f} s/frame)")
                    for i, result in enumerate(yolo_results):
                        detections = []
                        try:
                            detected_labels = []
                            for box in result.boxes:
                                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                                label = result.names[cls] if hasattr(result, 'names') else str(cls)
                                detected_labels.append(label)
                                # For debugging, draw all detections, not just 'player'
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                detections.append({
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                                    'confidence': conf,
                                    'class': label
                                })
                            logger.info(f"Frame {batch_indices[i]}: {len(detections)} detections, labels: {detected_labels}")
                            tracked = tracker.update(detections, batch_indices[i])
                            vis_frame = batch_frames[i].copy()
                            # Update last seen for stability
                            for det in tracked:
                                global_id = det.get('id', None)
                                if global_id is not None:
                                    last_seen[global_id] = batch_indices[i]
                            # Remove IDs not seen for a while
                            to_remove = [gid for gid, lastf in last_seen.items() if batch_indices[i] - lastf > max_absent]
                            for gid in to_remove:
                                if gid in team_player_id_map:
                                    del team_player_id_map[gid]
                                del last_seen[gid]
                            for det in tracked:
                                if not det.get('bbox') or len(det['bbox']) != 4:
                                    continue
                                x, y, w, h = det['bbox']
                                label = det.get('class', str(cls))
                                conf = det.get('confidence', 0)
                                global_id = det.get('id', None)
                                # Assign team for players using color clustering
                                team = 'unknown'
                                dominant_hsv = None
                                if label == 'player':
                                    color_profile = det.get('color_profile')
                                    if color_profile is not None and len(color_profile) > 0:
                                        b, g, r = color_profile[0]
                                        color = np.uint8([[[b, g, r]]])
                                        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
                                        dominant_hsv = tuple(int(x) for x in hsv)
                                        # Log the dominant HSV color for tuning
                                        logger.info(f"Frame {batch_indices[i]}: Player bbox {det['bbox']} dominant HSV: {dominant_hsv}")
                                    team = color_tracker.assign_team(color_profile)
                                # Assign team+player ID
                                if label == 'player' and global_id is not None:
                                    if global_id not in team_player_id_map:
                                        team_player_id_map[global_id] = (team, team_counters[team])
                                        team_counters[team] += 1
                                    team_label, team_player_id = team_player_id_map[global_id]
                                    if team_label == 'team1':
                                        player_label = f"T1_P{team_player_id}"
                                    elif team_label == 'team2':
                                        player_label = f"T2_P{team_player_id}"
                                    else:
                                        player_label = f"U_P{team_player_id}"
                                elif label == 'goalkeeper':
                                    player_label = 'GK'
                                elif label == 'referee':
                                    player_label = 'REF'
                                else:
                                    player_label = label
                                # Set color and style by team or role
                                if label == 'goalkeeper':
                                    color = (0, 140, 255)  # orange
                                elif label == 'referee':
                                    color = (0, 0, 0)  # black
                                elif label == 'player' and team == 'team1':
                                    color = (255, 0, 0)  # blue
                                elif label == 'player' and team == 'team2':
                                    color = (0, 0, 255)  # red
                                elif label == 'player':
                                    color = (0, 255, 255)  # yellow for unknown team
                                else:
                                    color = (0, 255, 255)  # yellow for unknown/other
                                # Draw rectangle and label above
                                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 3)
                                label_size, _ = cv2.getTextSize(player_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                label_x = x + (w - label_size[0]) // 2
                                label_y = y - 10 if y - 10 > 20 else y + h + 20
                                cv2.rectangle(vis_frame, (label_x - 2, label_y - label_size[1] - 2), (label_x + label_size[0] + 2, label_y + 4), color, -1)
                                cv2.putText(vis_frame, player_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255) if color != (255,255,255) else (0,0,0), 2)
                                # Save player crop
                                if label == 'player' and global_id is not None:
                                    crop = vis_frame[y:y+h, x:x+w]
                                    crop_path = os.path.join(args.output, 'tracks', f'{player_label}_frame_{batch_indices[i]:04d}.jpg')
                                    if crop.size > 0:
                                        cv2.imwrite(crop_path, crop)
                        except Exception as e:
                            logger.error(f"Error during detection/visualization for frame {batch_indices[i]}: {e}")
                            vis_frame = batch_frames[i]  # fallback to original frame
                        if writer:
                            if vis_frame.shape[1] != props['width'] or vis_frame.shape[0] != props['height']:
                                vis_frame = cv2.resize(vis_frame, (props['width'], props['height']))
                            try:
                                writer.write(vis_frame)
                                frames_written += 1
                                logger.info(f"Wrote frame {batch_indices[i]} to video.")
                            except Exception as e:
                                logger.error(f"Failed to write frame {batch_indices[i]}: {e}")
                        results.append({'frame': batch_indices[i], 'detections': [d for d in tracked if 'id' in d] if 'tracked' in locals() else []})
                    batch_frames = []
                    batch_indices = []

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                continue

        # Cleanup
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved output video to {video_out_path}")
            logger.info(f"Total frames written to video: {frames_written}")
        # Log total pipeline speed
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        avg_fps = frames_written / total_time if total_time > 0 else 0
        logger.info(f"Total pipeline time: {total_time:.2f} seconds for {frames_written} frames (avg {avg_fps:.2f} FPS)")

        # Save results as JSON
        if args.save_json:
            try:
                json_out_path = os.path.join(args.output, 'reports', 'results.json')
                save_results(results, json_out_path)
                logger.info(f"Saved results to {json_out_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
                return 1

        logger.info("Processing complete.")
        return 0

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
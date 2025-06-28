# detect.py

import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter, defaultdict
from datetime import datetime
import os

# Install required packages:
# pip install deep-sort-realtime
# pip install ultralytics

try:
    import sys
    sys.path.append('.')
    from deep_sort_realtime.deepsort_tracker import DeepSort
    TRACKING_AVAILABLE = True
    print("DeepSORT loaded successfully")
except ImportError as e:
    print(f"DeepSORT import error: {e}")
    TRACKING_AVAILABLE = False
except Exception as e:
    print(f"DeepSORT initialization error: {e}")
    TRACKING_AVAILABLE = False

# Load the YOLO model
model = YOLO('models/best.pt') 

# Class names (must match the model's order)
CLASS_NAMES = ['Didi', 'HelloRide', 'Meituan']

# Color mapping for different brands
BRAND_COLORS = {
    'Didi': (255, 165, 0),    # Orange
    'HelloRide': (0, 255, 0), # Green  
    'Meituan': (255, 255, 0)  # Yellow
}

class BikeDetector:
    def __init__(self, confidence_threshold=0.3, use_tracking=True):
        global TRACKING_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.use_tracking = use_tracking
        
        # Initialize tracker if available
        if TRACKING_AVAILABLE and use_tracking:
            try:
                self.tracker = DeepSort(max_age=50, n_init=3)
            except Exception as e:
                print(f"Tracker init failed: {e}")
                self.tracker = None
                TRACKING_AVAILABLE = False
        else:
            self.tracker = None
            
        # Statistics tracking
        self.unique_bikes = set()
        self.brand_counts = Counter()
        self.detection_history = []
        self.tracked_objects = defaultdict(list)
        
    def detect_and_track(self, frame, frame_number):
        """Detect bikes and track them across frames"""
        results = model(frame, verbose=False)
        detections = results[0].boxes.data
        
        # Filter detections by confidence
        filtered_detections = []
        detection_list = []
        
        for det in detections:
            confidence = det[4].cpu().numpy().item()
            if confidence >= self.confidence_threshold:
                cls_id = int(det[5].cpu().numpy().item())
                if 0 <= cls_id < len(CLASS_NAMES):
                    x1, y1, x2, y2 = map(int, det[:4].cpu().numpy())
                    label = CLASS_NAMES[cls_id]
                    
                    filtered_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': label,
                        'class_id': cls_id
                    })
                    
                    # Prepare for tracking (bbox format: [x1, y1, w, h])
                    detection_list.append(([x1, y1, x2-x1, y2-y1], confidence, label))
        
        # Apply tracking if available
        tracked_objects = []
        if self.tracker and detection_list:
            tracks = self.tracker.update_tracks(detection_list, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                bbox = track.to_ltwh()  # [left, top, width, height]
                
                # Find corresponding detection
                for det in filtered_detections:
                    det_bbox = det['bbox']
                    # Simple matching by overlap (you could improve this)
                    if self._bbox_overlap(bbox, det_bbox):
                        tracked_objects.append({
                            'track_id': track_id,
                            'bbox': det['bbox'],
                            'confidence': det['confidence'],
                            'class': det['class'],
                            'class_id': det['class_id']
                        })
                        
                        # Update statistics
                        bike_id = f"{det['class']}_{track_id}"
                        if bike_id not in self.unique_bikes:
                            self.unique_bikes.add(bike_id)
                            self.brand_counts[det['class']] += 1
                        
                        # Store tracking history
                        self.tracked_objects[track_id].append({
                            'frame': frame_number,
                            'bbox': det['bbox'],
                            'class': det['class'],
                            'timestamp': datetime.now()
                        })
                        break
        else:
            # If no tracking, treat each detection as separate
            for i, det in enumerate(filtered_detections):
                tracked_objects.append({
                    'track_id': f"det_{frame_number}_{i}",
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class': det['class'], 
                    'class_id': det['class_id']
                })
                pass
        
        return tracked_objects
    
    def _bbox_overlap(self, bbox1, bbox2):
        """Simple bbox overlap check"""
        x1, y1, w1, h1 = bbox1
        x2, y2, x3, y3 = bbox2
        
        # Convert to same format
        overlap_x = max(0, min(x1 + w1, x3) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y3) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        bbox1_area = w1 * h1
        bbox2_area = (x3 - x2) * (y3 - y2)
        
        return overlap_area > 0.3 * min(bbox1_area, bbox2_area)
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes with enhanced information"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['class']
            confidence = det['confidence']
            track_id = det.get('track_id', 'N/A')
            
            # Get color for brand
            color = BRAND_COLORS.get(label, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if isinstance(track_id, str) and track_id.startswith('det_'):
                label_text = f"{label} ({confidence:.2f})"
            else:
                label_text = f"{label} ID:{track_id} ({confidence:.2f})"
            
            # Calculate text size and background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw text background
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'total_detections': sum(self.brand_counts.values()),
            'unique_bikes': len(self.unique_bikes),
            'brand_counts': dict(self.brand_counts),
            'tracked_objects': len(self.tracked_objects)
        }

def detect_bikes_from_video(video_path, output_path="outputs/processed_video.mp4", 
                           confidence_threshold=0.3, use_tracking=True, progress_callback=None):
    """Enhanced video processing with object tracking and progress callbacks"""
    
    # Initialize detector
    detector = BikeDetector(confidence_threshold, use_tracking)
    
    # Store detector reference in callback for real-time stats
    if progress_callback:
        progress_callback.detector = detector
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use H.264 codec for better web compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Check if video writer was initialized successfully
    if not out.isOpened():
        print(f"Error: Could not open video writer with avc1 codec")
        # Try alternative web-compatible codecs
        codecs_to_try = [
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi')
        ]
        
        for codec_name, extension in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            if not output_path.endswith(extension):
                output_path_alt = output_path.rsplit('.', 1)[0] + extension
            else:
                output_path_alt = output_path
                
            out = cv2.VideoWriter(output_path_alt, fourcc, fps, (width, height))
            if out.isOpened():
                output_path = output_path_alt
                print(f"Using {codec_name} codec, output: {output_path}")
                break
        
        if not out.isOpened():
            raise Exception("Could not initialize video writer with any codec")
    
    frame_number = 0
    start_time = datetime.now()
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    print(f"Output path: {output_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track bikes
        detections = detector.detect_and_track(frame, frame_number)
        
        # Draw detections on frame
        annotated_frame = detector.draw_detections(frame.copy(), detections)
        
        # Add frame statistics overlay
        stats = detector.get_statistics()
        stats_text = [
            f"Frame: {frame_number + 1}/{total_frames}",
            f"Detections: {len(detections)}",
            f"Unique Bikes: {stats['unique_bikes']}",
            f"Didi: {stats['brand_counts'].get('Didi', 0)}",
            f"HelloRide: {stats['brand_counts'].get('HelloRide', 0)}",
            f"Meituan: {stats['brand_counts'].get('Meituan', 0)}"
        ]
        
        # Draw statistics
        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(annotated_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Write frame
        out.write(annotated_frame)
        
        frame_number += 1
        
        # Progress callback
        if progress_callback and frame_number % 5 == 0:
            try:
                progress_callback(frame_number, total_frames)
            except Exception as e:
                print(f"Progress callback error: {e}")
        
        # Console progress indicator
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    end_time = datetime.now()
    
    # Verify output file was created
    if not os.path.exists(output_path):
        raise Exception(f"Output video was not created at {output_path}")
    
    # Get file size for verification
    file_size = os.path.getsize(output_path)
    print(f"Output video created: {output_path} (Size: {file_size / (1024*1024):.2f} MB)")
    
    # Final statistics
    final_stats = detector.get_statistics()
    
    return {
        "counts": final_stats['brand_counts'],
        "unique_bikes": final_stats['unique_bikes'],
        "total_detections": final_stats['total_detections'],
        "start_time": start_time,
        "end_time": end_time,
        "duration": str(end_time - start_time),
        "output_video": output_path,
        "fps": fps,
        "total_frames": total_frames,
        "tracking_enabled": use_tracking and TRACKING_AVAILABLE,
        "file_size_mb": file_size / (1024*1024)
    }

# Test function
if __name__ == "__main__":
    # Test with a sample video
    def test_progress(current, total):
        print(f"Test Progress: {current}/{total} ({current/total*100:.1f}%)")
    
    result = detect_bikes_from_video(
        "test_video.mp4", 
        confidence_threshold=0.3, 
        progress_callback=test_progress
    )
    print("Detection Results:")
    for key, value in result.items():
        print(f"{key}: {value}")
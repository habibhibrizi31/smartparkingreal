from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import math
import threading
import queue
import time
from collections import deque
import logging
import os
import sys
import json
import base64
from flask_socketio import SocketIO, emit
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'parking_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
class Config:
    # Fixed parking grid coordinates (x1, y1, x2, y2)
    PARKING_GRID = (0, 185, 810, 450)
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    
    # Tracker parameters
    TRACKER_MAX_AGE = 20
    TRACKER_MIN_HITS = 3
    TRACKER_IOU_THRESHOLD = 0.3
    
    # Performance parameters
    TARGET_FPS = 15
    FRAME_SKIP = 2
    MAX_RESOLUTION_WIDTH = 1280
    BUFFER_SIZE = 3
    
    # Parking calculation
    SLOT_LENGTH_CM = 150  # Average motorcycle parking slot length
    SCALING_FACTOR = 1.0  # Pixels to cm conversion (adjust based on camera distance)
    
    # Violation detection parameters
    VIOLATION_BUFFER_TIME = 3.0  # Seconds to wait before marking as violation
    MIN_VIOLATION_SIZE = 30  # Minimum bounding box size for violation detection

# Global variables for camera and processing
camera = None
model = None
tracker = None
performance_monitor = None
processing_active = False
current_frame = None
current_stats = {}
frame_lock = threading.Lock()

# Enhanced Frame Buffer with better threading
class FrameBuffer:
    def __init__(self, maxsize=Config.BUFFER_SIZE):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.dropped_frames = 0
        
    def put(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            try:
                self.buffer.put_nowait(frame.copy())
            except queue.Full:
                try:
                    self.buffer.get_nowait()
                    self.buffer.put_nowait(frame.copy())
                except queue.Empty:
                    pass
                self.dropped_frames += 1
    
    def get(self):
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            with self.lock:
                return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_dropped_count(self):
        return self.dropped_frames

def convert_to_landscape(frame):
    """Convert frame from portrait to landscape orientation with validation"""
    if frame is None:
        return None
        
    height, width = frame.shape[:2]
    
    if height > width * 1.2:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        logger.info(f"Frame rotated from portrait ({width}x{height}) to landscape ({frame.shape[1]}x{frame.shape[0]})")
    
    return frame

def resize_frame_if_needed(frame, max_width=Config.MAX_RESOLUTION_WIDTH):
    """Resize frame if it's too large for optimal processing"""
    if frame is None:
        return None, 1.0
        
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_frame, scale
    
    return frame, 1.0

# Enhanced Multi-Object Tracker with Violation Detection
class EnhancedTracker:
    def __init__(self, max_age=Config.TRACKER_MAX_AGE, min_hits=Config.TRACKER_MIN_HITS, 
                 iou_threshold=Config.TRACKER_IOU_THRESHOLD):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
        self.frame_count = 0
        self.violation_tracks = {}
        self.current_time = time.time()
    
    def calculate_iou_optimized(self, boxes1, boxes2):
        """Optimized IoU calculation with better numerical stability"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        boxes1 = np.array(boxes1, dtype=np.float32)
        boxes2 = np.array(boxes2, dtype=np.float32)
        
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, None] + area2[None, :] - intersection
        union = np.maximum(union, 1e-6)
        iou = intersection / union
        
        return iou
    
    def update(self, detections_in_grid, detections_outside_grid):
        """Enhanced tracking update with violation detection"""
        self.frame_count += 1
        self.current_time = time.time()
        
        all_detections = []
        detection_locations = []
        
        for det in detections_in_grid:
            all_detections.append(det)
            detection_locations.append('inside')
            
        for det in detections_outside_grid:
            all_detections.append(det)
            detection_locations.append('outside')
        
        if len(all_detections) == 0:
            for track in self.tracks:
                track['age'] += 1
                track['hit_streak'] = 0
            self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
            return self.get_active_tracks()
        
        predicted_boxes = []
        for track in self.tracks:
            predicted_box = track['box'].copy()
            if 'velocity' in track and track['age'] < 5:
                damping = 0.8 ** track['age']
                predicted_box[0] += track['velocity'][0] * damping
                predicted_box[1] += track['velocity'][1] * damping
                predicted_box[2] += track['velocity'][0] * damping
                predicted_box[3] += track['velocity'][1] * damping
            predicted_boxes.append(predicted_box)
        
        if len(self.tracks) > 0:
            detection_boxes = [det[:4] for det in all_detections]
            iou_matrix = self.calculate_iou_optimized(detection_boxes, predicted_boxes)
            matched_pairs = self._greedy_assignment(iou_matrix)
            
            used_detections = set(pair[0] for pair in matched_pairs)
            used_tracks = set(pair[1] for pair in matched_pairs)
            
            for det_idx, track_idx in matched_pairs:
                old_box = self.tracks[track_idx]['box']
                new_box = np.array(all_detections[det_idx][:4])
                
                velocity = new_box[:2] - old_box[:2]
                if 'velocity' in self.tracks[track_idx]:
                    self.tracks[track_idx]['velocity'] = (
                        0.7 * self.tracks[track_idx]['velocity'] + 0.3 * velocity
                    )
                else:
                    self.tracks[track_idx]['velocity'] = velocity
                
                self.tracks[track_idx]['box'] = new_box
                self.tracks[track_idx]['confidence'] = all_detections[det_idx][4]
                self.tracks[track_idx]['age'] = 0
                self.tracks[track_idx]['hits'] += 1
                self.tracks[track_idx]['hit_streak'] += 1
                self.tracks[track_idx]['last_seen'] = self.frame_count
                
                location = detection_locations[det_idx]
                self.tracks[track_idx]['location'] = location
                self._update_violation_status(track_idx, location)
            
            for det_idx in range(len(all_detections)):
                if det_idx not in used_detections:
                    location = detection_locations[det_idx]
                    self._create_new_track(all_detections[det_idx], location)
            
            for track_idx in range(len(self.tracks)):
                if track_idx not in used_tracks:
                    self.tracks[track_idx]['age'] += 1
                    self.tracks[track_idx]['hit_streak'] = 0
        else:
            for i, det in enumerate(all_detections):
                location = detection_locations[i]
                self._create_new_track(det, location)
        
        active_track_ids = set()
        self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
        for track in self.tracks:
            active_track_ids.add(track['id'])
        
        self.violation_tracks = {k: v for k, v in self.violation_tracks.items() 
                               if k in active_track_ids}
        
        return self.get_active_tracks()
    
    def _update_violation_status(self, track_idx, location):
        """Update violation status for a track"""
        track_id = self.tracks[track_idx]['id']
        
        if location == 'outside':
            if track_id not in self.violation_tracks:
                self.violation_tracks[track_id] = self.current_time
                logger.info(f"Track {track_id}: Started violation monitoring")
            
            violation_duration = self.current_time - self.violation_tracks[track_id]
            self.tracks[track_idx]['violation_status'] = 'potential' if violation_duration < Config.VIOLATION_BUFFER_TIME else 'confirmed'
            self.tracks[track_idx]['violation_duration'] = violation_duration
        else:
            if track_id in self.violation_tracks:
                logger.info(f"Track {track_id}: Violation cleared - moved to parking area")
                del self.violation_tracks[track_id]
            self.tracks[track_idx]['violation_status'] = 'none'
            self.tracks[track_idx]['violation_duration'] = 0
    
    def _greedy_assignment(self, iou_matrix):
        """Greedy assignment algorithm for track-detection matching"""
        matched_pairs = []
        
        while True:
            max_iou = 0
            best_match = None
            
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > self.iou_threshold:
                        max_iou = iou_matrix[i, j]
                        best_match = (i, j)
            
            if best_match is None:
                break
                
            matched_pairs.append(best_match)
            iou_matrix[best_match[0], :] = 0
            iou_matrix[:, best_match[1]] = 0
        
        return matched_pairs
    
    def _create_new_track(self, detection, location):
        """Create a new track from detection"""
        new_track = {
            'id': self.track_id_count,
            'box': np.array(detection[:4]),
            'confidence': detection[4],
            'age': 0,
            'hits': 1,
            'hit_streak': 1,
            'velocity': np.array([0.0, 0.0]),
            'last_seen': self.frame_count,
            'location': location,
            'violation_status': 'none',
            'violation_duration': 0
        }
        
        if location == 'outside':
            self.violation_tracks[self.track_id_count] = self.current_time
            new_track['violation_status'] = 'potential'
        
        self.tracks.append(new_track)
        self.track_id_count += 1
    
    def get_active_tracks(self):
        """Get tracks that should be displayed"""
        result = []
        for track in self.tracks:
            if (track['hits'] >= self.min_hits and track['hit_streak'] > 0) or track['age'] == 0:
                x1, y1, x2, y2 = track['box']
                result.append([x1, y1, x2, y2, track['id']])
        return np.array(result, dtype=np.float32) if result else np.empty((0, 5))
    
    def get_violation_info(self):
        """Get current violation information"""
        violations = {'potential': 0, 'confirmed': 0, 'details': []}
        
        for track in self.tracks:
            if (track['hits'] >= self.min_hits and track['hit_streak'] > 0) or track['age'] == 0:
                if track.get('violation_status') == 'potential':
                    violations['potential'] += 1
                elif track.get('violation_status') == 'confirmed':
                    violations['confirmed'] += 1
                    
                if track.get('violation_status') in ['potential', 'confirmed']:
                    violations['details'].append({
                        'track_id': track['id'],
                        'status': track['violation_status'],
                        'duration': track.get('violation_duration', 0),
                        'box': track['box']
                    })
        
        return violations

class PerformanceMonitor:
    """Enhanced performance monitoring with violation tracking"""
    def __init__(self, window_size=30):
        self.detection_times = deque(maxlen=window_size)
        self.total_frame_times = deque(maxlen=window_size)
        self.processing_fps_history = deque(maxlen=window_size)
        self.last_update = time.time()
        self.total_violations_detected = 0
        
    def update_detection_time(self, detection_time):
        self.detection_times.append(detection_time)
        
    def update_frame_time(self, frame_time):
        self.total_frame_times.append(frame_time)
        current_time = time.time()
        if current_time - self.last_update > 0:
            fps = 1.0 / (current_time - self.last_update)
            self.processing_fps_history.append(fps)
        self.last_update = current_time
        
    def update_violations(self, violation_count):
        self.total_violations_detected = max(self.total_violations_detected, violation_count)
        
    def get_stats(self):
        return {
            'avg_detection_ms': (sum(self.detection_times) / max(len(self.detection_times), 1)) * 1000,
            'avg_frame_ms': (sum(self.total_frame_times) / max(len(self.total_frame_times), 1)) * 1000,
            'avg_processing_fps': sum(self.processing_fps_history) / max(len(self.processing_fps_history), 1),
            'detection_ms': self.detection_times[-1] * 1000 if self.detection_times else 0,
            'processing_fps': self.processing_fps_history[-1] if self.processing_fps_history else 0,
            'total_violations': self.total_violations_detected
        }

def load_yolo_model():
    """Load YOLO model with error handling"""
    try:
        model_path = "best.pt"  # Place your model file in the same directory
        if not os.path.exists(model_path):
            logger.info("Using YOLOv8n model (will be downloaded automatically)")
            model_path = 'yolov8n.pt'
        
        model = YOLO(model_path)
        model.fuse()
        logger.info("YOLO model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None

def detect_motorcycles_optimized(model, frame, conf_threshold=Config.CONFIDENCE_THRESHOLD):
    """Enhanced motorcycle detection"""
    if frame is None or model is None:
        return np.empty((0, 5))
        
    try:
        results = model(frame, 
                       verbose=False, 
                       conf=conf_threshold, 
                       iou=Config.IOU_THRESHOLD)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Accept multiple vehicle classes (car, motorcycle, bus, truck)
                    if class_id in [0, 1, 2, 3, 5, 7] and confidence > conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        if (box_width > Config.MIN_VIOLATION_SIZE and 
                            box_height > Config.MIN_VIOLATION_SIZE and 
                            x2 > x1 and y2 > y1):
                            detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(detections, dtype=np.float32) if detections else np.empty((0, 5))
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return np.empty((0, 5))

def is_in_parking_grid(box, grid):
    """Check if bounding box overlaps significantly with parking grid"""
    x1, y1, x2, y2 = box
    gx1, gy1, gx2, gy2 = grid
    
    overlap_x1 = max(x1, gx1)
    overlap_y1 = max(y1, gy1)
    overlap_x2 = min(x2, gx2)
    overlap_y2 = min(y2, gy2)
    
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return False
    
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    box_area = (x2 - x1) * (y2 - y1)
    
    overlap_ratio = overlap_area / max(box_area, 1)
    return overlap_ratio > 0.5

def calculate_parking_slots(centers, parking_grid):
    """Enhanced parking slot calculation"""
    if len(centers) == 0:
        parking_width_pixels = parking_grid[2] - parking_grid[0]
        parking_width_cm = parking_width_pixels / Config.SCALING_FACTOR
        total_slots = max(0, math.floor(parking_width_cm / Config.SLOT_LENGTH_CM))
        return total_slots
    
    sorted_centers = sorted(centers, key=lambda x: x[0])
    
    total_slots = 0
    left_boundary = parking_grid[0]
    right_boundary = parking_grid[2]
    
    if sorted_centers:
        distance_pixels = abs(sorted_centers[0][0] - left_boundary)
        distance_cm = distance_pixels / Config.SCALING_FACTOR
        slots_before = max(0, math.floor(distance_cm / Config.SLOT_LENGTH_CM))
        total_slots += slots_before
    
    for i in range(len(sorted_centers) - 1):
        distance_pixels = abs(sorted_centers[i+1][0] - sorted_centers[i][0])
        distance_cm = distance_pixels / Config.SCALING_FACTOR
        slots_between = max(0, math.floor(distance_cm / Config.SLOT_LENGTH_CM) - 1)
        total_slots += slots_between
    
    if sorted_centers:
        distance_pixels = abs(right_boundary - sorted_centers[-1][0])
        distance_cm = distance_pixels / Config.SCALING_FACTOR
        slots_after = max(0, math.floor(distance_cm / Config.SLOT_LENGTH_CM))
        total_slots += slots_after
    
    return total_slots

def draw_enhanced_visualization(frame, centers, parking_grid, tracked_data, violation_data):
    """Enhanced visualization with violation indicators"""
    if frame is None:
        return frame
        
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 165, 0)   # Orange
    ]
    
    violation_colors = {
        'potential': (0, 165, 255),  # Orange
        'confirmed': (0, 0, 255),    # Red
        'none': (0, 255, 0)          # Green
    }
    
    # Draw parking grid
    cv2.rectangle(frame, (parking_grid[0], parking_grid[1]), 
                  (parking_grid[2], parking_grid[3]), (0, 255, 0), 3)
    
    # Draw tracked objects
    for data in tracked_data:
        x1, y1, x2, y2, track_id, x_center, y_center, location, violation_status = data
        
        if violation_status in violation_colors:
            color = violation_colors[violation_status]
        else:
            color_idx = int(track_id) % len(colors)
            color = colors[color_idx]
        
        thickness = 5 if violation_status in ['potential', 'confirmed'] else 3
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        violation_text = ""
        if violation_status == 'potential':
            violation_text = " [POTENTIAL VIOLATION]"
        elif violation_status == 'confirmed':
            violation_text = " [VIOLATION!]"
        
        text = f'Vehicle-{int(track_id)}{violation_text}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        bg_height = text_size[1] + 15
        bg_width = text_size[0] + 15
        
        cv2.rectangle(frame, (int(x1), int(y1) - bg_height), 
                     (int(x1) + bg_width, int(y1)), color, -1)
        cv2.putText(frame, text, (int(x1) + 5, int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.circle(frame, (int(x_center), int(y_center)), 5, color, -1)
        cv2.circle(frame, (int(x_center), int(y_center)), 5, (255, 255, 255), 2)
    
    return frame

def process_frame():
    """Process video frame for detection"""
    global current_frame, current_stats, processing_active
    
    if camera is None or model is None or tracker is None or performance_monitor is None:
        return
    
    frame_start_time = time.time()
    
    # Read frame from camera
    ret, frame = camera.read()
    if not ret:
        return
    
    # Convert and resize frame
    frame = convert_to_landscape(frame)
    frame, _ = resize_frame_if_needed(frame)
    
    if frame is None:
        return
    
    # Detect objects
    detection_start = time.time()
    detections = detect_motorcycles_optimized(model, frame)
    detection_time = time.time() - detection_start
    performance_monitor.update_detection_time(detection_time)
    
    # Separate detections
    detections_in_grid = []
    detections_outside_grid = []
    
    for detection in detections:
        if is_in_parking_grid(detection[:4], Config.PARKING_GRID):
            detections_in_grid.append(detection)
        else:
            detections_outside_grid.append(detection)
    
    detections_in_grid = np.array(detections_in_grid) if detections_in_grid else np.empty((0, 5))
    detections_outside_grid = np.array(detections_outside_grid) if detections_outside_grid else np.empty((0, 5))
    
    # Update tracker
    tracked_objects = tracker.update(detections_in_grid, detections_outside_grid)
    violations = tracker.get_violation_info()
    performance_monitor.update_violations(violations['confirmed'])
    
    # Prepare visualization data
    centers = []
    tracked_data = []
    
    for track in tracker.tracks:
        if (track['hits'] >= tracker.min_hits and track['hit_streak'] > 0) or track['age'] == 0:
            x1, y1, x2, y2 = track['box']
            track_id = track['id']
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            location = track.get('location', 'inside')
            violation_status = track.get('violation_status', 'none')
            
            if location == 'inside':
                centers.append((x_center, y_center))
            
            tracked_data.append((x1, y1, x2, y2, track_id, x_center, y_center, location, violation_status))
    
    # Calculate available slots
    available_slots = calculate_parking_slots(centers, Config.PARKING_GRID)
    
    # Draw visualizations
    frame = draw_enhanced_visualization(frame, centers, Config.PARKING_GRID, tracked_data, violations)
    
    # Update performance
    frame_time = time.time() - frame_start_time
    performance_monitor.update_frame_time(frame_time)
    perf_stats = performance_monitor.get_stats()
    
    # Update global stats
    with frame_lock:
        current_frame = frame.copy()
        current_stats = {
            'available_slots': available_slots,
            'vehicles_in_grid': len(centers),
            'total_vehicles': len(tracked_data),
            'violations': violations,
            'performance': perf_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    # Emit real-time data via WebSocket
    socketio.emit('detection_update', current_stats)

def camera_thread():
    """Camera processing thread"""
    global processing_active
    
    while processing_active:
        try:
            process_frame()
            time.sleep(1/Config.TARGET_FPS)
        except Exception as e:
            logger.error(f"Error in camera thread: {e}")
            time.sleep(0.1)

# Flask Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1/Config.TARGET_FPS)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current detection statistics"""
    with frame_lock:
        return jsonify(current_stats)

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    global camera, model, tracker, performance_monitor, processing_active
    
    try:
        if processing_active:
            return jsonify({'status': 'error', 'message': 'Detection already active'})
        
        # Initialize camera
        camera = cv2.VideoCapture(0)  # Use default camera
        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Could not open camera'})
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
        
        # Load model
        model = load_yolo_model()
        if model is None:
            camera.release()
            return jsonify({'status': 'error', 'message': 'Could not load YOLO model'})
        
        # Initialize tracker and performance monitor
        tracker = EnhancedTracker()
        performance_monitor = PerformanceMonitor()
        
        # Start processing
        processing_active = True
        threading.Thread(target=camera_thread, daemon=True).start()
        
        logger.info("Detection system started successfully")
        return jsonify({'status': 'success', 'message': 'Detection started'})
        
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        if camera:
            camera.release()
        processing_active = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    global camera, processing_active
    
    try:
        processing_active = False
        
        if camera:
            camera.release()
            camera = None
        
        logger.info("Detection system stopped")
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
        
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/update_config', methods=['POST'])
def update_config():
    """Update configuration parameters"""
    try:
        data = request.get_json()
        
        if 'parking_grid' in data:
            Config.PARKING_GRID = tuple(data['parking_grid'])
        if 'confidence_threshold' in data:
            Config.CONFIDENCE_THRESHOLD = data['confidence_threshold']
        if 'violation_buffer_time' in data:
            Config.VIOLATION_BUFFER_TIME = data['violation_buffer_time']
        
        return jsonify({'status': 'success', 'message': 'Configuration updated'})
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get_config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'parking_grid': Config.PARKING_GRID,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'violation_buffer_time': Config.VIOLATION_BUFFER_TIME,
        'slot_length_cm': Config.SLOT_LENGTH_CM
    })

@app.route('/api/capture_screenshot', methods=['POST'])
def capture_screenshot():
    """Capture current frame as screenshot"""
    try:
        with frame_lock:
            if current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"parking_detection_{timestamp}.jpg"
                cv2.imwrite(filename, current_frame)
                return jsonify({'status': 'success', 'filename': filename})
            else:
                return jsonify({'status': 'error', 'message': 'No frame available'})
                
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to parking detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request from client"""
    with frame_lock:
        emit('stats_update', current_stats)

if __name__ == '__main__':
    logger.info("Starting Web-based Parking Detection System...")
    logger.info("Access the application at: http://localhost:5000")
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
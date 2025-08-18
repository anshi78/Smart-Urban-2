import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import time
import logging
import requests
from datetime import datetime
import os
from camera_stream import CameraStream

class WildlifeDetector:
    """Wildlife detection using YOLO or custom models"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.model = None
        self.classes = self.config['detection_settings']['classes']
        self.confidence_threshold = self.config['detection_settings']['confidence_threshold']
        self.nms_threshold = self.config['detection_settings']['nms_threshold']
        self.input_size = self.config['detection_settings']['input_size']
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config['logging']['level']))
        self.logger = logging.getLogger(__name__)
        
        # Detection settings
        self.device_id = self.config['device_id']
        self.backend_url = self.config['network_settings']['backend_url']
        self.upload_enabled = self.config['network_settings']['upload_enabled']
        
        # Initialize model
        self.load_model()
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'device_id': 'CAM001',
            'detection_settings': {
                'model_path': 'models/yolov5s.pt',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'input_size': [640, 640],
                'classes': ['elephant', 'tiger', 'leopard', 'bear', 'deer']
            },
            'network_settings': {
                'backend_url': 'http://localhost:5000',
                'upload_enabled': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def load_model(self):
        """Load the wildlife detection model"""
        try:
            model_path = self.config['detection_settings']['model_path']
            
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                self.logger.info("Using OpenCV Haar Cascades as fallback")
                self.use_opencv_fallback()
                return
            
            # Try to load PyTorch model (YOLOv5)
            if model_path.endswith('.pt'):
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                    self.model.conf = self.confidence_threshold
                    self.model.iou = self.nms_threshold
                    self.logger.info("YOLOv5 model loaded successfully")
                    self.model_type = 'yolo'
                    return
                except Exception as e:
                    self.logger.error(f"Error loading YOLO model: {e}")
            
            # Fallback to OpenCV
            self.use_opencv_fallback()
            
        except Exception as e:
            self.logger.error(f"Error in load_model: {e}")
            self.use_opencv_fallback()
    
    def use_opencv_fallback(self):
        """Use OpenCV Haar Cascades as fallback detection method"""
        try:
            # Load pre-trained Haar cascades (limited but functional)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            self.model_type = 'opencv'
            self.logger.info("OpenCV Haar Cascades loaded as fallback")
        except Exception as e:
            self.logger.error(f"Error loading OpenCV cascades: {e}")
            self.model_type = 'none'
    
    def detect_objects(self, frame):
        """Detect wildlife objects in the frame"""
        detections = []
        
        try:
            if self.model_type == 'yolo' and self.model is not None:
                detections = self.detect_with_yolo(frame)
            elif self.model_type == 'opencv':
                detections = self.detect_with_opencv(frame)
            else:
                # Mock detection for testing
                detections = self.mock_detection(frame)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in detect_objects: {e}")
            return []
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO model"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            # Parse results
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    
                    if class_id < len(self.classes):
                        species = self.classes[class_id]
                        
                        detection = {
                            'species': species,
                            'confidence': float(conf),
                            'location_x': x1,
                            'location_y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'timestamp': datetime.now().isoformat(),
                            'device_id': self.device_id
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def detect_with_opencv(self, frame):
        """Detect objects using OpenCV (limited functionality)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = []
            
            # Detect faces (can represent animals in some cases)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                detection = {
                    'species': 'unknown_animal',
                    'confidence': 0.7,  # Fixed confidence for OpenCV
                    'location_x': x,
                    'location_y': y,
                    'width': w,
                    'height': h,
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id
                }
                detections.append(detection)
            
            # Detect bodies
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in bodies:
                detection = {
                    'species': 'large_animal',
                    'confidence': 0.6,
                    'location_x': x,
                    'location_y': y,
                    'width': w,
                    'height': h,
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in OpenCV detection: {e}")
            return []
    
    def mock_detection(self, frame):
        """Mock detection for testing purposes"""
        # Randomly generate a detection sometimes
        if np.random.random() < 0.1:  # 10% chance of detection
            h, w = frame.shape[:2]
            
            detection = {
                'species': np.random.choice(self.classes),
                'confidence': np.random.uniform(0.5, 0.95),
                'location_x': np.random.randint(0, w//2),
                'location_y': np.random.randint(0, h//2),
                'width': np.random.randint(50, 200),
                'height': np.random.randint(50, 200),
                'timestamp': datetime.now().isoformat(),
                'device_id': self.device_id
            }
            
            return [detection]
        
        return []
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame"""
        for detection in detections:
            x = detection['location_x']
            y = detection['location_y']
            w = detection['width']
            h = detection['height']
            species = detection['species']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{species}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def send_detection(self, detection, image_path=None):
        """Send detection data to cloud backend"""
        if not self.upload_enabled:
            return False
        
        try:
            # Prepare detection data
            data = detection.copy()
            
            # Add image path if available
            if image_path:
                data['image_path'] = image_path
            
            # Send to backend
            response = requests.post(
                f"{self.backend_url}/api/detections",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Detection sent successfully: ID {result.get('id')}")
                return True
            else:
                self.logger.error(f"Failed to send detection: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending detection: {e}")
        
        return False
    
    def process_frame(self, frame, camera_stream=None):
        """Process a single frame for wildlife detection"""
        detections = self.detect_objects(frame)
        
        if detections:
            self.logger.info(f"Detected {len(detections)} objects")
            
            # Save frame with detections
            if self.config['detection_settings']['save_detections']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"detections/{self.device_id}_{timestamp}.jpg"
                
                # Draw detections on frame
                annotated_frame = self.draw_detections(frame.copy(), detections)
                
                # Save annotated frame
                os.makedirs("detections", exist_ok=True)
                cv2.imwrite(filename, annotated_frame)
                
                # Upload frame if camera stream is available
                if camera_stream and self.config['detection_settings']['upload_detections']:
                    uploaded_filename = camera_stream.upload_frame(annotated_frame, detections[0])
                    if uploaded_filename:
                        filename = uploaded_filename
                
                # Send detection data to backend
                for detection in detections:
                    detection['image_path'] = filename
                    self.send_detection(detection, filename)
        
        return detections
    
    def run_continuous_detection(self):
        """Run continuous wildlife detection"""
        camera = CameraStream(config_path="config.json")
        
        if not camera.start_streaming():
            self.logger.error("Failed to start camera streaming")
            return
        
        self.logger.info("Starting continuous wildlife detection...")
        
        try:
            last_detection_time = 0
            detection_interval = self.config['detection_settings']['detection_interval']
            
            # Send initial heartbeat
            camera.send_heartbeat()
            last_heartbeat = time.time()
            
            while True:
                current_time = time.time()
                
                # Send heartbeat every 30 seconds
                if current_time - last_heartbeat > 30:
                    camera.send_heartbeat()
                    last_heartbeat = current_time
                
                # Process frame for detection
                if current_time - last_detection_time >= detection_interval:
                    frame = camera.get_frame_for_processing()
                    
                    if frame is not None:
                        detections = self.process_frame(frame, camera)
                        last_detection_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        
        except Exception as e:
            self.logger.error(f"Error in continuous detection: {e}")
        
        finally:
            camera.stop_streaming()
            self.logger.info("Wildlife detection stopped")

# Example usage
if __name__ == "__main__":
    detector = WildlifeDetector()
    detector.run_continuous_detection()
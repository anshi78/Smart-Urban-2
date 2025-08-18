import cv2
import numpy as np
import threading
import time
import json
import requests
from datetime import datetime
import os
import logging
from queue import Queue

class CameraStream:
    """Handles camera streaming and frame capture for wildlife detection"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.camera = None
        self.streaming = False
        self.frame_queue = Queue(maxsize=10)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Camera settings
        self.camera_index = self.config.get('camera_index', 0)
        self.resolution = self.config.get('resolution', [640, 480])
        self.fps = self.config.get('fps', 30)
        self.device_id = self.config.get('device_id', 'CAM001')
        
        # Cloud backend settings
        self.backend_url = self.config.get('backend_url', 'http://localhost:5000')
        self.upload_enabled = self.config.get('upload_enabled', True)
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in config file {config_path}")
            return {}
    
    def initialize_camera(self):
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                self.logger.error("Failed to capture test frame")
                return False
            
            self.logger.info(f"Camera initialized successfully: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
    
    def start_streaming(self):
        """Start the camera streaming thread"""
        if self.streaming:
            self.logger.warning("Camera is already streaming")
            return False
        
        if not self.initialize_camera():
            return False
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        self.logger.info("Camera streaming started")
        return True
    
    def stop_streaming(self):
        """Stop the camera streaming"""
        self.streaming = False
        
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=5)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.logger.info("Camera streaming stopped")
    
    def _stream_worker(self):
        """Worker thread for continuous frame capture"""
        frame_count = 0
        last_fps_time = time.time()
        
        while self.streaming:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Update current frame with thread safety
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Add frame to queue for processing (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    # Queue is full, skip this frame
                    pass
                
                frame_count += 1
                
                # Log FPS every 100 frames
                if frame_count % 100 == 0:
                    current_time = time.time()
                    fps = 100 / (current_time - last_fps_time)
                    self.logger.debug(f"Actual FPS: {fps:.2f}")
                    last_fps_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in stream worker: {e}")
                break
        
        self.streaming = False
    
    def get_frame(self):
        """Get the latest frame (thread-safe)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_frame_for_processing(self):
        """Get a frame from the processing queue"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
    
    def save_frame(self, frame, filename=None):
        """Save a frame to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"captured_frames/{self.device_id}_{timestamp}.jpg"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Frame saved: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return None
    
    def upload_frame(self, frame, detection_data=None):
        """Upload frame to cloud backend"""
        if not self.upload_enabled:
            return None
        
        try:
            # Save frame temporarily
            temp_filename = f"temp_{int(time.time())}.jpg"
            cv2.imwrite(temp_filename, frame)
            
            # Upload to backend
            with open(temp_filename, 'rb') as f:
                files = {'file': f}
                data = {'device_id': self.device_id}
                
                if detection_data:
                    data.update(detection_data)
                
                response = requests.post(
                    f"{self.backend_url}/api/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"Frame uploaded successfully: {result.get('filename')}")
                    
                    # Clean up temp file
                    os.remove(temp_filename)
                    
                    return result.get('filename')
                else:
                    self.logger.error(f"Upload failed: {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Error uploading frame: {e}")
        
        # Clean up temp file on error
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass
        
        return None
    
    def send_heartbeat(self):
        """Send heartbeat to cloud backend"""
        if not self.upload_enabled:
            return False
        
        try:
            data = {
                'name': self.config.get('device_name', self.device_id),
                'location': self.config.get('location', 'Unknown'),
                'model': self.config.get('model', 'Generic Camera'),
                'firmware_version': self.config.get('firmware_version', '1.0.0')
            }
            
            response = requests.post(
                f"{self.backend_url}/api/devices/{self.device_id}/heartbeat",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.debug("Heartbeat sent successfully")
                return True
            else:
                self.logger.error(f"Heartbeat failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
        
        return False
    
    def get_camera_info(self):
        """Get camera information and status"""
        info = {
            'device_id': self.device_id,
            'streaming': self.streaming,
            'resolution': self.resolution,
            'fps': self.fps,
            'queue_size': self.frame_queue.qsize(),
            'backend_url': self.backend_url,
            'upload_enabled': self.upload_enabled
        }
        
        if self.camera:
            info.update({
                'actual_width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'actual_height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'actual_fps': self.camera.get(cv2.CAP_PROP_FPS)
            })
        
        return info
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_streaming()

# Example usage and testing
if __name__ == "__main__":
    # Create camera stream instance
    camera = CameraStream()
    
    try:
        # Start streaming
        if camera.start_streaming():
            print("Camera streaming started. Press 'q' to quit.")
            
            # Display frames for testing
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    cv2.imshow('Wildlife Camera', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Send heartbeat every 30 seconds
                time.sleep(0.1)
        
        else:
            print("Failed to start camera streaming")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        camera.stop_streaming()
        cv2.destroyAllWindows()
        print("Camera stream stopped")
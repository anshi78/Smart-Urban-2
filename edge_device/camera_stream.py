import cv2
import numpy as np
import time
import threading
from typing import Optional, Callable, Dict, Any
import logging
from utils import load_config, setup_logging

class CameraStream:
    """Camera stream management class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_config = config.get('camera', {})
        self.logger = logging.getLogger(__name__)
        
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.motion_callback = None
        self.last_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Camera settings
        self.resolution = tuple(self.camera_config.get('resolution', [1920, 1080]))
        self.target_fps = self.camera_config.get('fps', 30)
        self.rotation = self.camera_config.get('rotation', 0)
        self.brightness = self.camera_config.get('brightness', 50)
        self.contrast = self.camera_config.get('contrast', 50)
        self.saturation = self.camera_config.get('saturation', 50)
        self.gain = self.camera_config.get('gain', 0)
        
        # Motion detection settings
        self.motion_threshold = config.get('detection', {}).get('motion_threshold', 0.1)
        self.min_motion_area = config.get('detection', {}).get('min_motion_area', 500)
        
        self.logger.info(f"Camera initialized with resolution {self.resolution}, FPS {self.target_fps}")
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera at index {camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation / 100.0)
            self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for frame processing"""
        self.frame_callback = callback
    
    def set_motion_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Set callback function for motion detection"""
        self.motion_callback = callback
    
    def start_stream(self):
        """Start camera stream in a separate thread"""
        if self.cap is None:
            if not self.initialize_camera():
                return False
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        self.logger.info("Camera stream started")
        return True
    
    def stop_stream(self):
        """Stop camera stream"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.logger.info("Camera stream stopped")
    
    def _stream_loop(self):
        """Main stream processing loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Apply rotation if needed
                if self.rotation != 0:
                    frame = self._rotate_frame(frame, self.rotation)
                
                # Update FPS counter
                self._update_fps()
                
                # Process frame
                self._process_frame(frame)
                
                # Check for motion
                if self.motion_callback and self.last_frame is not None:
                    motion_score = self._detect_motion(frame, self.last_frame)
                    if motion_score > self.motion_threshold:
                        self.motion_callback(frame, motion_score)
                
                self.last_frame = frame.copy()
                self.frame_count += 1
                
                # Control frame rate
                time.sleep(1.0 / self.target_fps)
                
            except Exception as e:
                self.logger.error(f"Error in stream loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray):
        """Process individual frame"""
        if self.frame_callback:
            try:
                self.frame_callback(frame)
            except Exception as e:
                self.logger.error(f"Error in frame callback: {e}")
    
    def _detect_motion(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> float:
        """Detect motion between frames"""
        try:
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            current_blur = cv2.GaussianBlur(current_gray, (21, 21), 0)
            previous_blur = cv2.GaussianBlur(previous_gray, (21, 21), 0)
            
            # Calculate difference
            frame_delta = cv2.absdiff(previous_blur, current_blur)
            
            # Apply threshold
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate to fill holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate motion score
            motion_area = 0
            for contour in contours:
                if cv2.contourArea(contour) > self.min_motion_area:
                    motion_area += cv2.contourArea(contour)
            
            # Normalize by frame area
            frame_area = current_frame.shape[0] * current_frame.shape[1]
            motion_score = motion_area / frame_area
            
            return motion_score
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
            return 0.0
    
    def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Rotate frame by specified angle"""
        if angle == 0:
            return frame
        
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        
        return rotated_frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_frame_count(self) -> int:
        """Get total frame count"""
        return self.frame_count
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.cap is None:
            if not self.initialize_camera():
                return None
        
        ret, frame = self.cap.read()
        if ret:
            if self.rotation != 0:
                frame = self._rotate_frame(frame, self.rotation)
            return frame
        
        return None
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set camera property"""
        if self.cap is None:
            return False
        
        return self.cap.set(property_id, value)
    
    def get_camera_property(self, property_id: int) -> float:
        """Get camera property"""
        if self.cap is None:
            return 0.0
        
        return self.cap.get(property_id)
    
    def is_streaming(self) -> bool:
        """Check if stream is running"""
        return self.is_running and self.cap is not None and self.cap.isOpened()

def create_camera_stream(config_path: str = "config.json") -> CameraStream:
    """Factory function to create camera stream"""
    config = load_config(config_path)
    return CameraStream(config)

if __name__ == "__main__":
    # Test camera stream
    config = load_config()
    logger = setup_logging(config)
    
    camera = CameraStream(config)
    
    def frame_callback(frame):
        print(f"Frame received: {frame.shape}")
    
    def motion_callback(frame, score):
        print(f"Motion detected: {score:.3f}")
    
    camera.set_frame_callback(frame_callback)
    camera.set_motion_callback(motion_callback)
    
    if camera.start_stream():
        try:
            time.sleep(10)  # Run for 10 seconds
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop_stream()
    else:
        print("Failed to start camera stream")
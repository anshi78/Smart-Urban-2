import cv2
import numpy as np
import time
from typing import Callable, Optional, Tuple, List, Dict, Any
import logging
from utils import load_config

class MotionDetector:
    """Motion detection using background subtraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_config = config.get('detection', {})
        self.logger = logging.getLogger(__name__)
        
        # Motion detection settings
        self.motion_threshold = self.detection_config.get('motion_threshold', 0.1)
        self.min_motion_area = self.detection_config.get('min_motion_area', 500)
        self.learning_rate = 0.01
        self.history_length = 5
        
        # Background subtraction
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Motion history
        self.motion_history = []
        self.last_motion_time = 0
        self.motion_cooldown = 2.0  # seconds between motion events
        
        # Callbacks
        self.motion_callback = None
        self.motion_start_callback = None
        self.motion_end_callback = None
        
        # State tracking
        self.is_motion_active = False
        self.motion_start_time = 0
        self.background_model = None
        
        self.logger.info("Motion detector initialized")
    
    def set_motion_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Set callback for motion detection"""
        self.motion_callback = callback
    
    def set_motion_start_callback(self, callback: Callable[[], None]):
        """Set callback for motion start"""
        self.motion_start_callback = callback
    
    def set_motion_end_callback(self, callback: Callable[[float], None]):
        """Set callback for motion end"""
        self.motion_end_callback = callback
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """Detect motion in frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(blurred, learningRate=self.learning_rate)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate motion score
            motion_area = 0
            motion_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_motion_area:
                    motion_area += area
                    motion_contours.append(contour)
            
            # Normalize by frame area
            frame_area = frame.shape[0] * frame.shape[1]
            motion_score = motion_area / frame_area
            
            # Update motion history
            current_time = time.time()
            self.motion_history.append({
                'timestamp': current_time,
                'score': motion_score,
                'area': motion_area
            })
            
            # Keep only recent history
            if len(self.motion_history) > self.history_length:
                self.motion_history.pop(0)
            
            # Determine if motion is detected
            motion_detected = motion_score > self.motion_threshold
            
            # Handle motion events
            self._handle_motion_events(motion_detected, motion_score, current_time)
            
            # Create motion visualization
            motion_visualization = self._create_motion_visualization(
                frame, fg_mask, motion_contours, motion_score
            )
            
            return motion_detected, motion_score, motion_visualization
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
            return False, 0.0, None
    
    def _handle_motion_events(self, motion_detected: bool, motion_score: float, current_time: float):
        """Handle motion start/end events"""
        # Check cooldown
        if current_time - self.last_motion_time < self.motion_cooldown:
            return
        
        if motion_detected and not self.is_motion_active:
            # Motion started
            self.is_motion_active = True
            self.motion_start_time = current_time
            self.last_motion_time = current_time
            
            self.logger.info(f"Motion detected: score={motion_score:.3f}")
            
            if self.motion_start_callback:
                try:
                    self.motion_start_callback()
                except Exception as e:
                    self.logger.error(f"Error in motion start callback: {e}")
        
        elif not motion_detected and self.is_motion_active:
            # Motion ended
            motion_duration = current_time - self.motion_start_time
            self.is_motion_active = False
            
            self.logger.info(f"Motion ended: duration={motion_duration:.2f}s")
            
            if self.motion_end_callback:
                try:
                    self.motion_end_callback(motion_duration)
                except Exception as e:
                    self.logger.error(f"Error in motion end callback: {e}")
    
    def _create_motion_visualization(self, 
                                   frame: np.ndarray, 
                                   fg_mask: np.ndarray, 
                                   contours: List[np.ndarray], 
                                   motion_score: float) -> np.ndarray:
        """Create visualization of motion detection"""
        visualization = frame.copy()
        
        # Draw contours
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        # Add motion score text
        text = f"Motion: {motion_score:.3f}"
        cv2.putText(visualization, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add motion status
        status = "ACTIVE" if self.is_motion_active else "INACTIVE"
        status_color = (0, 255, 0) if self.is_motion_active else (0, 0, 255)
        cv2.putText(visualization, status, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        return visualization
    
    def get_motion_statistics(self) -> Dict[str, Any]:
        """Get motion detection statistics"""
        if not self.motion_history:
            return {
                'average_score': 0.0,
                'max_score': 0.0,
                'motion_events': 0,
                'total_duration': 0.0
            }
        
        scores = [entry['score'] for entry in self.motion_history]
        
        return {
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'motion_events': len([s for s in scores if s > self.motion_threshold]),
            'total_duration': len(self.motion_history) * 0.1,  # Assuming 10 FPS
            'is_active': self.is_motion_active
        }
    
    def reset_background_model(self):
        """Reset background subtraction model"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        self.logger.info("Background model reset")
    
    def update_learning_rate(self, learning_rate: float):
        """Update background subtraction learning rate"""
        self.learning_rate = max(0.0, min(1.0, learning_rate))
        self.logger.info(f"Learning rate updated to {self.learning_rate}")
    
    def set_motion_threshold(self, threshold: float):
        """Set motion detection threshold"""
        self.motion_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Motion threshold updated to {self.motion_threshold}")
    
    def set_min_motion_area(self, area: int):
        """Set minimum motion area"""
        self.min_motion_area = max(0, area)
        self.logger.info(f"Minimum motion area updated to {self.min_motion_area}")
    
    def get_motion_history(self) -> List[Dict[str, Any]]:
        """Get motion history"""
        return self.motion_history.copy()
    
    def clear_motion_history(self):
        """Clear motion history"""
        self.motion_history.clear()
        self.logger.info("Motion history cleared")

class AdaptiveMotionDetector(MotionDetector):
    """Adaptive motion detector that adjusts parameters based on environment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Adaptive parameters
        self.adaptive_threshold = True
        self.threshold_adjustment_rate = 0.1
        self.min_threshold = 0.05
        self.max_threshold = 0.5
        
        # Environment analysis
        self.light_level_history = []
        self.noise_level_history = []
        
        self.logger.info("Adaptive motion detector initialized")
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """Detect motion with adaptive parameters"""
        # Analyze frame characteristics
        light_level = self._analyze_light_level(frame)
        noise_level = self._analyze_noise_level(frame)
        
        # Update environment history
        self.light_level_history.append(light_level)
        self.noise_level_history.append(noise_level)
        
        # Keep history manageable
        if len(self.light_level_history) > 100:
            self.light_level_history.pop(0)
            self.noise_level_history.pop(0)
        
        # Adapt parameters based on environment
        if self.adaptive_threshold and len(self.light_level_history) > 10:
            self._adapt_parameters()
        
        # Call parent method
        return super().detect_motion(frame)
    
    def _analyze_light_level(self, frame: np.ndarray) -> float:
        """Analyze light level in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0
    
    def _analyze_noise_level(self, frame: np.ndarray) -> float:
        """Analyze noise level in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian filter to detect edges (noise)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)
        
        # Normalize
        return min(1.0, noise_level / 1000.0)
    
    def _adapt_parameters(self):
        """Adapt detection parameters based on environment"""
        avg_light = np.mean(self.light_level_history[-10:])
        avg_noise = np.mean(self.noise_level_history[-10:])
        
        # Adjust threshold based on light and noise levels
        target_threshold = self.motion_threshold
        
        # Lower threshold in low light conditions
        if avg_light < 0.3:
            target_threshold *= 0.8
        
        # Higher threshold in high noise conditions
        if avg_noise > 0.7:
            target_threshold *= 1.2
        
        # Apply adjustment
        new_threshold = self.motion_threshold + self.threshold_adjustment_rate * (target_threshold - self.motion_threshold)
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        if abs(new_threshold - self.motion_threshold) > 0.01:
            self.set_motion_threshold(new_threshold)
            self.logger.debug(f"Adapted threshold to {new_threshold:.3f} (light: {avg_light:.2f}, noise: {avg_noise:.2f})")

def create_motion_detector(config_path: str = "config.json", adaptive: bool = True) -> MotionDetector:
    """Factory function to create motion detector"""
    config = load_config(config_path)
    
    if adaptive:
        return AdaptiveMotionDetector(config)
    else:
        return MotionDetector(config)

if __name__ == "__main__":
    # Test motion detector
    import cv2
    
    config = load_config()
    motion_detector = create_motion_detector(config)
    
    def motion_callback(frame, score):
        print(f"Motion detected: {score:.3f}")
    
    def motion_start_callback():
        print("Motion started")
    
    def motion_end_callback(duration):
        print(f"Motion ended after {duration:.2f} seconds")
    
    motion_detector.set_motion_callback(motion_callback)
    motion_detector.set_motion_start_callback(motion_start_callback)
    motion_detector.set_motion_end_callback(motion_end_callback)
    
    # Test with video capture
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            motion_detected, score, visualization = motion_detector.detect_motion(frame)
            
            if visualization is not None:
                cv2.imshow('Motion Detection', visualization)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import logging
from collections import deque
import json

class MotionDetector:
    """Motion detection for wildlife monitoring to trigger detection"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        
        # Motion detection parameters
        self.motion_threshold = self.config.get('motion_threshold', 25)
        self.min_area = self.config.get('min_motion_area', 500)
        self.blur_size = self.config.get('blur_size', 21)
        self.dilate_iterations = self.config.get('dilate_iterations', 2)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        # Motion history
        self.motion_history = deque(maxlen=30)  # Last 30 frames
        self.motion_detected = False
        self.last_motion_time = 0
        self.motion_cooldown = 2.0  # seconds
        
        # Previous frame for frame differencing
        self.prev_frame = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_frames = 0
        self.motion_frames = 0
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def detect_motion_background_subtraction(self, frame):
        """Detect motion using background subtraction"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Remove shadows
            fg_mask[fg_mask == 127] = 0
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=self.dilate_iterations)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            motion_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > self.min_area:
                    motion_contours.append(contour)
            
            return motion_contours, fg_mask
            
        except Exception as e:
            self.logger.error(f"Error in background subtraction: {e}")
            return [], None
    
    def detect_motion_frame_difference(self, frame):
        """Detect motion using frame differencing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            
            # If no previous frame, store current and return
            if self.prev_frame is None:
                self.prev_frame = gray
                return [], None
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            
            # Threshold the difference
            _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate to fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.dilate(thresh, kernel, iterations=self.dilate_iterations)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            motion_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > self.min_area:
                    motion_contours.append(contour)
            
            # Update previous frame
            self.prev_frame = gray
            
            return motion_contours, thresh
            
        except Exception as e:
            self.logger.error(f"Error in frame differencing: {e}")
            return [], None
    
    def detect_motion_optical_flow(self, frame):
        """Detect motion using optical flow (Lucas-Kanade)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return [], None
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Detect corners in previous frame
            corners = cv2.goodFeaturesToTrack(
                self.prev_frame,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            if corners is not None:
                # Calculate optical flow
                new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, corners, None, **lk_params
                )
                
                # Select good points
                good_new = new_corners[status == 1]
                good_old = corners[status == 1]
                
                # Calculate motion vectors
                motion_vectors = good_new - good_old
                motion_magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
                
                # Check for significant motion
                significant_motion = motion_magnitudes > 5.0
                
                if np.any(significant_motion):
                    # Create a simple motion contour (bounding box of motion points)
                    motion_points = good_new[significant_motion]
                    if len(motion_points) > 0:
                        x, y, w, h = cv2.boundingRect(motion_points.astype(np.int32))
                        motion_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
                        self.prev_frame = gray
                        return [motion_contour], None
            
            self.prev_frame = gray
            return [], None
            
        except Exception as e:
            self.logger.error(f"Error in optical flow: {e}")
            return [], None
    
    def process_frame(self, frame, method='background_subtraction'):
        """Process frame for motion detection"""
        self.total_frames += 1
        current_time = time.time()
        
        try:
            # Choose detection method
            if method == 'background_subtraction':
                contours, mask = self.detect_motion_background_subtraction(frame)
            elif method == 'frame_difference':
                contours, mask = self.detect_motion_frame_difference(frame)
            elif method == 'optical_flow':
                contours, mask = self.detect_motion_optical_flow(frame)
            else:
                self.logger.error(f"Unknown motion detection method: {method}")
                return False, [], None
            
            # Check if motion is detected
            motion_detected = len(contours) > 0
            
            if motion_detected:
                self.motion_frames += 1
                self.last_motion_time = current_time
                
                # Calculate motion intensity
                total_area = sum(cv2.contourArea(contour) for contour in contours)
                motion_intensity = min(total_area / (frame.shape[0] * frame.shape[1]), 1.0)
                
                # Add to motion history
                self.motion_history.append({
                    'timestamp': current_time,
                    'intensity': motion_intensity,
                    'contour_count': len(contours),
                    'total_area': total_area
                })
                
                self.logger.debug(f"Motion detected: {len(contours)} regions, intensity: {motion_intensity:.3f}")
            
            # Update motion state (with cooldown)
            if motion_detected:
                self.motion_detected = True
            elif current_time - self.last_motion_time > self.motion_cooldown:
                self.motion_detected = False
            
            return motion_detected, contours, mask
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return False, [], None
    
    def is_motion_active(self):
        """Check if motion is currently active"""
        return self.motion_detected
    
    def get_motion_intensity(self):
        """Get current motion intensity"""
        if not self.motion_history:
            return 0.0
        
        # Average intensity over recent frames
        recent_frames = [entry for entry in self.motion_history 
                        if time.time() - entry['timestamp'] < 5.0]
        
        if not recent_frames:
            return 0.0
        
        return np.mean([entry['intensity'] for entry in recent_frames])
    
    def get_motion_statistics(self):
        """Get motion detection statistics"""
        motion_percentage = (self.motion_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'motion_frames': self.motion_frames,
            'motion_percentage': motion_percentage,
            'current_motion_detected': self.motion_detected,
            'last_motion_time': self.last_motion_time,
            'motion_history_length': len(self.motion_history),
            'current_intensity': self.get_motion_intensity()
        }
    
    def draw_motion_overlay(self, frame, contours, mask=None):
        """Draw motion detection overlay on frame"""
        try:
            # Draw contours
            if contours:
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                
                # Draw bounding rectangles
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add area text
                    area = cv2.contourArea(contour)
                    cv2.putText(frame, f"Area: {int(area)}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add motion status text
            status_text = "MOTION DETECTED" if self.motion_detected else "NO MOTION"
            color = (0, 255, 0) if self.motion_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add motion intensity
            intensity = self.get_motion_intensity()
            cv2.putText(frame, f"Intensity: {intensity:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Overlay motion mask if available
            if mask is not None:
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
                frame = cv2.addWeighted(frame, 0.8, mask_colored, 0.2, 0)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing motion overlay: {e}")
            return frame
    
    def reset(self):
        """Reset motion detector state"""
        self.motion_history.clear()
        self.motion_detected = False
        self.last_motion_time = 0
        self.prev_frame = None
        self.total_frames = 0
        self.motion_frames = 0
        
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        self.logger.info("Motion detector reset")

# Example usage and testing
if __name__ == "__main__":
    # Create motion detector
    detector = MotionDetector()
    
    # Test with camera (if available)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No camera available, creating test video")
        # Create test frames with motion
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add moving object every 10 frames
            if i % 10 == 0:
                cv2.circle(frame, (100 + i*5, 200), 30, (255, 255, 255), -1)
            
            motion_detected, contours, mask = detector.process_frame(frame)
            
            if motion_detected:
                print(f"Frame {i}: Motion detected with {len(contours)} regions")
    
    else:
        print("Testing with camera. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect motion
            motion_detected, contours, mask = detector.process_frame(frame)
            
            # Draw overlay
            frame_with_overlay = detector.draw_motion_overlay(frame, contours, mask)
            
            # Display
            cv2.imshow('Motion Detection', frame_with_overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Print statistics
    stats = detector.get_motion_statistics()
    print(f"Motion Detection Statistics: {stats}")
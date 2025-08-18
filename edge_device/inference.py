import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Dict, Any, Tuple, Optional
import logging
from utils import load_config

class WildlifeDetector:
    """Wildlife detection using ONNX model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_config = config.get('detection', {})
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.model_path = self.detection_config.get('model_path', '../model/wildlife_detection_model.onnx')
        self.confidence_threshold = self.detection_config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.detection_config.get('nms_threshold', 0.4)
        self.classes = self.detection_config.get('classes', ['deer', 'bear', 'coyote', 'fox', 'raccoon', 'human'])
        self.max_detections = self.detection_config.get('max_detections', 10)
        
        # Model input/output settings
        self.input_size = (640, 640)  # Standard YOLO input size
        self.input_name = 'images'
        self.output_names = ['output0']  # YOLO output name
        
        # Initialize ONNX runtime session
        self.session = None
        self._load_model()
        
        self.logger.info(f"Wildlife detector initialized with {len(self.classes)} classes")
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            # Create ONNX runtime session
            providers = ['CPUExecutionProvider']
            
            # Try to use GPU if available
            try:
                ort.get_device('GPU')
                providers.insert(0, 'CUDAExecutionProvider')
                self.logger.info("GPU acceleration enabled")
            except:
                self.logger.info("Using CPU for inference")
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model input/output info
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            self.logger.info(f"Model loaded successfully: {self.model_path}")
            self.logger.info(f"Input shape: {self.input_shape}")
            self.logger.info(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for model inference"""
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format (batch, channels, height, width)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Calculate scale factors for coordinate transformation
        scale_x = original_width / self.input_size[0]
        scale_y = original_height / self.input_size[1]
        
        return input_tensor, (scale_x, scale_y), (original_width, original_height)
    
    def postprocess_detections(self, 
                              outputs: np.ndarray, 
                              scale_factors: Tuple[float, float],
                              original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess model outputs to get detections"""
        detections = []
        
        # YOLO output format: [batch, num_detections, 6] where 6 = [x, y, w, h, confidence, class_id]
        # For newer YOLO versions, output might be different
        if len(outputs.shape) == 3:
            # Standard YOLO format
            predictions = outputs[0]  # Remove batch dimension
        else:
            # Handle different output formats
            predictions = outputs
        
        # Filter detections by confidence
        for detection in predictions:
            if len(detection) >= 6:
                x, y, w, h, confidence, class_id = detection[:6]
                
                if confidence >= self.confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    scale_x, scale_y = scale_factors
                    
                    # Convert center coordinates to corner coordinates
                    x1 = (x - w/2) * scale_x
                    y1 = (y - h/2) * scale_y
                    x2 = (x + w/2) * scale_x
                    y2 = (y + h/2) * scale_y
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, original_size[0]))
                    y1 = max(0, min(y1, original_size[1]))
                    x2 = max(0, min(x2, original_size[0]))
                    y2 = max(0, min(y2, original_size[1]))
                    
                    # Get class name
                    class_id = int(class_id)
                    if 0 <= class_id < len(self.classes):
                        class_name = self.classes[class_id]
                    else:
                        class_name = f"unknown_{class_id}"
                    
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    detections.append(detection_info)
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections)
        
        # Limit number of detections
        detections = detections[:self.max_detections]
        
        return detections
    
    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Convert to numpy arrays for easier processing
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        # Convert indices to list
        if len(indices) > 0:
            indices = indices.flatten().tolist()
        else:
            indices = []
        
        # Return filtered detections
        return [detections[i] for i in indices]
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Perform wildlife detection on image"""
        try:
            start_time = time.time()
            
            # Preprocess image
            input_tensor, scale_factors, original_size = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Postprocess detections
            detections = self.postprocess_detections(outputs[0], scale_factors, original_size)
            
            inference_time = time.time() - start_time
            
            self.logger.debug(f"Detection completed in {inference_time:.3f}s, found {len(detections)} objects")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes on image"""
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Convert bbox to integer coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = self._get_class_color(detection['class_id'])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
            (128, 128, 0),  # Olive
        ]
        
        return colors[class_id % len(colors)]
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of detections"""
        if not detections:
            return {
                'total_detections': 0,
                'species_count': {},
                'highest_confidence': 0.0,
                'average_confidence': 0.0
            }
        
        # Count by species
        species_count = {}
        confidences = []
        
        for detection in detections:
            species = detection['class_name']
            confidence = detection['confidence']
            
            species_count[species] = species_count.get(species, 0) + 1
            confidences.append(confidence)
        
        return {
            'total_detections': len(detections),
            'species_count': species_count,
            'highest_confidence': max(confidences),
            'average_confidence': sum(confidences) / len(confidences)
        }

def create_detector(config_path: str = "config.json") -> WildlifeDetector:
    """Factory function to create wildlife detector"""
    config = load_config(config_path)
    return WildlifeDetector(config)

if __name__ == "__main__":
    # Test detector
    import cv2
    
    config = load_config()
    detector = WildlifeDetector(config)
    
    # Load test image
    test_image = cv2.imread("test_image.jpg")
    if test_image is not None:
        detections = detector.detect(test_image)
        
        print(f"Found {len(detections)} detections:")
        for detection in detections:
            print(f"  {detection['class_name']}: {detection['confidence']:.2f}")
        
        # Draw detections
        result_image = detector.draw_detections(test_image, detections)
        cv2.imwrite("result_image.jpg", result_image)
        
        # Print summary
        summary = detector.get_detection_summary(detections)
        print(f"Summary: {summary}")
    else:
        print("Test image not found")
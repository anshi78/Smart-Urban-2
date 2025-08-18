import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import json
import logging
import time
from pathlib import Path

class WildlifeInferenceEngine:
    """High-performance inference engine for wildlife detection models"""
    
    def __init__(self, model_path, config_path="config.json"):
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = self.config.get('detection_settings', {}).get('classes', [])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.inference_times = []
        self.total_detections = 0
        
        # Load model
        self.load_model()
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}")
            return {}
    
    def load_model(self):
        """Load the trained wildlife detection model"""
        try:
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load PyTorch model
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
                self.logger.info(f"PyTorch model loaded on {self.device}")
                
            # Load ONNX model (if available)
            elif self.model_path.endswith('.onnx'):
                try:
                    import onnxruntime as ort
                    self.model = ort.InferenceSession(self.model_path)
                    self.logger.info("ONNX model loaded")
                except ImportError:
                    self.logger.error("ONNX runtime not available")
                    return False
            
            else:
                self.logger.error(f"Unsupported model format: {self.model_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Apply transforms
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            return None
    
    def postprocess_predictions(self, predictions, original_shape):
        """Postprocess model predictions to extract detections"""
        try:
            detections = []
            
            # Handle different model output formats
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # Assuming YOLO-style output: [batch, detections, (x, y, w, h, conf, class_scores...)]
            if len(predictions.shape) == 3:
                batch_predictions = predictions[0]  # Take first batch
                
                for detection in batch_predictions:
                    if len(detection) >= 5:
                        x, y, w, h, conf = detection[:5]
                        
                        # Skip low confidence detections
                        if conf < self.config.get('detection_settings', {}).get('confidence_threshold', 0.5):
                            continue
                        
                        # Get class prediction
                        if len(detection) > 5:
                            class_scores = detection[5:]
                            class_id = np.argmax(class_scores)
                            class_conf = class_scores[class_id]
                        else:
                            class_id = 0
                            class_conf = conf
                        
                        # Scale coordinates to original image size
                        orig_h, orig_w = original_shape[:2]
                        x = int(x * orig_w / 640)
                        y = int(y * orig_h / 640)
                        w = int(w * orig_w / 640)
                        h = int(h * orig_h / 640)
                        
                        # Create detection object
                        if class_id < len(self.classes):
                            species = self.classes[class_id]
                        else:
                            species = f"class_{class_id}"
                        
                        detection_obj = {
                            'species': species,
                            'confidence': float(conf),
                            'class_confidence': float(class_conf),
                            'location_x': max(0, x - w//2),
                            'location_y': max(0, y - h//2),
                            'width': w,
                            'height': h,
                            'center_x': x,
                            'center_y': y
                        }
                        
                        detections.append(detection_obj)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            return []
    
    def run_inference(self, image):
        """Run inference on a single image"""
        start_time = time.time()
        
        try:
            # Store original shape
            original_shape = image.shape
            
            # Preprocess
            input_tensor = self.preprocess_image(image)
            if input_tensor is None:
                return []
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    predictions = self.model(input_tensor)
                else:
                    # For TorchScript models
                    predictions = self.model(input_tensor)
            
            # Postprocess
            detections = self.postprocess_predictions(predictions, original_shape)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(detections)
            
            # Log performance periodically
            if len(self.inference_times) % 100 == 0:
                avg_time = np.mean(self.inference_times[-100:])
                self.logger.info(f"Average inference time (last 100): {avg_time:.3f}s")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in inference: {e}")
            return []
    
    def run_batch_inference(self, images):
        """Run inference on a batch of images"""
        try:
            batch_detections = []
            
            # Process images in batches for better GPU utilization
            batch_size = 4
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                
                # Preprocess batch
                batch_tensors = []
                original_shapes = []
                
                for img in batch:
                    tensor = self.preprocess_image(img)
                    if tensor is not None:
                        batch_tensors.append(tensor.squeeze(0))  # Remove batch dim
                        original_shapes.append(img.shape)
                
                if not batch_tensors:
                    continue
                
                # Stack tensors
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    predictions = self.model(batch_tensor)
                
                # Postprocess each prediction
                for j, pred in enumerate(predictions):
                    if j < len(original_shapes):
                        detections = self.postprocess_predictions(
                            pred.unsqueeze(0), original_shapes[j]
                        )
                        batch_detections.append(detections)
            
            return batch_detections
            
        except Exception as e:
            self.logger.error(f"Error in batch inference: {e}")
            return []
    
    def benchmark_model(self, test_image, num_runs=100):
        """Benchmark model performance"""
        self.logger.info(f"Running benchmark with {num_runs} iterations...")
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            detections = self.run_inference(test_image)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if i % 10 == 0:
                self.logger.info(f"Completed {i+1}/{num_runs} runs")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        results = {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'fps': fps,
            'total_runs': num_runs
        }
        
        self.logger.info(f"Benchmark Results:")
        self.logger.info(f"  Average time: {avg_time:.4f}s")
        self.logger.info(f"  Min time: {min_time:.4f}s")
        self.logger.info(f"  Max time: {max_time:.4f}s")
        self.logger.info(f"  Std deviation: {std_time:.4f}s")
        self.logger.info(f"  Average FPS: {fps:.2f}")
        
        return results
    
    def optimize_model(self):
        """Optimize model for faster inference"""
        try:
            if isinstance(self.model, torch.nn.Module):
                # Convert to TorchScript for optimization
                self.logger.info("Converting model to TorchScript...")
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                scripted_model = torch.jit.trace(self.model, dummy_input)
                
                # Optimize for inference
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                self.model = scripted_model
                
                self.logger.info("Model optimized for inference")
                return True
                
        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            return False
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return None
        
        return {
            'total_inferences': len(self.inference_times),
            'total_detections': self.total_detections,
            'average_inference_time': np.mean(self.inference_times),
            'average_fps': 1.0 / np.mean(self.inference_times),
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
            'detections_per_inference': self.total_detections / len(self.inference_times)
        }
    
    def save_optimized_model(self, output_path):
        """Save optimized model"""
        try:
            if hasattr(self.model, 'save'):
                self.model.save(output_path)
            else:
                torch.jit.save(self.model, output_path)
            
            self.logger.info(f"Optimized model saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Create inference engine
    model_path = "models/wildlife_detection.pt"  # Replace with actual model path
    engine = WildlifeInferenceEngine(model_path)
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run single inference
    detections = engine.run_inference(test_image)
    print(f"Detected {len(detections)} objects")
    
    # Benchmark performance
    if engine.model is not None:
        benchmark_results = engine.benchmark_model(test_image, num_runs=50)
        
        # Get performance stats
        stats = engine.get_performance_stats()
        if stats:
            print(f"Performance Stats: {stats}")
    
    else:
        print("Model not loaded, skipping benchmark")
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file_path', './logs/edge_device.log')
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_timestamp() -> str:
    """Create timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_image(image: np.ndarray, filepath: str, quality: int = 85) -> bool:
    """Save image with specified quality"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save image
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True
    except Exception as e:
        logging.error(f"Failed to save image {filepath}: {e}")
        return False

def load_image(filepath: str) -> Optional[np.ndarray]:
    """Load image from file"""
    try:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Could not load image from {filepath}")
        return image
    except Exception as e:
        logging.error(f"Failed to load image {filepath}: {e}")
        return None

def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize image to target size while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def calculate_file_size_mb(filepath: str) -> float:
    """Calculate file size in megabytes"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def cleanup_old_files(directory: str, max_age_days: int) -> int:
    """Remove files older than specified days"""
    try:
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
        
        return removed_count
    except Exception as e:
        logging.error(f"Failed to cleanup old files: {e}")
        return 0

def get_storage_usage_mb(directory: str) -> float:
    """Calculate total storage usage in megabytes"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)
    except Exception as e:
        logging.error(f"Failed to calculate storage usage: {e}")
        return 0.0

def format_detection_data(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Format detection data for API submission"""
    return {
        'device_id': detection.get('device_id'),
        'species': detection.get('species'),
        'confidence': detection.get('confidence'),
        'bbox': detection.get('bbox'),
        'metadata': {
            'timestamp': detection.get('timestamp'),
            'image_path': detection.get('image_path'),
            'detection_id': detection.get('detection_id')
        }
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure"""
    required_sections = ['device', 'camera', 'detection', 'storage', 'cloud']
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate device configuration
    device = config.get('device', {})
    if not device.get('device_id'):
        logging.error("Device ID is required")
        return False
    
    # Validate camera configuration
    camera = config.get('camera', {})
    if not camera.get('resolution'):
        logging.error("Camera resolution is required")
        return False
    
    # Validate detection configuration
    detection = config.get('detection', {})
    if not detection.get('model_path'):
        logging.error("Model path is required")
        return False
    
    return True

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories"""
    directories = [
        config.get('storage', {}).get('local_storage_path', './detections'),
        os.path.dirname(config.get('logging', {}).get('file_path', './logs/edge_device.log')),
        os.path.dirname(config.get('detection', {}).get('model_path', '../model/wildlife_detection_model.onnx'))
    ]
    
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - psutil.boot_time()
        }
    except ImportError:
        logging.warning("psutil not available, system info not collected")
        return {}

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"
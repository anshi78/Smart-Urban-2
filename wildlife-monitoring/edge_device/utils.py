import cv2
import numpy as np
import json
import os
import time
import logging
import psutil
import socket
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import platform

class SystemUtils:
    """Utility functions for system monitoring and management"""
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        try:
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'hostname': socket.gethostname(),
                'python_version': platform.python_version(),
                'opencv_version': cv2.__version__ if hasattr(cv2, '__version__') else 'unknown'
            }
        except Exception as e:
            logging.error(f"Error getting system info: {e}")
            return {}
    
    @staticmethod
    def get_system_resources():
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            temperature = entries[0].current if entries else None
                            break
            except:
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'memory_total_gb': memory_total_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'disk_total_gb': disk_total_gb,
                'temperature_celsius': temperature,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error getting system resources: {e}")
            return {}
    
    @staticmethod
    def check_camera_available(camera_index=0):
        """Check if camera is available"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception as e:
            logging.error(f"Error checking camera: {e}")
            return False
    
    @staticmethod
    def get_network_interfaces():
        """Get network interface information"""
        try:
            interfaces = []
            for interface, addrs in psutil.net_if_addrs().items():
                interface_info = {'name': interface, 'addresses': []}
                
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        interface_info['addresses'].append({
                            'type': 'IPv4',
                            'address': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast
                        })
                    elif addr.family == socket.AF_INET6:  # IPv6
                        interface_info['addresses'].append({
                            'type': 'IPv6',
                            'address': addr.address,
                            'netmask': addr.netmask
                        })
                
                # Get interface statistics
                stats = psutil.net_if_stats().get(interface)
                if stats:
                    interface_info['is_up'] = stats.isup
                    interface_info['speed'] = stats.speed
                
                interfaces.append(interface_info)
            
            return interfaces
            
        except Exception as e:
            logging.error(f"Error getting network interfaces: {e}")
            return []

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def resize_image(image, target_size, maintain_aspect_ratio=True):
        """Resize image to target size"""
        try:
            if maintain_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calculate scaling factor
                scale = min(target_w / w, target_h / h)
                
                # Calculate new dimensions
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_w, new_h))
                
                # Create padded image if needed
                if new_w != target_w or new_h != target_h:
                    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                    y_offset = (target_h - new_h) // 2
                    x_offset = (target_w - new_w) // 2
                    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    return padded
                
                return resized
            else:
                return cv2.resize(image, target_size)
                
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def enhance_image(image, brightness=0, contrast=1.0, saturation=1.0):
        """Enhance image with brightness, contrast, and saturation adjustments"""
        try:
            # Brightness and contrast
            enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            
            # Saturation adjustment
            if saturation != 1.0:
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error enhancing image: {e}")
            return image
    
    @staticmethod
    def calculate_image_quality_score(image):
        """Calculate image quality score based on sharpness and noise"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate noise level using standard deviation of high-frequency components
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = cv2.filter2D(gray, cv2.CV_32F, kernel)
            noise_level = np.std(high_freq)
            
            # Combine metrics (higher is better for sharpness, lower is better for noise)
            quality_score = laplacian_var / (1 + noise_level)
            
            return {
                'quality_score': float(quality_score),
                'sharpness': float(laplacian_var),
                'noise_level': float(noise_level)
            }
            
        except Exception as e:
            logging.error(f"Error calculating image quality: {e}")
            return {'quality_score': 0.0, 'sharpness': 0.0, 'noise_level': float('inf')}
    
    @staticmethod
    def detect_blur(image, threshold=100):
        """Detect if image is blurry"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return {
                'is_blurry': laplacian_var < threshold,
                'blur_score': float(laplacian_var),
                'threshold': threshold
            }
            
        except Exception as e:
            logging.error(f"Error detecting blur: {e}")
            return {'is_blurry': True, 'blur_score': 0.0, 'threshold': threshold}
    
    @staticmethod
    def create_thumbnail(image, size=(150, 150)):
        """Create thumbnail of image"""
        try:
            return ImageUtils.resize_image(image, size, maintain_aspect_ratio=True)
        except Exception as e:
            logging.error(f"Error creating thumbnail: {e}")
            return image

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_directory(directory_path):
        """Ensure directory exists, create if it doesn't"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logging.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def get_file_hash(file_path, algorithm='md5'):
        """Calculate file hash"""
        try:
            hash_algo = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_algo.update(chunk)
            
            return hash_algo.hexdigest()
            
        except Exception as e:
            logging.error(f"Error calculating file hash: {e}")
            return None
    
    @staticmethod
    def clean_old_files(directory, max_age_days=7, file_pattern="*"):
        """Clean old files from directory"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return 0
            
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            deleted_count = 0
            
            for file_path in directory.glob(file_pattern):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logging.error(f"Error deleting file {file_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logging.error(f"Error cleaning old files: {e}")
            return 0
    
    @staticmethod
    def get_directory_size(directory):
        """Get total size of directory in bytes"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        pass  # Skip files that can't be accessed
            return total_size
        except Exception as e:
            logging.error(f"Error getting directory size: {e}")
            return 0

class ConfigUtils:
    """Utility functions for configuration management"""
    
    @staticmethod
    def load_config(config_path, default_config=None):
        """Load configuration with fallback to defaults"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults if provided
                if default_config:
                    merged_config = default_config.copy()
                    ConfigUtils._deep_merge(merged_config, config)
                    return merged_config
                
                return config
            else:
                logging.warning(f"Config file {config_path} not found")
                return default_config or {}
                
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config or {}
    
    @staticmethod
    def save_config(config, config_path):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            FileUtils.ensure_directory(os.path.dirname(config_path))
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, sort_keys=True)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def _deep_merge(base_dict, update_dict):
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigUtils._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @staticmethod
    def validate_config(config, required_fields):
        """Validate configuration has required fields"""
        missing_fields = []
        
        for field_path in required_fields:
            if not ConfigUtils._has_nested_key(config, field_path):
                missing_fields.append(field_path)
        
        return {
            'valid': len(missing_fields) == 0,
            'missing_fields': missing_fields
        }
    
    @staticmethod
    def _has_nested_key(config, key_path):
        """Check if nested key exists in config"""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        
        return True

class LoggingUtils:
    """Utility functions for logging setup and management"""
    
    @staticmethod
    def setup_logging(log_level='INFO', log_file=None, max_size_mb=10, backup_count=5):
        """Setup logging configuration"""
        try:
            import logging.handlers
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Setup root logger
            logger = logging.getLogger()
            logger.setLevel(getattr(logging, log_level.upper()))
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (if specified)
            if log_file:
                FileUtils.ensure_directory(os.path.dirname(log_file))
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_size_mb * 1024 * 1024,
                    backupCount=backup_count
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            return True
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            return False

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name):
        """End timing an operation and record the duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(duration)
            del self.start_times[name]
            
            return duration
        return None
    
    def get_stats(self, name):
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'total': sum(values)
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()

# Example usage and testing
if __name__ == "__main__":
    # Test system utilities
    print("System Info:", SystemUtils.get_system_info())
    print("System Resources:", SystemUtils.get_system_resources())
    print("Camera Available:", SystemUtils.check_camera_available())
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    monitor.start_timer('test_operation')
    time.sleep(0.1)
    duration = monitor.end_timer('test_operation')
    
    print(f"Test operation took: {duration:.3f}s")
    print("Performance stats:", monitor.get_stats('test_operation'))
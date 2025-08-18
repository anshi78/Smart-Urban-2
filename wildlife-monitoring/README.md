# Wildlife Monitoring System

A comprehensive wildlife monitoring system using edge devices with AI-powered detection capabilities and cloud-based data management.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Edge Device   │───▶│  Cloud Backend   │───▶│   Dashboard     │
│                 │    │                  │    │                 │
│ • Camera Stream │    │ • Flask API      │    │ • Web Interface │
│ • AI Detection │    │ • SQLite DB      │    │ • Real-time Data│
│ • Motion Detect │    │ • File Storage   │    │ • Charts & Stats│
│ • Offline Store │    │ • Device Mgmt    │    │ • Device Status │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

### Edge Device Capabilities
- **Real-time Wildlife Detection**: AI-powered species identification using YOLO models
- **Motion Detection**: Multiple algorithms (background subtraction, frame differencing, optical flow)
- **Camera Management**: High-performance video streaming with configurable settings
- **Offline Storage**: Local data storage when cloud connection is unavailable
- **Cloud Synchronization**: Automatic sync of detections and media files
- **System Monitoring**: Resource usage, performance metrics, and health monitoring

### Cloud Backend
- **RESTful API**: Complete API for detection data, device management, and file uploads
- **Database Management**: SQLite database with comprehensive schema for detections, devices, and logs
- **File Storage**: Secure storage and management of images and videos
- **Device Registration**: Automatic device discovery and heartbeat monitoring
- **Statistics & Analytics**: Real-time statistics and detection analytics

### Web Dashboard
- **Real-time Monitoring**: Live view of detections, device status, and system health
- **Interactive Charts**: Species distribution, detection timelines, and trends
- **Device Management**: Monitor device status, locations, and connectivity
- **Alert System**: Notifications for important events and system issues
- **Responsive Design**: Mobile-friendly interface with modern UI

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV-compatible camera or video source
- Network connectivity (for cloud features)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd wildlife-monitoring
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system:**
   ```bash
   # Edit edge device configuration
   nano edge_device/config.json
   
   # Update backend URL, device settings, and detection parameters
   ```

4. **Start the cloud backend:**
   ```bash
   cd cloud_backend/api
   python app.py
   ```

5. **Run the edge device:**
   ```bash
   cd edge_device
   python detect.py
   ```

6. **Access the dashboard:**
   Open http://localhost:5000 in your web browser

## Configuration

### Edge Device Configuration (`edge_device/config.json`)

```json
{
    "device_id": "CAM001",
    "device_name": "Forest Camera 1",
    "location": "North Forest Trail",
    
    "camera_settings": {
        "camera_index": 0,
        "resolution": [1280, 720],
        "fps": 30
    },
    
    "detection_settings": {
        "model_path": "models/wildlife_detection.pt",
        "confidence_threshold": 0.5,
        "classes": ["elephant", "tiger", "leopard", "bear", "deer"]
    },
    
    "network_settings": {
        "backend_url": "http://localhost:5000",
        "upload_enabled": true,
        "heartbeat_interval": 30
    }
}
```

### Backend Configuration
The Flask backend automatically configures the database and API endpoints. Key settings:
- Database: SQLite (`detections.db`)
- Upload directory: `uploads/`
- API endpoints: `/api/detections`, `/api/devices`, `/api/stats`

## API Documentation

### Detections API

**GET /api/detections**
- Returns list of wildlife detections
- Query parameters: `limit`, `species`, `device_id`, `start_date`, `end_date`

**POST /api/detections**
- Add new detection
- Body: `{species, confidence, location_x, location_y, device_id, ...}`

### Devices API

**GET /api/devices**
- Returns list of registered devices

**POST /api/devices/{device_id}/heartbeat**
- Update device heartbeat and status

### Statistics API

**GET /api/stats**
- Returns detection statistics and analytics

### File Upload API

**POST /api/upload**
- Upload detection images/videos
- Multipart form data with `file` field

## Usage Examples

### Running Edge Detection

```python
from edge_device.detect import WildlifeDetector

# Initialize detector
detector = WildlifeDetector("config.json")

# Run continuous detection
detector.run_continuous_detection()
```

### Manual Detection on Image

```python
import cv2
from edge_device.detect import WildlifeDetector

# Load detector
detector = WildlifeDetector()

# Load and process image
image = cv2.imread("wildlife_image.jpg")
detections = detector.detect_objects(image)

# Display results
for detection in detections:
    print(f"Detected {detection['species']} with {detection['confidence']:.2f} confidence")
```

### Accessing Cloud API

```python
import requests

# Get recent detections
response = requests.get("http://localhost:5000/api/detections")
detections = response.json()

# Add new detection
detection_data = {
    "species": "elephant",
    "confidence": 0.95,
    "device_id": "CAM001"
}
response = requests.post("http://localhost:5000/api/detections", json=detection_data)
```

## System Components

### Edge Device Components

1. **camera_stream.py**: Camera management and video streaming
2. **detect.py**: AI-powered wildlife detection using YOLO
3. **motion.py**: Motion detection algorithms
4. **inference.py**: High-performance model inference engine
5. **offline_storage.py**: Local data storage and management
6. **sync_cloud.py**: Cloud synchronization and data upload
7. **utils.py**: System utilities and helper functions
8. **config.json**: Device configuration file

### Cloud Backend Components

1. **app.py**: Main Flask application and API endpoints
2. **database.py**: Database operations and management
3. **models.py**: Data models and validation
4. **schemas.py**: Database schema definitions

### Dashboard Components

1. **dashboard.html**: Web interface with real-time monitoring
2. **JavaScript**: Chart.js for data visualization
3. **CSS**: Modern responsive design with glassmorphism effects

## Advanced Features

### Offline Operation
- Automatic offline storage when cloud connection is unavailable
- Intelligent sync when connection is restored
- Local database for detection history
- Storage management with automatic cleanup

### Performance Optimization
- Multi-threaded processing for camera and detection
- Configurable detection intervals to balance accuracy and performance
- Model optimization with TorchScript and ONNX support
- Efficient image processing and compression

### System Monitoring
- Real-time system resource monitoring
- Performance metrics and benchmarking
- Device health monitoring with alerts
- Network connectivity monitoring

### Scalability
- Support for multiple edge devices
- Centralized device management
- Load balancing for high-traffic scenarios
- Database optimization for large datasets

## Troubleshooting

### Common Issues

1. **Camera not detected:**
   ```bash
   # Check camera index
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

2. **Model not found:**
   ```bash
   # Download YOLO model
   wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
   mv yolov5s.pt edge_device/models/
   ```

3. **Connection issues:**
   ```bash
   # Test backend connectivity
   curl http://localhost:5000/api/devices
   ```

4. **Permission errors:**
   ```bash
   # Fix camera permissions on Linux
   sudo usermod -a -G video $USER
   ```

### Performance Tuning

1. **Reduce detection frequency** for better performance
2. **Lower camera resolution** if processing is slow
3. **Adjust confidence threshold** to filter detections
4. **Use GPU acceleration** if available (CUDA)

## Development

### Setting up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

## Acknowledgments

- OpenCV for computer vision capabilities
- PyTorch and Ultralytics for YOLO model support
- Flask for the web framework
- Chart.js for data visualization
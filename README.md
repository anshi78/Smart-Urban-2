# Wildlife Monitoring System

A comprehensive wildlife monitoring system with edge device detection and cloud backend management. The system uses computer vision and machine learning to detect and track wildlife in real-time.

## Project Structure

```
wildlife-monitoring/
├── cloud_backend/           # Cloud API and dashboard
│   ├── api/                # Flask API server
│   │   ├── app.py          # Main API application
│   │   ├── models.py       # Database models
│   │   ├── database.py     # Database initialization
│   │   ├── schemas.py      # Data validation schemas
│   │   └── requirements.txt # API dependencies
│   ├── dashboard/          # Web dashboard (future)
│   └── uploads/            # Uploaded images and database
├── edge_device/            # Edge device components
│   ├── detect.py           # Main detection script
│   ├── camera_stream.py    # Camera management
│   ├── motion.py           # Motion detection
│   ├── inference.py        # Wildlife detection model
│   ├── offline_storage.py  # Local storage management
│   ├── sync_cloud.py       # Cloud synchronization
│   ├── utils.py            # Utility functions
│   ├── config.json         # Device configuration
│   └── requirements.txt    # Edge device dependencies
├── captured_frames/        # Captured video frames
├── data/                   # Training and test data
├── datasets/               # Dataset management
├── deployment/             # Deployment configurations
├── model/                  # ML models
└── README.md              # This file
```

## Features

### Edge Device
- **Real-time Camera Stream**: Continuous video capture with configurable settings
- **Motion Detection**: Background subtraction-based motion detection
- **Wildlife Detection**: ONNX-based object detection for wildlife species
- **Offline Storage**: Local SQLite database for detection storage
- **Cloud Sync**: Automatic synchronization with cloud backend
- **Power Management**: Configurable power modes and sleep settings

### Cloud Backend
- **RESTful API**: Flask-based API for data management
- **Database**: SQLite database with SQLAlchemy ORM
- **Device Management**: Device registration and heartbeat monitoring
- **Detection Storage**: Centralized detection data storage
- **Statistics**: Detection analytics and reporting
- **Image Management**: Detection image storage and retrieval

## Supported Wildlife Species

The system is configured to detect the following species:
- Deer
- Bear
- Coyote
- Fox
- Raccoon
- Human (for safety monitoring)

## Installation

### Prerequisites

- Python 3.8+
- OpenCV with camera support
- ONNX Runtime
- Flask (for cloud backend)

### Edge Device Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd wildlife-monitoring/edge_device
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the device**:
   Edit `config.json` to set your device parameters:
   ```json
   {
     "device": {
       "device_id": "CAM001",
       "name": "Forest Trail Camera 1",
       "location": "North Forest Trail"
     },
     "camera": {
       "resolution": [1920, 1080],
       "fps": 30
     },
     "cloud": {
       "api_url": "http://your-cloud-server:5000/api"
     }
   }
   ```

4. **Download the detection model**:
   Place your ONNX wildlife detection model in `../model/wildlife_detection_model.onnx`

### Cloud Backend Setup

1. **Navigate to cloud backend**:
   ```bash
   cd ../cloud_backend/api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**:
   ```bash
   python -c "from database import init_db; from app import app; init_db(app)"
   ```

4. **Start the API server**:
   ```bash
   python app.py
   ```

## Usage

### Running the Edge Device

#### Interactive Mode
```bash
python detect.py --interactive
```
- Press 'q' to quit
- Press 's' to show status
- Press 'h' for help
- Press 'c' to force cloud sync

#### Daemon Mode
```bash
python detect.py --daemon
```
Runs continuously in the background until interrupted.

#### Custom Configuration
```bash
python detect.py --config my_config.json
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Get Detections
```bash
curl http://localhost:5000/api/detections
```

#### Get Devices
```bash
curl http://localhost:5000/api/devices
```

#### Get Statistics
```bash
curl http://localhost:5000/api/stats/detections
```

## Configuration

### Edge Device Configuration

The `config.json` file controls all aspects of the edge device:

```json
{
  "device": {
    "device_id": "CAM001",
    "name": "Forest Trail Camera 1",
    "location": "North Forest Trail",
    "latitude": 45.5231,
    "longitude": -122.6765
  },
  "camera": {
    "resolution": [1920, 1080],
    "fps": 30,
    "rotation": 0,
    "brightness": 50,
    "contrast": 50
  },
  "detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "model_path": "../model/wildlife_detection_model.onnx",
    "classes": ["deer", "bear", "coyote", "fox", "raccoon", "human"]
  },
  "storage": {
    "local_storage_path": "./detections",
    "max_local_storage_mb": 1024,
    "retention_days": 30
  },
  "cloud": {
    "api_url": "http://localhost:5000/api",
    "sync_interval_seconds": 300,
    "batch_size": 10
  }
}
```

### Cloud Backend Configuration

The cloud backend uses environment variables for configuration:

```bash
export FLASK_ENV=production
export DATABASE_URL=sqlite:///uploads/detections.db
export UPLOAD_FOLDER=../uploads
```

## Development

### Adding New Species

1. **Update the model**: Retrain your detection model to include new species
2. **Update configuration**: Add the new species to the `classes` list in `config.json`
3. **Update database**: Add species information to the cloud backend

### Custom Detection Models

The system supports ONNX format models. To use a custom model:

1. Convert your model to ONNX format
2. Update the `model_path` in `config.json`
3. Adjust input/output handling in `inference.py` if needed

### Extending the API

The cloud backend is built with Flask and SQLAlchemy. To add new endpoints:

1. Add new routes in `app.py`
2. Create corresponding database models in `models.py`
3. Add validation schemas in `schemas.py`

## Monitoring and Logging

### Log Files

- Edge device logs: `./logs/edge_device.log`
- Cloud API logs: Console output (configurable)

### System Status

The system provides comprehensive status monitoring:

```python
# Get edge device status
status = system.get_system_status()
print(f"Detections: {status['total_detections']}")
print(f"FPS: {status['current_fps']}")
print(f"Storage: {status['storage_stats']}")
```

### Cloud Sync Status

Monitor cloud synchronization:

```python
sync_status = cloud_sync.get_status()
print(f"Last sync: {sync_status['last_sync_time']}")
print(f"Synced: {sync_status['stats']['total_synced']}")
```

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Verify camera index in configuration
   - Test with `cv2.VideoCapture(0)`

2. **Model loading errors**:
   - Verify ONNX model path
   - Check model compatibility with ONNX Runtime
   - Ensure model input/output format matches expectations

3. **Cloud sync failures**:
   - Check network connectivity
   - Verify API URL configuration
   - Check cloud server status

4. **Storage issues**:
   - Verify disk space
   - Check file permissions
   - Review storage configuration

### Performance Optimization

1. **Reduce resolution** for better performance
2. **Lower FPS** to reduce CPU usage
3. **Adjust detection thresholds** for accuracy vs. speed
4. **Use GPU acceleration** if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration examples

## Roadmap

- [ ] Web dashboard for visualization
- [ ] Mobile app for monitoring
- [ ] Advanced analytics and reporting
- [ ] Multi-camera support
- [ ] Real-time alerts and notifications
- [ ] Integration with conservation databases
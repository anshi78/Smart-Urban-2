from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class Detection:
    """Wildlife detection data model"""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    species: str = ""
    confidence: float = 0.0
    location_x: Optional[int] = None
    location_y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_path: Optional[str] = None
    device_id: Optional[str] = None
    processed: bool = False
    notes: Optional[str] = None
    
    def to_dict(self):
        """Convert detection to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'species': self.species,
            'confidence': self.confidence,
            'location_x': self.location_x,
            'location_y': self.location_y,
            'width': self.width,
            'height': self.height,
            'image_path': self.image_path,
            'device_id': self.device_id,
            'processed': self.processed,
            'notes': self.notes
        }
    
    def to_json(self):
        """Convert detection to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class Device:
    """Edge device data model"""
    id: str
    name: str
    location: Optional[str] = None
    status: str = "active"
    last_heartbeat: Optional[datetime] = None
    ip_address: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    
    def to_dict(self):
        """Convert device to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'status': self.status,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'ip_address': self.ip_address,
            'model': self.model,
            'firmware_version': self.firmware_version
        }
    
    def to_json(self):
        """Convert device to JSON string"""
        return json.dumps(self.to_dict())
    
    def is_online(self, timeout_minutes=5):
        """Check if device is online based on last heartbeat"""
        if not self.last_heartbeat:
            return False
        
        time_diff = datetime.now() - self.last_heartbeat
        return time_diff.total_seconds() < (timeout_minutes * 60)

@dataclass
class Species:
    """Species data model"""
    id: Optional[int] = None
    name: str = ""
    scientific_name: Optional[str] = None
    conservation_status: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self):
        """Convert species to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'scientific_name': self.scientific_name,
            'conservation_status': self.conservation_status,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def to_json(self):
        """Convert species to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class DetectionLog:
    """Detection log data model"""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    level: str = "INFO"
    message: str = ""
    device_id: Optional[str] = None
    detection_id: Optional[int] = None
    
    def to_dict(self):
        """Convert log to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level,
            'message': self.message,
            'device_id': self.device_id,
            'detection_id': self.detection_id
        }
    
    def to_json(self):
        """Convert log to JSON string"""
        return json.dumps(self.to_dict())

class DetectionStats:
    """Statistics for wildlife detections"""
    
    def __init__(self, total_detections=0, species_stats=None, 
                 recent_detections=0, active_devices=0, confidence_stats=None):
        self.total_detections = total_detections
        self.species_stats = species_stats or []
        self.recent_detections = recent_detections
        self.active_devices = active_devices
        self.confidence_stats = confidence_stats or []
    
    def to_dict(self):
        """Convert stats to dictionary"""
        return {
            'total_detections': self.total_detections,
            'species_stats': self.species_stats,
            'recent_detections': self.recent_detections,
            'active_devices': self.active_devices,
            'confidence_stats': self.confidence_stats
        }
    
    def to_json(self):
        """Convert stats to JSON string"""
        return json.dumps(self.to_dict())

# Utility functions for data validation
def validate_detection_data(data):
    """Validate detection data"""
    required_fields = ['species', 'confidence', 'device_id']
    errors = []
    
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    if 'confidence' in data:
        try:
            confidence = float(data['confidence'])
            if not 0.0 <= confidence <= 1.0:
                errors.append("Confidence must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            errors.append("Confidence must be a valid number")
    
    return errors

def validate_device_data(data):
    """Validate device data"""
    required_fields = ['id', 'name']
    errors = []
    
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    return errors
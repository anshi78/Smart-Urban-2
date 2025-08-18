from datetime import datetime
from typing import Optional, List, Dict, Any

class DetectionSchema:
    """Schema for detection data"""
    def __init__(self, 
                 device_id: str,
                 species: str,
                 confidence: float,
                 bbox: Optional[Dict[str, float]] = None,
                 image_path: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.device_id = device_id
        self.species = species
        self.confidence = confidence
        self.bbox = bbox or {}
        self.image_path = image_path
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'device_id': self.device_id,
            'species': self.species,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'image_path': self.image_path,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            device_id=data.get('device_id'),
            species=data.get('species'),
            confidence=data.get('confidence'),
            bbox=data.get('bbox'),
            image_path=data.get('image_path'),
            metadata=data.get('metadata')
        )

class DeviceSchema:
    """Schema for device data"""
    def __init__(self,
                 device_id: str,
                 name: str,
                 location: str,
                 latitude: Optional[float] = None,
                 longitude: Optional[float] = None,
                 status: str = 'active'):
        self.device_id = device_id
        self.name = name
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.status = status
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'device_id': self.device_id,
            'name': self.name,
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            device_id=data.get('device_id'),
            name=data.get('name'),
            location=data.get('location'),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            status=data.get('status', 'active')
        )

class SpeciesSchema:
    """Schema for species data"""
    def __init__(self,
                 name: str,
                 scientific_name: Optional[str] = None,
                 category: Optional[str] = None,
                 description: Optional[str] = None,
                 conservation_status: Optional[str] = None):
        self.name = name
        self.scientific_name = scientific_name
        self.category = category
        self.description = description
        self.conservation_status = conservation_status
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'name': self.name,
            'scientific_name': self.scientific_name,
            'category': self.category,
            'description': self.description,
            'conservation_status': self.conservation_status,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            name=data.get('name'),
            scientific_name=data.get('scientific_name'),
            category=data.get('category'),
            description=data.get('description'),
            conservation_status=data.get('conservation_status')
        )

class DetectionResponseSchema:
    """Schema for detection response"""
    def __init__(self,
                 id: int,
                 device_id: str,
                 species: str,
                 confidence: float,
                 timestamp: datetime,
                 bbox: Optional[Dict[str, float]] = None,
                 image_path: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.device_id = device_id
        self.species = species
        self.confidence = confidence
        self.timestamp = timestamp
        self.bbox = bbox
        self.image_path = image_path
        self.metadata = metadata
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'device_id': self.device_id,
            'species': self.species,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'bbox': self.bbox,
            'image_path': self.image_path,
            'metadata': self.metadata
        }
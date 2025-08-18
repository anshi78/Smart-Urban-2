from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Device(db.Model):
    """Model for edge devices/cameras"""
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='active')  # active, inactive, maintenance
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with detections
    detections = db.relationship('Detection', backref='device', lazy=True)

class Detection(db.Model):
    """Model for wildlife detections"""
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('device.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    species = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox = db.Column(db.Text)  # JSON string of bounding box coordinates
    image_path = db.Column(db.String(500), nullable=True)
    metadata = db.Column(db.Text)  # JSON string for additional data
    
    def set_bbox(self, bbox_dict):
        """Set bounding box as JSON string"""
        self.bbox = json.dumps(bbox_dict)
    
    def get_bbox(self):
        """Get bounding box as dictionary"""
        return json.loads(self.bbox) if self.bbox else None
    
    def set_metadata(self, metadata_dict):
        """Set metadata as JSON string"""
        self.metadata = json.dumps(metadata_dict)
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        return json.loads(self.metadata) if self.metadata else None

class Species(db.Model):
    """Model for species information"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    scientific_name = db.Column(db.String(100), nullable=True)
    category = db.Column(db.String(50), nullable=True)  # mammal, bird, reptile, etc.
    description = db.Column(db.Text, nullable=True)
    conservation_status = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
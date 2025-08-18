from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io

from models import db, Device, Detection, Species
from database import init_db
from schemas import DetectionSchema, DeviceSchema, SpeciesSchema, DetectionResponseSchema

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../uploads/detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# Detection endpoints
@app.route('/api/detections', methods=['POST'])
def create_detection():
    """Create a new detection"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['device_id', 'species', 'confidence']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Find device
        device = Device.query.filter_by(device_id=data['device_id']).first()
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        
        # Create detection
        detection = Detection(
            device_id=device.id,
            species=data['species'],
            confidence=data['confidence']
        )
        
        # Handle bounding box
        if 'bbox' in data:
            detection.set_bbox(data['bbox'])
        
        # Handle metadata
        if 'metadata' in data:
            detection.set_metadata(data['metadata'])
        
        # Handle image upload
        if 'image' in data:
            # Save image data
            image_filename = f"detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{detection.id}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            detection.image_path = image_path
        
        db.session.add(detection)
        db.session.commit()
        
        return jsonify({
            'message': 'Detection created successfully',
            'detection_id': detection.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get all detections with optional filters"""
    try:
        # Query parameters
        device_id = request.args.get('device_id')
        species = request.args.get('species')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        
        # Build query
        query = Detection.query
        
        if device_id:
            device = Device.query.filter_by(device_id=device_id).first()
            if device:
                query = query.filter_by(device_id=device.id)
        
        if species:
            query = query.filter_by(species=species)
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                query = query.filter(Detection.timestamp >= start_dt)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format'}), 400
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                query = query.filter(Detection.timestamp <= end_dt)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format'}), 400
        
        # Order by timestamp descending and limit
        detections = query.order_by(Detection.timestamp.desc()).limit(limit).all()
        
        # Convert to response format
        detection_list = []
        for detection in detections:
            detection_data = DetectionResponseSchema(
                id=detection.id,
                device_id=detection.device.device_id,
                species=detection.species,
                confidence=detection.confidence,
                timestamp=detection.timestamp,
                bbox=detection.get_bbox(),
                image_path=detection.image_path,
                metadata=detection.get_metadata()
            ).to_dict()
            detection_list.append(detection_data)
        
        return jsonify({
            'detections': detection_list,
            'count': len(detection_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get a specific detection by ID"""
    try:
        detection = Detection.query.get_or_404(detection_id)
        
        detection_data = DetectionResponseSchema(
            id=detection.id,
            device_id=detection.device.device_id,
            species=detection.species,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            bbox=detection.get_bbox(),
            image_path=detection.image_path,
            metadata=detection.get_metadata()
        ).to_dict()
        
        return jsonify(detection_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections/<int:detection_id>/image', methods=['GET'])
def get_detection_image(detection_id):
    """Get detection image"""
    try:
        detection = Detection.query.get_or_404(detection_id)
        
        if not detection.image_path or not os.path.exists(detection.image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(detection.image_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Device endpoints
@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get all devices"""
    try:
        devices = Device.query.all()
        device_list = []
        
        for device in devices:
            device_data = {
                'id': device.id,
                'device_id': device.device_id,
                'name': device.name,
                'location': device.location,
                'latitude': device.latitude,
                'longitude': device.longitude,
                'status': device.status,
                'last_seen': device.last_seen.isoformat() if device.last_seen else None,
                'created_at': device.created_at.isoformat()
            }
            device_list.append(device_data)
        
        return jsonify({
            'devices': device_list,
            'count': len(device_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices', methods=['POST'])
def create_device():
    """Create a new device"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['device_id', 'name', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if device already exists
        existing_device = Device.query.filter_by(device_id=data['device_id']).first()
        if existing_device:
            return jsonify({'error': 'Device already exists'}), 409
        
        # Create device
        device = Device(
            device_id=data['device_id'],
            name=data['name'],
            location=data['location'],
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            status=data.get('status', 'active')
        )
        
        db.session.add(device)
        db.session.commit()
        
        return jsonify({
            'message': 'Device created successfully',
            'device_id': device.device_id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices/<device_id>/heartbeat', methods=['POST'])
def device_heartbeat(device_id):
    """Update device heartbeat"""
    try:
        device = Device.query.filter_by(device_id=device_id).first()
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        
        device.last_seen = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Heartbeat updated successfully',
            'timestamp': device.last_seen.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Species endpoints
@app.route('/api/species', methods=['GET'])
def get_species():
    """Get all species"""
    try:
        species_list = Species.query.all()
        species_data = []
        
        for species in species_list:
            species_info = {
                'id': species.id,
                'name': species.name,
                'scientific_name': species.scientific_name,
                'category': species.category,
                'description': species.description,
                'conservation_status': species.conservation_status,
                'created_at': species.created_at.isoformat()
            }
            species_data.append(species_info)
        
        return jsonify({
            'species': species_data,
            'count': len(species_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/species', methods=['POST'])
def create_species():
    """Create a new species"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'name' not in data:
            return jsonify({'error': 'Missing required field: name'}), 400
        
        # Check if species already exists
        existing_species = Species.query.filter_by(name=data['name']).first()
        if existing_species:
            return jsonify({'error': 'Species already exists'}), 409
        
        # Create species
        species = Species(
            name=data['name'],
            scientific_name=data.get('scientific_name'),
            category=data.get('category'),
            description=data.get('description'),
            conservation_status=data.get('conservation_status')
        )
        
        db.session.add(species)
        db.session.commit()
        
        return jsonify({
            'message': 'Species created successfully',
            'species_id': species.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Statistics endpoints
@app.route('/api/stats/detections', methods=['GET'])
def get_detection_stats():
    """Get detection statistics"""
    try:
        # Get date range from query parameters
        days = request.args.get('days', 7, type=int)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get detections in date range
        detections = Detection.query.filter(
            Detection.timestamp >= start_date
        ).all()
        
        # Calculate statistics
        total_detections = len(detections)
        species_counts = {}
        device_counts = {}
        
        for detection in detections:
            # Count by species
            species_counts[detection.species] = species_counts.get(detection.species, 0) + 1
            
            # Count by device
            device_counts[detection.device.device_id] = device_counts.get(detection.device.device_id, 0) + 1
        
        return jsonify({
            'total_detections': total_detections,
            'date_range_days': days,
            'species_counts': species_counts,
            'device_counts': device_counts,
            'start_date': start_date.isoformat(),
            'end_date': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
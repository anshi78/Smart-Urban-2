from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
import json
from datetime import datetime
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
DATABASE_PATH = '../detections.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            species TEXT NOT NULL,
            confidence REAL NOT NULL,
            location_x INTEGER,
            location_y INTEGER,
            image_path TEXT,
            device_id TEXT,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Create devices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            status TEXT DEFAULT 'active',
            last_heartbeat DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get all wildlife detections"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, species, confidence, location_x, location_y, 
                   image_path, device_id, processed
            FROM detections
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'id': row[0],
                'timestamp': row[1],
                'species': row[2],
                'confidence': row[3],
                'location_x': row[4],
                'location_y': row[5],
                'image_path': row[6],
                'device_id': row[7],
                'processed': row[8]
            })
        
        conn.close()
        return jsonify({'detections': detections})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections', methods=['POST'])
def add_detection():
    """Add a new wildlife detection"""
    try:
        data = request.get_json()
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (species, confidence, location_x, location_y, 
                                  image_path, device_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['species'], data['confidence'], data.get('location_x'), 
              data.get('location_y'), data.get('image_path'), data['device_id']))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({'id': detection_id, 'message': 'Detection added successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload detection image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({'filename': filename, 'path': file_path})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get all registered devices"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, location, status, last_heartbeat FROM devices')
        
        devices = []
        for row in cursor.fetchall():
            devices.append({
                'id': row[0],
                'name': row[1],
                'location': row[2],
                'status': row[3],
                'last_heartbeat': row[4]
            })
        
        conn.close()
        return jsonify({'devices': devices})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices/<device_id>/heartbeat', methods=['POST'])
def device_heartbeat(device_id):
    """Update device heartbeat"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO devices (id, name, location, status, last_heartbeat)
            VALUES (?, ?, ?, 'active', CURRENT_TIMESTAMP)
        ''', (device_id, request.json.get('name', device_id), 
              request.json.get('location', 'Unknown')))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Heartbeat updated'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        # Detections by species
        cursor.execute('''
            SELECT species, COUNT(*) as count 
            FROM detections 
            GROUP BY species 
            ORDER BY count DESC
        ''')
        species_stats = [{'species': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Recent detections (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM detections 
            WHERE datetime(timestamp) >= datetime('now', '-1 day')
        ''')
        recent_detections = cursor.fetchone()[0]
        
        # Active devices
        cursor.execute('''
            SELECT COUNT(*) FROM devices 
            WHERE datetime(last_heartbeat) >= datetime('now', '-5 minutes')
        ''')
        active_devices = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_detections': total_detections,
            'species_stats': species_stats,
            'recent_detections': recent_detections,
            'active_devices': active_devices
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize database
    init_database()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
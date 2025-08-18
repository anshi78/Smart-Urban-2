import sqlite3
import os
from datetime import datetime

class WildlifeDatabase:
    def __init__(self, db_path='detections.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
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
                width INTEGER,
                height INTEGER,
                image_path TEXT,
                device_id TEXT,
                processed BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
        ''')
        
        # Create devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'active',
                last_heartbeat DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                model TEXT,
                firmware_version TEXT
            )
        ''')
        
        # Create species table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                scientific_name TEXT,
                conservation_status TEXT,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create detection_logs table for system logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                device_id TEXT,
                detection_id INTEGER,
                FOREIGN KEY (detection_id) REFERENCES detections (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_detection(self, species, confidence, location_x=None, location_y=None, 
                     width=None, height=None, image_path=None, device_id=None, notes=None):
        """Add a new wildlife detection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (species, confidence, location_x, location_y, 
                                  width, height, image_path, device_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (species, confidence, location_x, location_y, width, height, 
              image_path, device_id, notes))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return detection_id
    
    def get_detections(self, limit=100, species=None, device_id=None, 
                      start_date=None, end_date=None):
        """Get wildlife detections with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, timestamp, species, confidence, location_x, location_y, 
                   width, height, image_path, device_id, processed, notes
            FROM detections
            WHERE 1=1
        '''
        params = []
        
        if species:
            query += ' AND species = ?'
            params.append(species)
        
        if device_id:
            query += ' AND device_id = ?'
            params.append(device_id)
        
        if start_date:
            query += ' AND datetime(timestamp) >= datetime(?)'
            params.append(start_date)
        
        if end_date:
            query += ' AND datetime(timestamp) <= datetime(?)'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'id': row[0],
                'timestamp': row[1],
                'species': row[2],
                'confidence': row[3],
                'location_x': row[4],
                'location_y': row[5],
                'width': row[6],
                'height': row[7],
                'image_path': row[8],
                'device_id': row[9],
                'processed': row[10],
                'notes': row[11]
            })
        
        conn.close()
        return detections
    
    def register_device(self, device_id, name, location=None, ip_address=None, 
                       model=None, firmware_version=None):
        """Register a new edge device"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO devices (id, name, location, ip_address, 
                                          model, firmware_version, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (device_id, name, location, ip_address, model, firmware_version))
        
        conn.commit()
        conn.close()
    
    def update_device_heartbeat(self, device_id):
        """Update device heartbeat timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE devices 
            SET last_heartbeat = CURRENT_TIMESTAMP, status = 'active'
            WHERE id = ?
        ''', (device_id,))
        
        conn.commit()
        conn.close()
    
    def get_devices(self):
        """Get all registered devices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, location, status, last_heartbeat, ip_address, 
                   model, firmware_version
            FROM devices
        ''')
        
        devices = []
        for row in cursor.fetchall():
            devices.append({
                'id': row[0],
                'name': row[1],
                'location': row[2],
                'status': row[3],
                'last_heartbeat': row[4],
                'ip_address': row[5],
                'model': row[6],
                'firmware_version': row[7]
            })
        
        conn.close()
        return devices
    
    def add_species(self, name, scientific_name=None, conservation_status=None, 
                   description=None):
        """Add a new species to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO species (name, scientific_name, conservation_status, description)
                VALUES (?, ?, ?, ?)
            ''', (name, scientific_name, conservation_status, description))
            
            species_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return species_id
        
        except sqlite3.IntegrityError:
            conn.close()
            return None  # Species already exists
    
    def get_species(self):
        """Get all species"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, scientific_name, conservation_status, description, created_at
            FROM species
            ORDER BY name
        ''')
        
        species = []
        for row in cursor.fetchall():
            species.append({
                'id': row[0],
                'name': row[1],
                'scientific_name': row[2],
                'conservation_status': row[3],
                'description': row[4],
                'created_at': row[5]
            })
        
        conn.close()
        return species
    
    def get_statistics(self):
        """Get detection statistics"""
        conn = sqlite3.connect(self.db_path)
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
        
        # Active devices (heartbeat within last 5 minutes)
        cursor.execute('''
            SELECT COUNT(*) FROM devices 
            WHERE datetime(last_heartbeat) >= datetime('now', '-5 minutes')
        ''')
        active_devices = cursor.fetchone()[0]
        
        # Average confidence by species
        cursor.execute('''
            SELECT species, AVG(confidence) as avg_confidence
            FROM detections 
            GROUP BY species
        ''')
        confidence_stats = [{'species': row[0], 'avg_confidence': row[1]} 
                           for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'species_stats': species_stats,
            'recent_detections': recent_detections,
            'active_devices': active_devices,
            'confidence_stats': confidence_stats
        }
    
    def log_message(self, level, message, device_id=None, detection_id=None):
        """Add a log message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_logs (level, message, device_id, detection_id)
            VALUES (?, ?, ?, ?)
        ''', (level, message, device_id, detection_id))
        
        conn.commit()
        conn.close()
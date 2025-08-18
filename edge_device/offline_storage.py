import sqlite3
import json
import os
import time
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from utils import load_config, create_timestamp, calculate_file_size_mb, get_storage_usage_mb

class OfflineStorage:
    """Offline storage management for edge device"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_config = config.get('storage', {})
        self.logger = logging.getLogger(__name__)
        
        # Storage settings
        self.storage_path = self.storage_config.get('local_storage_path', './detections')
        self.max_storage_mb = self.storage_config.get('max_local_storage_mb', 1024)
        self.compression_quality = self.storage_config.get('compression_quality', 85)
        self.retention_days = self.storage_config.get('retention_days', 30)
        
        # Database settings
        self.db_path = os.path.join(self.storage_path, 'detections.db')
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Offline storage initialized: {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT,
                    image_path TEXT,
                    metadata TEXT,
                    synced INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create images table for file management
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER,
                    filename TEXT NOT NULL,
                    file_size_mb REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (detection_id) REFERENCES detections (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections(synced)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_device ON detections(device_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_detection(self, 
                       device_id: str,
                       species: str,
                       confidence: float,
                       image: Optional[np.ndarray] = None,
                       bbox: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store detection in local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            bbox_json = json.dumps(bbox) if bbox else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert detection record
            cursor.execute('''
                INSERT INTO detections (timestamp, device_id, species, confidence, bbox, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, device_id, species, confidence, bbox_json, metadata_json))
            
            detection_id = cursor.lastrowid
            
            # Store image if provided
            image_path = None
            if image is not None:
                image_path = self._save_image(image, detection_id)
                
                # Update detection with image path
                cursor.execute('''
                    UPDATE detections SET image_path = ? WHERE id = ?
                ''', (image_path, detection_id))
                
                # Record image file info
                file_size = calculate_file_size_mb(image_path)
                cursor.execute('''
                    INSERT INTO images (detection_id, filename, file_size_mb)
                    VALUES (?, ?, ?)
                ''', (detection_id, os.path.basename(image_path), file_size))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Detection stored: ID={detection_id}, Species={species}, Confidence={confidence:.2f}")
            
            # Check storage limits
            self._check_storage_limits()
            
            return detection_id
            
        except Exception as e:
            self.logger.error(f"Failed to store detection: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise
    
    def _save_image(self, image: np.ndarray, detection_id: int) -> str:
        """Save detection image"""
        timestamp = create_timestamp()
        filename = f"detection_{detection_id}_{timestamp}.jpg"
        filepath = os.path.join(self.storage_path, 'images', filename)
        
        # Ensure images directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save image with compression
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality])
        
        return filepath
    
    def get_unsynced_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get detections that haven't been synced to cloud"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, device_id, species, confidence, bbox, image_path, metadata
                FROM detections
                WHERE synced = 0
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (limit,))
            
            detections = []
            for row in cursor.fetchall():
                detection = {
                    'id': row[0],
                    'timestamp': row[1],
                    'device_id': row[2],
                    'species': row[3],
                    'confidence': row[4],
                    'bbox': json.loads(row[5]) if row[5] else None,
                    'image_path': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                detections.append(detection)
            
            conn.close()
            return detections
            
        except Exception as e:
            self.logger.error(f"Failed to get unsynced detections: {e}")
            return []
    
    def mark_detection_synced(self, detection_id: int):
        """Mark detection as synced to cloud"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE detections SET synced = 1 WHERE id = ?
            ''', (detection_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Detection {detection_id} marked as synced")
            
        except Exception as e:
            self.logger.error(f"Failed to mark detection as synced: {e}")
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """Get detection by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, device_id, species, confidence, bbox, image_path, metadata, synced
                FROM detections
                WHERE id = ?
            ''', (detection_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'timestamp': row[1],
                    'device_id': row[2],
                    'species': row[3],
                    'confidence': row[4],
                    'bbox': json.loads(row[5]) if row[5] else None,
                    'image_path': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None,
                    'synced': bool(row[8])
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get detection: {e}")
            return None
    
    def get_detections_by_date_range(self, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detections within date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, timestamp, device_id, species, confidence, bbox, image_path, metadata, synced
                FROM detections
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if device_id:
                query += ' AND device_id = ?'
                params.append(device_id)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            
            detections = []
            for row in cursor.fetchall():
                detection = {
                    'id': row[0],
                    'timestamp': row[1],
                    'device_id': row[2],
                    'species': row[3],
                    'confidence': row[4],
                    'bbox': json.loads(row[5]) if row[5] else None,
                    'image_path': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None,
                    'synced': bool(row[8])
                }
                detections.append(detection)
            
            conn.close()
            return detections
            
        except Exception as e:
            self.logger.error(f"Failed to get detections by date range: {e}")
            return []
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total detections
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_detections = cursor.fetchone()[0]
            
            # Get unsynced detections
            cursor.execute('SELECT COUNT(*) FROM detections WHERE synced = 0')
            unsynced_detections = cursor.fetchone()[0]
            
            # Get total storage usage
            cursor.execute('SELECT SUM(file_size_mb) FROM images')
            total_storage_mb = cursor.fetchone()[0] or 0.0
            
            # Get species count
            cursor.execute('''
                SELECT species, COUNT(*) as count
                FROM detections
                GROUP BY species
                ORDER BY count DESC
            ''')
            species_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_detections': total_detections,
                'unsynced_detections': unsynced_detections,
                'total_storage_mb': total_storage_mb,
                'max_storage_mb': self.max_storage_mb,
                'storage_usage_percent': (total_storage_mb / self.max_storage_mb) * 100,
                'species_counts': species_counts
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}
    
    def _check_storage_limits(self):
        """Check and enforce storage limits"""
        try:
            current_usage = get_storage_usage_mb(self.storage_path)
            
            if current_usage > self.max_storage_mb:
                self.logger.warning(f"Storage limit exceeded: {current_usage:.1f}MB > {self.max_storage_mb}MB")
                self._cleanup_old_data()
                
        except Exception as e:
            self.logger.error(f"Failed to check storage limits: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to free storage space"""
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old detections
            cursor.execute('''
                SELECT id, image_path FROM detections
                WHERE timestamp < ? AND synced = 1
                ORDER BY timestamp ASC
            ''', (cutoff_date.isoformat(),))
            
            old_detections = cursor.fetchall()
            
            deleted_count = 0
            for detection_id, image_path in old_detections:
                # Delete image file
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete image {image_path}: {e}")
                
                # Delete detection record
                cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
                cursor.execute('DELETE FROM images WHERE detection_id = ?', (detection_id,))
                
                deleted_count += 1
                
                # Check if we've freed enough space
                current_usage = get_storage_usage_mb(self.storage_path)
                if current_usage < self.max_storage_mb * 0.8:  # Leave 20% buffer
                    break
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old detections")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.db_path)
                self.logger.info(f"Database restored from {backup_path}")
            else:
                self.logger.error(f"Backup file not found: {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
    
    def export_detections(self, export_path: str, format: str = 'json'):
        """Export detections to file"""
        try:
            detections = self.get_detections_by_date_range(
                datetime.utcnow() - timedelta(days=365),  # Last year
                datetime.utcnow()
            )
            
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(detections, f, indent=2)
            elif format.lower() == 'csv':
                import csv
                with open(export_path, 'w', newline='') as f:
                    if detections:
                        writer = csv.DictWriter(f, fieldnames=detections[0].keys())
                        writer.writeheader()
                        writer.writerows(detections)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(detections)} detections to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export detections: {e}")

def create_offline_storage(config_path: str = "config.json") -> OfflineStorage:
    """Factory function to create offline storage"""
    config = load_config(config_path)
    return OfflineStorage(config)

if __name__ == "__main__":
    # Test offline storage
    import cv2
    import numpy as np
    
    config = load_config()
    storage = create_offline_storage(config)
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Store test detection
    detection_id = storage.store_detection(
        device_id="CAM001",
        species="deer",
        confidence=0.85,
        image=test_image,
        bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 400},
        metadata={"test": True}
    )
    
    print(f"Stored detection with ID: {detection_id}")
    
    # Get statistics
    stats = storage.get_storage_statistics()
    print(f"Storage statistics: {stats}")
    
    # Get unsynced detections
    unsynced = storage.get_unsynced_detections()
    print(f"Unsynced detections: {len(unsynced)}")
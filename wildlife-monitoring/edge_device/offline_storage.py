import os
import json
import sqlite3
import shutil
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import cv2
import numpy as np

class OfflineStorage:
    """Manages offline storage of detections and media files when cloud connection is unavailable"""
    
    def __init__(self, config_path="config.json", storage_dir="offline_storage"):
        self.config = self.load_config(config_path)
        self.storage_dir = Path(storage_dir)
        self.db_path = self.storage_dir / "offline_detections.db"
        
        # Storage settings
        self.max_storage_gb = self.config.get('storage_settings', {}).get('max_storage_gb', 10)
        self.cleanup_days = self.config.get('storage_settings', {}).get('cleanup_older_than_days', 7)
        self.image_quality = self.config.get('storage_settings', {}).get('image_quality', 85)
        
        # Setup directories
        self.setup_directories()
        
        # Setup database
        self.init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Background processing
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Statistics
        self.stored_detections = 0
        self.stored_images = 0
        self.total_storage_used = 0
        
        # Start background processing
        self.start_background_processing()
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.storage_dir,
            self.storage_dir / "images",
            self.storage_dir / "videos", 
            self.storage_dir / "logs",
            self.storage_dir / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for offline storage"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS offline_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    location_x INTEGER,
                    location_y INTEGER,
                    width INTEGER,
                    height INTEGER,
                    image_path TEXT,
                    video_path TEXT,
                    device_id TEXT,
                    synced BOOLEAN DEFAULT FALSE,
                    sync_attempts INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER DEFAULT 0
                )
            ''')
            
            # Create sync log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    detection_id INTEGER,
                    FOREIGN KEY (detection_id) REFERENCES offline_detections (id)
                )
            ''')
            
            # Create storage stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS storage_stats (
                    id INTEGER PRIMARY KEY,
                    last_cleanup DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_files INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    detections_count INTEGER DEFAULT 0
                )
            ''')
            
            # Initialize stats if not exists
            cursor.execute('INSERT OR IGNORE INTO storage_stats (id) VALUES (1)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Offline storage database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def store_detection(self, detection_data, image=None, video_path=None):
        """Store detection data and associated media offline"""
        try:
            # Save image if provided
            image_path = None
            if image is not None:
                image_path = self.save_image(image, detection_data.get('device_id', 'unknown'))
            
            # Store detection in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            file_size = 0
            if image_path and os.path.exists(image_path):
                file_size += os.path.getsize(image_path)
            if video_path and os.path.exists(video_path):
                file_size += os.path.getsize(video_path)
            
            cursor.execute('''
                INSERT INTO offline_detections 
                (species, confidence, location_x, location_y, width, height, 
                 image_path, video_path, device_id, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_data.get('species'),
                detection_data.get('confidence'),
                detection_data.get('location_x'),
                detection_data.get('location_y'),
                detection_data.get('width'),
                detection_data.get('height'),
                str(image_path) if image_path else None,
                video_path,
                detection_data.get('device_id'),
                file_size
            ))
            
            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Update statistics
            self.stored_detections += 1
            if image_path:
                self.stored_images += 1
            self.total_storage_used += file_size
            
            # Add to processing queue for potential sync
            self.processing_queue.put({
                'type': 'sync_detection',
                'detection_id': detection_id
            })
            
            self.logger.info(f"Detection stored offline: ID {detection_id}")
            return detection_id
            
        except Exception as e:
            self.logger.error(f"Error storing detection: {e}")
            return None
    
    def save_image(self, image, device_id):
        """Save image to local storage with compression"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{device_id}_{timestamp}.jpg"
            image_path = self.storage_dir / "images" / filename
            
            # Compress and save image
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            cv2.imwrite(str(image_path), image, encode_params)
            
            self.logger.debug(f"Image saved: {image_path}")
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            return None
    
    def get_unsynced_detections(self, limit=50):
        """Get detections that haven't been synced to cloud"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, species, confidence, location_x, location_y,
                       width, height, image_path, video_path, device_id, sync_attempts
                FROM offline_detections
                WHERE synced = FALSE AND sync_attempts < 5
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (limit,))
            
            detections = []
            for row in cursor.fetchall():
                detection = {
                    'id': row[0],
                    'timestamp': row[1],
                    'species': row[2],
                    'confidence': row[3],
                    'location_x': row[4],
                    'location_y': row[5],
                    'width': row[6],
                    'height': row[7],
                    'image_path': row[8],
                    'video_path': row[9],
                    'device_id': row[10],
                    'sync_attempts': row[11]
                }
                detections.append(detection)
            
            conn.close()
            return detections
            
        except Exception as e:
            self.logger.error(f"Error getting unsynced detections: {e}")
            return []
    
    def mark_detection_synced(self, detection_id, success=True):
        """Mark detection as synced or update sync attempts"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if success:
                cursor.execute('''
                    UPDATE offline_detections 
                    SET synced = TRUE 
                    WHERE id = ?
                ''', (detection_id,))
                
                # Log successful sync
                cursor.execute('''
                    INSERT INTO sync_log (action, status, detection_id)
                    VALUES ('sync', 'success', ?)
                ''', (detection_id,))
                
                self.logger.info(f"Detection {detection_id} marked as synced")
                
            else:
                cursor.execute('''
                    UPDATE offline_detections 
                    SET sync_attempts = sync_attempts + 1 
                    WHERE id = ?
                ''', (detection_id,))
                
                # Log failed sync
                cursor.execute('''
                    INSERT INTO sync_log (action, status, detection_id)
                    VALUES ('sync', 'failed', ?)
                ''', (detection_id,))
                
                self.logger.warning(f"Sync attempt failed for detection {detection_id}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating sync status: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.cleanup_days)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get old detections with file paths
            cursor.execute('''
                SELECT id, image_path, video_path, file_size
                FROM offline_detections
                WHERE timestamp < ? AND synced = TRUE
            ''', (cutoff_date,))
            
            old_detections = cursor.fetchall()
            deleted_files = 0
            freed_space = 0
            
            for detection_id, image_path, video_path, file_size in old_detections:
                # Delete associated files
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_files += 1
                
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                    deleted_files += 1
                
                freed_space += file_size or 0
                
                # Delete database record
                cursor.execute('DELETE FROM offline_detections WHERE id = ?', (detection_id,))
            
            # Clean up old sync logs
            cursor.execute('DELETE FROM sync_log WHERE timestamp < ?', (cutoff_date,))
            
            # Update storage stats
            cursor.execute('''
                UPDATE storage_stats 
                SET last_cleanup = CURRENT_TIMESTAMP,
                    total_files = total_files - ?,
                    total_size_bytes = total_size_bytes - ?
                WHERE id = 1
            ''', (deleted_files, freed_space))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleanup completed: {len(old_detections)} detections, {deleted_files} files, {freed_space/1024/1024:.2f}MB freed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def check_storage_space(self):
        """Check available storage space and perform cleanup if needed"""
        try:
            # Get current storage usage
            total_size = 0
            for root, dirs, files in os.walk(self.storage_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            # Convert to GB
            total_size_gb = total_size / (1024 ** 3)
            
            self.logger.debug(f"Current storage usage: {total_size_gb:.2f}GB / {self.max_storage_gb}GB")
            
            # Check if cleanup is needed
            if total_size_gb > self.max_storage_gb * 0.9:  # 90% threshold
                self.logger.warning("Storage approaching limit, starting cleanup...")
                self.cleanup_old_data()
                
                # Force cleanup of unsynced old data if still over limit
                if total_size_gb > self.max_storage_gb:
                    self.force_cleanup()
            
            return total_size_gb
            
        except Exception as e:
            self.logger.error(f"Error checking storage space: {e}")
            return 0
    
    def force_cleanup(self):
        """Force cleanup of oldest data regardless of sync status"""
        try:
            self.logger.warning("Performing force cleanup due to storage limit")
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get oldest 20% of detections
            cursor.execute('SELECT COUNT(*) FROM offline_detections')
            total_count = cursor.fetchone()[0]
            cleanup_count = max(1, total_count // 5)  # 20% or at least 1
            
            cursor.execute('''
                SELECT id, image_path, video_path, file_size
                FROM offline_detections
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (cleanup_count,))
            
            old_detections = cursor.fetchall()
            
            for detection_id, image_path, video_path, file_size in old_detections:
                # Delete files
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                
                # Delete database record
                cursor.execute('DELETE FROM offline_detections WHERE id = ?', (detection_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.warning(f"Force cleanup completed: {len(old_detections)} detections removed")
            
        except Exception as e:
            self.logger.error(f"Error in force cleanup: {e}")
    
    def get_storage_statistics(self):
        """Get storage statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get detection counts
            cursor.execute('SELECT COUNT(*) FROM offline_detections')
            total_detections = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM offline_detections WHERE synced = TRUE')
            synced_detections = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM offline_detections WHERE synced = FALSE')
            unsynced_detections = cursor.fetchone()[0]
            
            # Get storage usage
            cursor.execute('SELECT SUM(file_size) FROM offline_detections')
            total_size_bytes = cursor.fetchone()[0] or 0
            
            # Get recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM offline_detections 
                WHERE datetime(timestamp) >= datetime('now', '-24 hours')
            ''')
            recent_detections = cursor.fetchone()[0]
            
            conn.close()
            
            # Get actual disk usage
            disk_usage = self.check_storage_space()
            
            return {
                'total_detections': total_detections,
                'synced_detections': synced_detections,
                'unsynced_detections': unsynced_detections,
                'recent_detections_24h': recent_detections,
                'total_size_bytes': total_size_bytes,
                'total_size_gb': total_size_bytes / (1024 ** 3),
                'disk_usage_gb': disk_usage,
                'max_storage_gb': self.max_storage_gb,
                'storage_utilization': disk_usage / self.max_storage_gb if self.max_storage_gb > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage statistics: {e}")
            return {}
    
    def start_background_processing(self):
        """Start background processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Background processing started")
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("Background processing stopped")
    
    def _background_worker(self):
        """Background worker for periodic tasks"""
        last_cleanup = time.time()
        last_stats_update = time.time()
        
        while self.running:
            try:
                # Process queue items
                try:
                    item = self.processing_queue.get(timeout=1)
                    if item['type'] == 'sync_detection':
                        # This would trigger sync attempt in a real implementation
                        pass
                except queue.Empty:
                    pass
                
                current_time = time.time()
                
                # Periodic cleanup (every hour)
                if current_time - last_cleanup > 3600:
                    self.check_storage_space()
                    last_cleanup = current_time
                
                # Update statistics (every 5 minutes)
                if current_time - last_stats_update > 300:
                    stats = self.get_storage_statistics()
                    self.logger.debug(f"Storage stats: {stats}")
                    last_stats_update = current_time
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in background worker: {e}")
                time.sleep(5)
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_background_processing()

# Example usage and testing
if __name__ == "__main__":
    # Create offline storage instance
    storage = OfflineStorage()
    
    # Test storing detections
    for i in range(5):
        detection_data = {
            'species': f'test_animal_{i}',
            'confidence': 0.8 + i * 0.05,
            'location_x': 100 + i * 10,
            'location_y': 200 + i * 10,
            'width': 150,
            'height': 100,
            'device_id': 'TEST_CAM_001'
        }
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detection_id = storage.store_detection(detection_data, test_image)
        print(f"Stored detection {detection_id}")
    
    # Get statistics
    stats = storage.get_storage_statistics()
    print(f"Storage Statistics: {stats}")
    
    # Get unsynced detections
    unsynced = storage.get_unsynced_detections()
    print(f"Unsynced detections: {len(unsynced)}")
    
    # Cleanup
    storage.stop_background_processing()
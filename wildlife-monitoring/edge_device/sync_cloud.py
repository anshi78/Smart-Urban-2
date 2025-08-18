import requests
import json
import time
import logging
import threading
from datetime import datetime
import os
from pathlib import Path
from offline_storage import OfflineStorage
import queue

class CloudSync:
    """Handles synchronization of offline data with cloud backend"""
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        
        # Cloud settings
        self.backend_url = self.config.get('network_settings', {}).get('backend_url', 'http://localhost:5000')
        self.device_id = self.config.get('device_id', 'CAM001')
        self.sync_enabled = self.config.get('network_settings', {}).get('upload_enabled', True)
        
        # Sync settings
        self.sync_interval = self.config.get('network_settings', {}).get('heartbeat_interval', 30)
        self.batch_size = 10  # Number of detections to sync at once
        self.max_retry_attempts = self.config.get('network_settings', {}).get('retry_attempts', 3)
        self.timeout = self.config.get('network_settings', {}).get('timeout_seconds', 30)
        
        # Initialize offline storage
        self.offline_storage = OfflineStorage(config_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sync state
        self.is_online = False
        self.last_sync_attempt = 0
        self.last_successful_sync = 0
        self.sync_errors = []
        
        # Background sync
        self.sync_thread = None
        self.sync_running = False
        self.sync_queue = queue.Queue()
        
        # Statistics
        self.sync_stats = {
            'total_synced': 0,
            'sync_failures': 0,
            'last_sync_time': None,
            'bytes_synced': 0,
            'connection_failures': 0
        }
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def test_connection(self):
        """Test connection to cloud backend"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/devices",
                timeout=5
            )
            
            if response.status_code == 200:
                self.is_online = True
                self.logger.debug("Cloud connection successful")
                return True
            else:
                self.is_online = False
                self.logger.warning(f"Cloud connection failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.is_online = False
            self.logger.warning(f"Cloud connection failed: {e}")
            self.sync_stats['connection_failures'] += 1
            return False
    
    def send_heartbeat(self):
        """Send heartbeat to cloud backend"""
        if not self.sync_enabled:
            return False
        
        try:
            data = {
                'name': self.config.get('device_name', self.device_id),
                'location': self.config.get('location', 'Unknown'),
                'model': self.config.get('model', 'Generic Camera'),
                'firmware_version': self.config.get('firmware_version', '1.0.0'),
                'status': 'online',
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{self.backend_url}/api/devices/{self.device_id}/heartbeat",
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.is_online = True
                self.logger.debug("Heartbeat sent successfully")
                return True
            else:
                self.logger.error(f"Heartbeat failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
            return False
    
    def upload_detection(self, detection_data):
        """Upload a single detection to cloud backend"""
        try:
            # Prepare detection data
            data = {
                'species': detection_data['species'],
                'confidence': detection_data['confidence'],
                'location_x': detection_data.get('location_x'),
                'location_y': detection_data.get('location_y'),
                'width': detection_data.get('width'),
                'height': detection_data.get('height'),
                'device_id': detection_data['device_id'],
                'timestamp': detection_data.get('timestamp', datetime.now().isoformat())
            }
            
            # Upload image if available
            image_filename = None
            if detection_data.get('image_path') and os.path.exists(detection_data['image_path']):
                image_filename = self.upload_image(detection_data['image_path'])
                if image_filename:
                    data['image_path'] = image_filename
            
            # Send detection data
            response = requests.post(
                f"{self.backend_url}/api/detections",
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Detection uploaded successfully: ID {result.get('id')}")
                self.sync_stats['total_synced'] += 1
                return True
            else:
                self.logger.error(f"Detection upload failed: HTTP {response.status_code}")
                self.sync_stats['sync_failures'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading detection: {e}")
            self.sync_stats['sync_failures'] += 1
            return False
    
    def upload_image(self, image_path):
        """Upload image file to cloud backend"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'device_id': self.device_id}
                
                response = requests.post(
                    f"{self.backend_url}/api/upload",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    filename = result.get('filename')
                    
                    # Update bytes synced
                    file_size = os.path.getsize(image_path)
                    self.sync_stats['bytes_synced'] += file_size
                    
                    self.logger.debug(f"Image uploaded: {filename}")
                    return filename
                else:
                    self.logger.error(f"Image upload failed: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error uploading image: {e}")
            return None
    
    def sync_pending_detections(self):
        """Sync all pending detections to cloud"""
        if not self.sync_enabled or not self.test_connection():
            return 0
        
        try:
            # Get unsynced detections
            unsynced_detections = self.offline_storage.get_unsynced_detections(self.batch_size)
            
            if not unsynced_detections:
                self.logger.debug("No pending detections to sync")
                return 0
            
            synced_count = 0
            
            for detection in unsynced_detections:
                success = self.upload_detection(detection)
                
                # Update sync status
                self.offline_storage.mark_detection_synced(detection['id'], success)
                
                if success:
                    synced_count += 1
                else:
                    # Stop syncing if we encounter failures
                    break
                
                # Small delay between uploads
                time.sleep(0.1)
            
            if synced_count > 0:
                self.last_successful_sync = time.time()
                self.sync_stats['last_sync_time'] = datetime.now().isoformat()
                self.logger.info(f"Synced {synced_count} detections to cloud")
            
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Error syncing detections: {e}")
            return 0
    
    def start_background_sync(self):
        """Start background sync thread"""
        if self.sync_thread and self.sync_thread.is_alive():
            return
        
        self.sync_running = True
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        self.logger.info("Background sync started")
    
    def stop_background_sync(self):
        """Stop background sync"""
        self.sync_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        
        self.logger.info("Background sync stopped")
    
    def _sync_worker(self):
        """Background worker for periodic sync"""
        last_heartbeat = 0
        last_sync = 0
        
        while self.sync_running:
            try:
                current_time = time.time()
                
                # Send heartbeat periodically
                if current_time - last_heartbeat >= self.sync_interval:
                    self.send_heartbeat()
                    last_heartbeat = current_time
                
                # Sync detections periodically
                if current_time - last_sync >= self.sync_interval * 2:  # Sync less frequently
                    if self.is_online:
                        synced_count = self.sync_pending_detections()
                        if synced_count > 0:
                            self.logger.info(f"Background sync completed: {synced_count} detections")
                    last_sync = current_time
                
                # Process sync queue
                try:
                    item = self.sync_queue.get(timeout=1)
                    if item['type'] == 'immediate_sync':
                        self.sync_pending_detections()
                except queue.Empty:
                    pass
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in sync worker: {e}")
                time.sleep(5)
    
    def force_sync(self):
        """Force immediate sync of pending detections"""
        self.sync_queue.put({'type': 'immediate_sync'})
    
    def get_sync_status(self):
        """Get current sync status"""
        # Get offline storage stats
        storage_stats = self.offline_storage.get_storage_statistics()
        
        return {
            'is_online': self.is_online,
            'sync_enabled': self.sync_enabled,
            'last_sync_attempt': self.last_sync_attempt,
            'last_successful_sync': self.last_successful_sync,
            'pending_detections': storage_stats.get('unsynced_detections', 0),
            'total_detections': storage_stats.get('total_detections', 0),
            'sync_stats': self.sync_stats.copy(),
            'backend_url': self.backend_url,
            'device_id': self.device_id
        }
    
    def retry_failed_syncs(self):
        """Retry previously failed sync attempts"""
        if not self.test_connection():
            return 0
        
        # Get detections with failed sync attempts
        failed_detections = self.offline_storage.get_unsynced_detections(self.batch_size)
        retry_count = 0
        
        for detection in failed_detections:
            if detection['sync_attempts'] > 0:  # Previously failed
                success = self.upload_detection(detection)
                self.offline_storage.mark_detection_synced(detection['id'], success)
                
                if success:
                    retry_count += 1
                    self.logger.info(f"Retry successful for detection {detection['id']}")
                
                time.sleep(0.1)
        
        return retry_count
    
    def clear_sync_errors(self):
        """Clear sync error log"""
        self.sync_errors.clear()
        self.logger.info("Sync errors cleared")
    
    def get_network_status(self):
        """Get detailed network status"""
        try:
            # Test connection with timing
            start_time = time.time()
            connection_ok = self.test_connection()
            response_time = time.time() - start_time
            
            return {
                'connected': connection_ok,
                'backend_url': self.backend_url,
                'response_time_ms': response_time * 1000,
                'last_test': datetime.now().isoformat(),
                'connection_failures': self.sync_stats['connection_failures']
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_test': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_background_sync()

# Example usage and testing
if __name__ == "__main__":
    # Create cloud sync instance
    sync = CloudSync()
    
    # Test connection
    if sync.test_connection():
        print("Cloud connection successful")
    else:
        print("Cloud connection failed")
    
    # Get sync status
    status = sync.get_sync_status()
    print(f"Sync Status: {status}")
    
    # Start background sync
    sync.start_background_sync()
    
    try:
        # Run for a while to test background sync
        print("Background sync running... Press Ctrl+C to stop")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("Stopping sync...")
    
    finally:
        sync.stop_background_sync()
        print("Cloud sync stopped")
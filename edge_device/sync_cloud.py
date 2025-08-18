import requests
import json
import time
import threading
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from utils import load_config, format_detection_data
from offline_storage import OfflineStorage

class CloudSync:
    """Cloud synchronization for edge device"""
    
    def __init__(self, config: Dict[str, Any], storage: OfflineStorage):
        self.config = config
        self.cloud_config = config.get('cloud', {})
        self.logger = logging.getLogger(__name__)
        self.storage = storage
        
        # Cloud settings
        self.api_url = self.cloud_config.get('api_url', 'http://localhost:5000/api')
        self.sync_interval = self.cloud_config.get('sync_interval_seconds', 300)
        self.batch_size = self.cloud_config.get('batch_size', 10)
        self.timeout = self.cloud_config.get('timeout_seconds', 30)
        self.retry_attempts = self.cloud_config.get('retry_attempts', 3)
        
        # Device info
        self.device_id = config.get('device', {}).get('device_id', 'CAM001')
        
        # Sync state
        self.is_syncing = False
        self.last_sync_time = None
        self.sync_thread = None
        self.stop_sync = False
        
        # Statistics
        self.sync_stats = {
            'total_synced': 0,
            'total_failed': 0,
            'last_sync_duration': 0,
            'last_sync_errors': []
        }
        
        self.logger.info(f"Cloud sync initialized: {self.api_url}")
    
    def start_sync(self):
        """Start automatic synchronization"""
        if self.sync_thread and self.sync_thread.is_alive():
            self.logger.warning("Sync thread already running")
            return
        
        self.stop_sync = False
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        self.logger.info("Cloud sync started")
    
    def stop_sync(self):
        """Stop automatic synchronization"""
        self.stop_sync = True
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        self.logger.info("Cloud sync stopped")
    
    def _sync_loop(self):
        """Main sync loop"""
        while not self.stop_sync:
            try:
                self._perform_sync()
                time.sleep(self.sync_interval)
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _perform_sync(self):
        """Perform synchronization with cloud"""
        if self.is_syncing:
            self.logger.debug("Sync already in progress, skipping")
            return
        
        self.is_syncing = True
        start_time = time.time()
        
        try:
            # Get unsynced detections
            unsynced_detections = self.storage.get_unsynced_detections(self.batch_size)
            
            if not unsynced_detections:
                self.logger.debug("No unsynced detections")
                return
            
            self.logger.info(f"Syncing {len(unsynced_detections)} detections")
            
            # Sync detections in batches
            successful_syncs = 0
            failed_syncs = 0
            errors = []
            
            for detection in unsynced_detections:
                try:
                    if self._sync_detection(detection):
                        self.storage.mark_detection_synced(detection['id'])
                        successful_syncs += 1
                    else:
                        failed_syncs += 1
                        errors.append(f"Failed to sync detection {detection['id']}")
                
                except Exception as e:
                    failed_syncs += 1
                    error_msg = f"Error syncing detection {detection['id']}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Update statistics
            sync_duration = time.time() - start_time
            self.sync_stats['total_synced'] += successful_syncs
            self.sync_stats['total_failed'] += failed_syncs
            self.sync_stats['last_sync_duration'] = sync_duration
            self.sync_stats['last_sync_errors'] = errors[:10]  # Keep last 10 errors
            
            self.last_sync_time = datetime.utcnow()
            
            self.logger.info(f"Sync completed: {successful_syncs} successful, {failed_syncs} failed, "
                           f"duration: {sync_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error during sync: {e}")
            self.sync_stats['last_sync_errors'].append(str(e))
        
        finally:
            self.is_syncing = False
    
    def _sync_detection(self, detection: Dict[str, Any]) -> bool:
        """Sync individual detection to cloud"""
        try:
            # Prepare detection data
            detection_data = {
                'device_id': detection['device_id'],
                'species': detection['species'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'metadata': detection['metadata']
            }
            
            # Add image if available
            if detection.get('image_path') and os.path.exists(detection['image_path']):
                with open(detection['image_path'], 'rb') as f:
                    files = {'image': f}
                    response = self._make_request(
                        'POST',
                        f'{self.api_url}/detections',
                        data={'detection': json.dumps(detection_data)},
                        files=files
                    )
            else:
                response = self._make_request(
                    'POST',
                    f'{self.api_url}/detections',
                    json=detection_data
                )
            
            if response and response.status_code in [200, 201]:
                self.logger.debug(f"Successfully synced detection {detection['id']}")
                return True
            else:
                self.logger.warning(f"Failed to sync detection {detection['id']}: "
                                  f"HTTP {response.status_code if response else 'No response'}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error syncing detection {detection['id']}: {e}")
            return False
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                kwargs.setdefault('timeout', self.timeout)
                response = requests.request(method, url, **kwargs)
                return response
            
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All request attempts failed for {url}")
                    return None
        
        return None
    
    def sync_now(self) -> Dict[str, Any]:
        """Perform immediate synchronization"""
        if self.is_syncing:
            return {'error': 'Sync already in progress'}
        
        start_time = time.time()
        
        try:
            self._perform_sync()
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
                'stats': self.sync_stats.copy()
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            'is_syncing': self.is_syncing,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'sync_interval': self.sync_interval,
            'api_url': self.api_url,
            'device_id': self.device_id,
            'stats': self.sync_stats.copy()
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update cloud configuration"""
        cloud_config = new_config.get('cloud', {})
        
        if 'api_url' in cloud_config:
            self.api_url = cloud_config['api_url']
        
        if 'sync_interval_seconds' in cloud_config:
            self.sync_interval = cloud_config['sync_interval_seconds']
        
        if 'batch_size' in cloud_config:
            self.batch_size = cloud_config['batch_size']
        
        if 'timeout_seconds' in cloud_config:
            self.timeout = cloud_config['timeout_seconds']
        
        if 'retry_attempts' in cloud_config:
            self.retry_attempts = cloud_config['retry_attempts']
        
        self.logger.info("Cloud sync configuration updated")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to cloud API"""
        try:
            # Test health endpoint
            health_url = f"{self.api_url.replace('/api', '')}/health"
            response = self._make_request('GET', health_url)
            
            if response and response.status_code == 200:
                return {
                    'success': True,
                    'status': 'Connected',
                    'response_time': response.elapsed.total_seconds(),
                    'api_version': response.json().get('version', 'Unknown')
                }
            else:
                return {
                    'success': False,
                    'status': f'HTTP {response.status_code if response else "No response"}',
                    'error': 'Health check failed'
                }
        
        except Exception as e:
            return {
                'success': False,
                'status': 'Connection failed',
                'error': str(e)
            }
    
    def register_device(self) -> bool:
        """Register device with cloud API"""
        try:
            device_data = {
                'device_id': self.device_id,
                'name': self.config.get('device', {}).get('name', 'Edge Device'),
                'location': self.config.get('device', {}).get('location', 'Unknown'),
                'latitude': self.config.get('device', {}).get('latitude'),
                'longitude': self.config.get('device', {}).get('longitude'),
                'status': 'active'
            }
            
            response = self._make_request(
                'POST',
                f'{self.api_url}/devices',
                json=device_data
            )
            
            if response and response.status_code in [200, 201]:
                self.logger.info("Device registered successfully")
                return True
            else:
                self.logger.warning(f"Device registration failed: HTTP {response.status_code if response else 'No response'}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error registering device: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to cloud API"""
        try:
            response = self._make_request(
                'POST',
                f'{self.api_url}/devices/{self.device_id}/heartbeat'
            )
            
            if response and response.status_code == 200:
                self.logger.debug("Heartbeat sent successfully")
                return True
            else:
                self.logger.warning(f"Heartbeat failed: HTTP {response.status_code if response else 'No response'}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
            return False

class CloudSyncManager:
    """Manager for multiple cloud sync instances"""
    
    def __init__(self, config: Dict[str, Any], storage: OfflineStorage):
        self.config = config
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Create sync instance
        self.sync = CloudSync(config, storage)
        
        # Auto-start sync if configured
        if config.get('cloud', {}).get('auto_start', True):
            self.sync.start_sync()
    
    def start(self):
        """Start cloud synchronization"""
        self.sync.start_sync()
    
    def stop(self):
        """Stop cloud synchronization"""
        self.sync.stop_sync()
    
    def sync_now(self) -> Dict[str, Any]:
        """Perform immediate synchronization"""
        return self.sync.sync_now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return self.sync.get_sync_status()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test cloud connection"""
        return self.sync.test_connection()
    
    def register_device(self) -> bool:
        """Register device with cloud"""
        return self.sync.register_device()

def create_cloud_sync(config_path: str = "config.json", storage: Optional[OfflineStorage] = None) -> CloudSyncManager:
    """Factory function to create cloud sync manager"""
    config = load_config(config_path)
    
    if storage is None:
        storage = OfflineStorage(config)
    
    return CloudSyncManager(config, storage)

if __name__ == "__main__":
    # Test cloud sync
    import os
    
    config = load_config()
    storage = OfflineStorage(config)
    cloud_sync = create_cloud_sync(config, storage)
    
    # Test connection
    connection_status = cloud_sync.test_connection()
    print(f"Connection status: {connection_status}")
    
    # Test device registration
    if connection_status['success']:
        registered = cloud_sync.register_device()
        print(f"Device registration: {'Success' if registered else 'Failed'}")
    
    # Test sync
    sync_result = cloud_sync.sync_now()
    print(f"Sync result: {sync_result}")
    
    # Get status
    status = cloud_sync.get_status()
    print(f"Sync status: {status}")
#!/usr/bin/env python3
"""
Main wildlife detection script for edge device
Orchestrates camera stream, motion detection, wildlife detection, and cloud sync
"""

import cv2
import numpy as np
import time
import signal
import sys
import threading
from typing import Dict, Any, Optional
import logging
import argparse

from utils import load_config, setup_logging, create_directories, validate_config
from camera_stream import CameraStream
from motion import create_motion_detector
from inference import WildlifeDetector
from offline_storage import OfflineStorage
from sync_cloud import CloudSyncManager

class WildlifeDetectionSystem:
    """Main wildlife detection system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        
        # Validate configuration
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Create necessary directories
        create_directories(self.config)
        
        # Initialize components
        self.camera = None
        self.motion_detector = None
        self.wildlife_detector = None
        self.storage = None
        self.cloud_sync = None
        
        # System state
        self.is_running = False
        self.detection_count = 0
        self.motion_count = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        self.logger.info("Wildlife detection system initialized")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize camera
            self.camera = CameraStream(self.config)
            
            # Initialize motion detector
            self.motion_detector = create_motion_detector(self.config)
            
            # Initialize wildlife detector
            self.wildlife_detector = WildlifeDetector(self.config)
            
            # Initialize storage
            self.storage = OfflineStorage(self.config)
            
            # Initialize cloud sync
            self.cloud_sync = CloudSyncManager(self.config, self.storage)
            
            # Set up callbacks
            self._setup_callbacks()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_callbacks(self):
        """Set up system callbacks"""
        # Camera frame callback
        self.camera.set_frame_callback(self._process_frame)
        
        # Motion detection callbacks
        self.motion_detector.set_motion_callback(self._on_motion_detected)
        self.motion_detector.set_motion_start_callback(self._on_motion_start)
        self.motion_detector.set_motion_end_callback(self._on_motion_end)
    
    def start(self):
        """Start the detection system"""
        if self.is_running:
            self.logger.warning("System already running")
            return
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Start camera stream
            if not self.camera.start_stream():
                raise RuntimeError("Failed to start camera stream")
            
            # Start cloud sync
            self.cloud_sync.start()
            
            # Register device with cloud
            self.cloud_sync.register_device()
            
            self.is_running = True
            self.start_time = time.time()
            
            self.logger.info("Wildlife detection system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the detection system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping wildlife detection system...")
        
        # Stop camera
        if self.camera:
            self.camera.stop_stream()
        
        # Stop cloud sync
        if self.cloud_sync:
            self.cloud_sync.stop()
        
        self.is_running = False
        self.logger.info("Wildlife detection system stopped")
    
    def _process_frame(self, frame: np.ndarray):
        """Process individual camera frame"""
        try:
            # Update FPS counter
            self._update_fps()
            
            # Detect motion
            motion_detected, motion_score, motion_viz = self.motion_detector.detect_motion(frame)
            
            if motion_detected:
                self.motion_count += 1
                
                # Perform wildlife detection
                detections = self.wildlife_detector.detect(frame)
                
                if detections:
                    self._handle_detections(frame, detections)
                
                # Log motion event
                self.logger.debug(f"Motion detected: score={motion_score:.3f}, "
                                f"detections={len(detections)}")
        
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def _handle_detections(self, frame: np.ndarray, detections: list):
        """Handle wildlife detections"""
        try:
            for detection in detections:
                self.detection_count += 1
                
                # Store detection locally
                detection_id = self.storage.store_detection(
                    device_id=self.config['device']['device_id'],
                    species=detection['class_name'],
                    confidence=detection['confidence'],
                    image=frame,
                    bbox=detection['bbox'],
                    metadata={
                        'detection_id': detection_id,
                        'motion_score': detection.get('motion_score', 0.0),
                        'timestamp': time.time()
                    }
                )
                
                # Log detection
                self.logger.info(f"Detection #{self.detection_count}: {detection['class_name']} "
                               f"(confidence: {detection['confidence']:.2f})")
                
                # Draw detection on frame (for debugging)
                if self.config.get('debug', {}).get('show_detections', False):
                    frame_with_detections = self.wildlife_detector.draw_detections(frame, detections)
                    cv2.imshow('Detections', frame_with_detections)
                    cv2.waitKey(1)
        
        except Exception as e:
            self.logger.error(f"Error handling detections: {e}")
    
    def _on_motion_detected(self, frame: np.ndarray, motion_score: float):
        """Callback for motion detection"""
        self.logger.debug(f"Motion detected: score={motion_score:.3f}")
    
    def _on_motion_start(self):
        """Callback for motion start"""
        self.logger.info("Motion started - beginning detection")
    
    def _on_motion_end(self, duration: float):
        """Callback for motion end"""
        self.logger.info(f"Motion ended after {duration:.2f} seconds")
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        uptime = time.time() - self.start_time
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'current_fps': self.current_fps,
            'total_detections': self.detection_count,
            'total_motion_events': self.motion_count,
            'camera_status': self.camera.is_streaming() if self.camera else False,
            'storage_stats': self.storage.get_storage_statistics() if self.storage else {},
            'cloud_sync_status': self.cloud_sync.get_status() if self.cloud_sync else {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {}
    
    def run_interactive(self):
        """Run system in interactive mode with keyboard controls"""
        self.start()
        
        print("Wildlife Detection System Running")
        print("Press 'q' to quit, 's' for status, 'h' for help")
        
        try:
            while self.is_running:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_status()
                elif key == ord('h'):
                    self._print_help()
                elif key == ord('c'):
                    self._force_cloud_sync()
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.stop()
    
    def _print_status(self):
        """Print system status"""
        status = self.get_system_status()
        
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        print(f"Running: {status['is_running']}")
        print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"FPS: {status['current_fps']}")
        print(f"Total Detections: {status['total_detections']}")
        print(f"Motion Events: {status['total_motion_events']}")
        print(f"Camera Active: {status['camera_status']}")
        
        # Storage info
        storage_stats = status.get('storage_stats', {})
        if storage_stats:
            print(f"Storage Usage: {storage_stats.get('total_storage_mb', 0):.1f}MB")
            print(f"Unsynced: {storage_stats.get('unsynced_detections', 0)}")
        
        # Cloud sync info
        sync_status = status.get('cloud_sync_status', {})
        if sync_status:
            print(f"Cloud Sync: {'Active' if sync_status.get('is_syncing', False) else 'Idle'}")
            print(f"Last Sync: {sync_status.get('last_sync_time', 'Never')}")
        
        print("="*50)
    
    def _print_help(self):
        """Print help information"""
        print("\n" + "="*50)
        print("CONTROLS")
        print("="*50)
        print("q - Quit system")
        print("s - Show status")
        print("h - Show this help")
        print("c - Force cloud sync")
        print("="*50)
    
    def _force_cloud_sync(self):
        """Force immediate cloud synchronization"""
        if self.cloud_sync:
            print("Forcing cloud sync...")
            result = self.cloud_sync.sync_now()
            if result.get('success'):
                print(f"Sync completed in {result.get('duration', 0):.2f}s")
            else:
                print(f"Sync failed: {result.get('error', 'Unknown error')}")

def signal_handler(signum, frame):
    """Handle system signals"""
    print("\nReceived signal to stop. Shutting down...")
    if hasattr(signal_handler, 'system'):
        signal_handler.system.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Wildlife Detection System')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--daemon', '-d', action='store_true',
                       help='Run as daemon (non-interactive)')
    
    args = parser.parse_args()
    
    # Create detection system
    try:
        system = WildlifeDetectionSystem(args.config)
        
        # Set up signal handlers
        signal_handler.system = system
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if args.interactive:
            # Run in interactive mode
            system.run_interactive()
        elif args.daemon:
            # Run as daemon
            system.start()
            
            # Keep running until interrupted
            try:
                while system.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                system.stop()
        else:
            # Default: run for a fixed time
            system.start()
            time.sleep(3600)  # Run for 1 hour
            system.stop()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
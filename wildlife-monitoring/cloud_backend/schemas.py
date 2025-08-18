"""
Database schema definitions for the wildlife monitoring system
"""

# SQLite schema creation scripts

DETECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    species TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    location_x INTEGER,
    location_y INTEGER,
    width INTEGER,
    height INTEGER,
    image_path TEXT,
    video_path TEXT,
    device_id TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices (id)
);
"""

DEVICES_TABLE = """
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    latitude REAL,
    longitude REAL,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    last_heartbeat DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    model TEXT,
    firmware_version TEXT,
    battery_level INTEGER CHECK (battery_level >= 0 AND battery_level <= 100),
    storage_used INTEGER DEFAULT 0,
    storage_total INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

SPECIES_TABLE = """
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    scientific_name TEXT,
    common_names TEXT, -- JSON array of common names
    conservation_status TEXT CHECK (conservation_status IN (
        'Least Concern', 'Near Threatened', 'Vulnerable', 
        'Endangered', 'Critically Endangered', 'Extinct in the Wild', 'Extinct'
    )),
    description TEXT,
    habitat TEXT,
    diet TEXT,
    behavior TEXT,
    image_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

DETECTION_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    device_id TEXT,
    detection_id INTEGER,
    error_code TEXT,
    stack_trace TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices (id),
    FOREIGN KEY (detection_id) REFERENCES detections (id)
);
"""

ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK (type IN ('species_detection', 'device_offline', 'low_battery', 'storage_full', 'system_error')),
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    device_id TEXT,
    detection_id INTEGER,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at DATETIME,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by TEXT,
    resolved_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices (id),
    FOREIGN KEY (detection_id) REFERENCES detections (id)
);
"""

USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    role TEXT DEFAULT 'viewer' CHECK (role IN ('admin', 'operator', 'viewer')),
    active BOOLEAN DEFAULT TRUE,
    last_login DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

# Indexes for better performance
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_detections_species ON detections(species);",
    "CREATE INDEX IF NOT EXISTS idx_detections_device_id ON detections(device_id);",
    "CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);",
    "CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);",
    "CREATE INDEX IF NOT EXISTS idx_devices_last_heartbeat ON devices(last_heartbeat);",
    "CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_detection_logs_level ON detection_logs(level);",
    "CREATE INDEX IF NOT EXISTS idx_detection_logs_device_id ON detection_logs(device_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);",
]

# Triggers for updating timestamps
TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS update_detections_timestamp 
    AFTER UPDATE ON detections
    FOR EACH ROW
    BEGIN
        UPDATE detections SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    """,
    
    """
    CREATE TRIGGER IF NOT EXISTS update_devices_timestamp 
    AFTER UPDATE ON devices
    FOR EACH ROW
    BEGIN
        UPDATE devices SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    """,
    
    """
    CREATE TRIGGER IF NOT EXISTS update_species_timestamp 
    AFTER UPDATE ON species
    FOR EACH ROW
    BEGIN
        UPDATE species SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    """,
    
    """
    CREATE TRIGGER IF NOT EXISTS update_alerts_timestamp 
    AFTER UPDATE ON alerts
    FOR EACH ROW
    BEGIN
        UPDATE alerts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    """,
    
    """
    CREATE TRIGGER IF NOT EXISTS update_users_timestamp 
    AFTER UPDATE ON users
    FOR EACH ROW
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    """
]

# Sample data for testing
SAMPLE_SPECIES = [
    ("African Elephant", "Loxodonta africana", "Endangered", "Large mammal native to Africa"),
    ("Bengal Tiger", "Panthera tigris tigris", "Endangered", "Large cat found in India and Bangladesh"),
    ("Giant Panda", "Ailuropoda melanoleuca", "Vulnerable", "Bear species endemic to China"),
    ("Mountain Gorilla", "Gorilla beringei beringei", "Critically Endangered", "Subspecies of eastern gorilla"),
    ("Snow Leopard", "Panthera uncia", "Vulnerable", "Large cat native to mountain ranges of Central Asia"),
    ("Polar Bear", "Ursus maritimus", "Vulnerable", "Bear native to the Arctic Circle"),
    ("Amur Leopard", "Panthera pardus orientalis", "Critically Endangered", "Leopard subspecies native to southeastern Russia"),
    ("Black Rhinoceros", "Diceros bicornis", "Critically Endangered", "Species of rhinoceros native to eastern Africa"),
]

SAMPLE_DEVICES = [
    ("CAM001", "Forest Camera 1", "North Forest Trail", 45.5231, -122.6765, "TrailCam Pro", "v2.1.3"),
    ("CAM002", "River Camera 2", "Riverside Path", 45.5198, -122.6819, "WildEye 4K", "v1.8.2"),
    ("CAM003", "Mountain Camera 3", "Mountain Ridge", 45.5267, -122.6745, "NatureCam HD", "v3.0.1"),
]

def get_all_tables():
    """Return all table creation scripts"""
    return [
        DETECTIONS_TABLE,
        DEVICES_TABLE,
        SPECIES_TABLE,
        DETECTION_LOGS_TABLE,
        ALERTS_TABLE,
        USERS_TABLE,
        SESSIONS_TABLE
    ]

def get_all_indexes():
    """Return all index creation scripts"""
    return INDEXES

def get_all_triggers():
    """Return all trigger creation scripts"""
    return TRIGGERS
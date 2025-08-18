import os
from flask_sqlalchemy import SQLAlchemy
from models import db, Device, Species

def init_db(app):
    """Initialize database with app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Add some default species if they don't exist
        add_default_species()
        
        # Add some default devices if they don't exist
        add_default_devices()

def add_default_species():
    """Add default species to the database"""
    default_species = [
        {
            'name': 'Deer',
            'scientific_name': 'Odocoileus virginianus',
            'category': 'mammal',
            'description': 'White-tailed deer commonly found in forested areas',
            'conservation_status': 'Least Concern'
        },
        {
            'name': 'Bear',
            'scientific_name': 'Ursus americanus',
            'category': 'mammal',
            'description': 'Black bear native to North America',
            'conservation_status': 'Least Concern'
        },
        {
            'name': 'Coyote',
            'scientific_name': 'Canis latrans',
            'category': 'mammal',
            'description': 'Coyote found in various habitats',
            'conservation_status': 'Least Concern'
        },
        {
            'name': 'Fox',
            'scientific_name': 'Vulpes vulpes',
            'category': 'mammal',
            'description': 'Red fox with distinctive red fur',
            'conservation_status': 'Least Concern'
        },
        {
            'name': 'Raccoon',
            'scientific_name': 'Procyon lotor',
            'category': 'mammal',
            'description': 'Nocturnal raccoon with distinctive mask',
            'conservation_status': 'Least Concern'
        }
    ]
    
    for species_data in default_species:
        existing = Species.query.filter_by(name=species_data['name']).first()
        if not existing:
            species = Species(**species_data)
            db.session.add(species)
    
    db.session.commit()

def add_default_devices():
    """Add default devices to the database"""
    default_devices = [
        {
            'device_id': 'CAM001',
            'name': 'Forest Trail Camera 1',
            'location': 'North Forest Trail',
            'latitude': 45.5231,
            'longitude': -122.6765,
            'status': 'active'
        },
        {
            'device_id': 'CAM002',
            'name': 'Mountain View Camera',
            'location': 'Mountain Ridge',
            'latitude': 45.5152,
            'longitude': -122.6784,
            'status': 'active'
        },
        {
            'device_id': 'CAM003',
            'name': 'River Bank Camera',
            'location': 'River Bank Trail',
            'latitude': 45.5200,
            'longitude': -122.6700,
            'status': 'active'
        }
    ]
    
    for device_data in default_devices:
        existing = Device.query.filter_by(device_id=device_data['device_id']).first()
        if not existing:
            device = Device(**device_data)
            db.session.add(device)
    
    db.session.commit()
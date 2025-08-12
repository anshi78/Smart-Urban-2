#!/usr/bin/env python3
"""
Deployment Script for Smart Urban Settlement Management System

This script automates the deployment process for the application.
It handles environment setup, dependency installation, model training,
and server configuration.

Usage:
    python scripts/deployment.py [--production] [--docker]
"""

import os
import sys
import subprocess
import argparse
import shutil
import json
from pathlib import Path

class DeploymentManager:
    """Manages the deployment process for the Smart Urban Settlement System"""
    
    def __init__(self, production=False, docker=False):
        self.production = production
        self.docker = docker
        self.project_root = Path(__file__).parent.parent
        self.venv_name = "myvenv"
        
    def check_prerequisites(self):
        """Check if all prerequisites are installed"""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
            print("❌ Python 3.7+ is required")
            return False
        
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            print("✅ pip is available")
        except subprocess.CalledProcessError:
            print("❌ pip is not available")
            return False
        
        # Check git
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            print("✅ git is available")
        except subprocess.CalledProcessError:
            print("⚠️  git is not available (optional)")
        
        return True
    
    def setup_virtual_environment(self):
        """Set up virtual environment"""
        print("🔧 Setting up virtual environment...")
        
        venv_path = self.project_root / self.venv_name
        
        if venv_path.exists():
            print(f"⚠️  Virtual environment '{self.venv_name}' already exists")
            response = input("Do you want to recreate it? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(venv_path)
            else:
                print("Using existing virtual environment")
                return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print(f"✅ Virtual environment created at {venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_python_executable(self):
        """Get the Python executable for the virtual environment"""
        if os.name == 'nt':  # Windows
            return self.project_root / self.venv_name / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            return self.project_root / self.venv_name / "bin" / "python"
    
    def get_pip_executable(self):
        """Get the pip executable for the virtual environment"""
        if os.name == 'nt':  # Windows
            return self.project_root / self.venv_name / "Scripts" / "pip.exe"
        else:  # Unix/Linux/Mac
            return self.project_root / self.venv_name / "bin" / "pip"
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        pip_executable = self.get_pip_executable()
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([str(pip_executable), "install", "-r", str(requirements_file)], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def train_models(self):
        """Train the machine learning models"""
        print("🤖 Training machine learning models...")
        
        python_executable = self.get_python_executable()
        train_script = self.project_root / "scripts" / "train_models.py"
        
        if not train_script.exists():
            print("❌ Model training script not found")
            return False
        
        try:
            subprocess.run([str(python_executable), str(train_script)], check=True)
            print("✅ Models trained successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to train models: {e}")
            return False
    
    def create_production_config(self):
        """Create production configuration"""
        print("⚙️  Creating production configuration...")
        
        config = {
            "production": True,
            "debug": False,
            "host": "0.0.0.0",
            "port": 5000,
            "workers": 4,
            "log_level": "INFO"
        }
        
        config_file = self.project_root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Production configuration created")
        return True
    
    def create_docker_files(self):
        """Create Docker configuration files"""
        print("🐳 Creating Docker configuration...")
        
        # Dockerfile
        dockerfile_content = """FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # docker-compose.yml
        compose_content = """version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY:-dev-secret-key}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
"""
        
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        # nginx.conf
        nginx_content = """events {
    worker_connections 1024;
}

http {
    upstream flask_app {
        server web:5000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://flask_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
        
        nginx_path = self.project_root / "nginx.conf"
        with open(nginx_path, 'w') as f:
            f.write(nginx_content)
        
        print("✅ Docker configuration files created")
        return True
    
    def create_systemd_service(self):
        """Create systemd service file for production deployment"""
        print("🔧 Creating systemd service...")
        
        service_content = f"""[Unit]
Description=Smart Urban Settlement Management System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'ubuntu')}
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/{self.venv_name}/bin
ExecStart={self.project_root}/{self.venv_name}/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = self.project_root / "smart-urban-settlement.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        print("✅ Systemd service file created")
        print(f"📝 To install the service, run:")
        print(f"   sudo cp {service_path} /etc/systemd/system/")
        print(f"   sudo systemctl daemon-reload")
        print(f"   sudo systemctl enable smart-urban-settlement")
        print(f"   sudo systemctl start smart-urban-settlement")
        
        return True
    
    def run_tests(self):
        """Run the test suite"""
        print("🧪 Running tests...")
        
        python_executable = self.get_python_executable()
        
        try:
            # Run unit tests
            subprocess.run([str(python_executable), "-m", "unittest", "discover", "tests"], check=True)
            print("✅ Tests passed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed: {e}")
            return False
    
    def start_development_server(self):
        """Start the development server"""
        print("🚀 Starting development server...")
        
        python_executable = self.get_python_executable()
        app_file = self.project_root / "app.py"
        
        try:
            print("🌐 Server starting at http://localhost:5000")
            print("Press Ctrl+C to stop the server")
            subprocess.run([str(python_executable), str(app_file)])
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start server: {e}")
            return False
        
        return True
    
    def build_docker_image(self):
        """Build Docker image"""
        print("🐳 Building Docker image...")
        
        try:
            subprocess.run(["docker", "build", "-t", "smart-urban-settlement", "."], check=True)
            print("✅ Docker image built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Docker image: {e}")
            return False
    
    def deploy(self):
        """Main deployment process"""
        print("🚀 Smart Urban Settlement Management System - Deployment")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Setup virtual environment
        if not self.setup_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Train models
        if not self.train_models():
            return False
        
        # Run tests
        if not self.run_tests():
            return False
        
        # Production-specific setup
        if self.production:
            if not self.create_production_config():
                return False
            
            if not self.create_systemd_service():
                return False
        
        # Docker-specific setup
        if self.docker:
            if not self.create_docker_files():
                return False
            
            if not self.build_docker_image():
                return False
        
        print("\n🎉 Deployment completed successfully!")
        
        if self.production:
            print("\n📋 Production Deployment Instructions:")
            print("1. Copy the systemd service file to /etc/systemd/system/")
            print("2. Enable and start the service")
            print("3. Configure your web server (nginx/apache) as reverse proxy")
            print("4. Set up SSL certificates")
        
        if self.docker:
            print("\n🐳 Docker Deployment Instructions:")
            print("1. Run: docker-compose up -d")
            print("2. Access the application at http://localhost")
            print("3. Check logs: docker-compose logs -f")
        
        if not self.production and not self.docker:
            print("\n🚀 Starting development server...")
            return self.start_development_server()
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deploy Smart Urban Settlement Management System")
    parser.add_argument("--production", action="store_true", help="Deploy for production")
    parser.add_argument("--docker", action="store_true", help="Create Docker configuration")
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager(production=args.production, docker=args.docker)
    
    # Run deployment
    success = manager.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
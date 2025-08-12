#!/usr/bin/env python3
"""
Model Training Script for Smart Urban Settlement Management System

This script demonstrates how the machine learning models were trained for:
1. Air Quality Prediction
2. Traffic Congestion Prediction  
3. Housing Allocation Prediction

Usage:
    python scripts/train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model trainer class for all prediction models"""
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.air_quality_model = None
        self.traffic_model = None
        self.housing_model = None
        
        # Initialize encoders
        self.country_encoder = LabelEncoder()
        self.status_encoder = LabelEncoder()
    
    def generate_sample_data(self):
        """Generate sample datasets for demonstration"""
        logger.info("Generating sample datasets...")
        
        # Air Quality Data
        np.random.seed(42)
        countries = ['United States', 'India', 'China', 'Germany', 'Brazil', 'Japan', 'France', 'Canada']
        statuses = ['Developed', 'Developing']
        
        n_samples = 1000
        air_quality_data = []
        
        for _ in range(n_samples):
            country = np.random.choice(countries)
            status = np.random.choice(statuses)
            year = np.random.randint(2020, 2025)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            
            # Generate realistic AQI based on factors
            base_aqi = 30 if status == 'Developed' else 80
            seasonal_factor = 1.2 if month in [6, 7, 8] else 0.8  # Summer has higher AQI
            country_factor = 1.5 if country in ['India', 'China'] else 1.0
            
            aqi = base_aqi * seasonal_factor * country_factor + np.random.normal(0, 15)
            aqi = max(0, min(300, aqi))  # Clamp between 0 and 300
            
            air_quality_data.append({
                'Country': country,
                'Status': status,
                'year': year,
                'month': month,
                'day': day,
                'AQI': aqi
            })
        
        air_quality_df = pd.DataFrame(air_quality_data)
        air_quality_df.to_csv(f'{self.data_dir}/air_quality_data.csv', index=False)
        logger.info(f"Generated air quality dataset with {len(air_quality_df)} samples")
        
        # Traffic Data
        traffic_data = []
        for _ in range(2000):
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            
            # Generate realistic congestion based on time patterns
            peak_hour_factor = 1.8 if hour in [7, 8, 9, 17, 18, 19] else 0.6
            weekday_factor = 1.5 if day_of_week < 5 else 0.8  # Weekdays have more traffic
            
            congestion = 30 * peak_hour_factor * weekday_factor + np.random.normal(0, 10)
            congestion = max(0, min(100, congestion))  # Clamp between 0 and 100
            
            traffic_data.append({
                'hour': hour,
                'day_of_week': day_of_week,
                'congestion_level': congestion
            })
        
        traffic_df = pd.DataFrame(traffic_data)
        traffic_df.to_csv(f'{self.data_dir}/traffic_data.csv', index=False)
        logger.info(f"Generated traffic dataset with {len(traffic_df)} samples")
        
        # Housing Data
        housing_data = []
        for _ in range(1500):
            population_density = np.random.randint(500, 20000)
            average_income = np.random.randint(20000, 150000)
            
            # Generate realistic housing prices based on factors
            density_factor = population_density / 10000
            income_factor = average_income / 50000
            
            base_price = 30000
            price = base_price * (1 + density_factor * 0.5 + income_factor * 0.3) + np.random.normal(0, 5000)
            price = max(10000, min(100000, price))  # Clamp between 10k and 100k
            
            housing_data.append({
                'population_density': population_density,
                'average_income': average_income,
                'housing_price': price
            })
        
        housing_df = pd.DataFrame(housing_data)
        housing_df.to_csv(f'{self.data_dir}/housing_data.csv', index=False)
        logger.info(f"Generated housing dataset with {len(housing_df)} samples")
    
    def train_air_quality_model(self):
        """Train air quality prediction model"""
        logger.info("Training air quality prediction model...")
        
        # Load data
        df = pd.read_csv(f'{self.data_dir}/air_quality_data.csv')
        
        # Encode categorical variables
        df['Country_encoded'] = self.country_encoder.fit_transform(df['Country'])
        df['Status_encoded'] = self.status_encoder.fit_transform(df['Status'])
        
        # Prepare features
        features = ['Country_encoded', 'Status_encoded', 'year', 'month', 'day']
        X = df[features]
        y = df['AQI']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.air_quality_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.air_quality_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.air_quality_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Air Quality Model Performance:")
        logger.info(f"  MSE: {mse:.2f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  MAE: {mae:.2f}")
        
        # Save model and encoders
        joblib.dump(self.air_quality_model, f'{self.models_dir}/air_quality_model.pkl')
        joblib.dump(self.country_encoder, f'{self.models_dir}/label_encoder_country.pkl')
        joblib.dump(self.status_encoder, f'{self.models_dir}/label_encoder_status.pkl')
        
        # Create visualization
        self.plot_air_quality_results(y_test, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'model': self.air_quality_model
        }
    
    def train_traffic_model(self):
        """Train traffic congestion prediction model"""
        logger.info("Training traffic congestion prediction model...")
        
        # Load data
        df = pd.read_csv(f'{self.data_dir}/traffic_data.csv')
        
        # Prepare features
        features = ['hour', 'day_of_week']
        X = df[features]
        y = df['congestion_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.traffic_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.traffic_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.traffic_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Traffic Model Performance:")
        logger.info(f"  MSE: {mse:.2f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  MAE: {mae:.2f}")
        
        # Save model
        joblib.dump(self.traffic_model, f'{self.models_dir}/traffic_congestion_model.pkl')
        
        # Create visualization
        self.plot_traffic_results(y_test, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'model': self.traffic_model
        }
    
    def train_housing_model(self):
        """Train housing price prediction model"""
        logger.info("Training housing price prediction model...")
        
        # Load data
        df = pd.read_csv(f'{self.data_dir}/housing_data.csv')
        
        # Prepare features
        features = ['population_density', 'average_income']
        X = df[features]
        y = df['housing_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.housing_model = LinearRegression()
        self.housing_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.housing_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Housing Model Performance:")
        logger.info(f"  MSE: {mse:.2f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  MAE: {mae:.2f}")
        
        # Save model
        joblib.dump(self.housing_model, f'{self.models_dir}/housing_model.pkl')
        
        # Create visualization
        self.plot_housing_results(y_test, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'model': self.housing_model
        }
    
    def plot_air_quality_results(self, y_true, y_pred):
        """Create visualization for air quality model results"""
        plt.figure(figsize=(12, 4))
        
        # Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Actual vs Predicted AQI')
        
        # Residuals
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Distribution of predictions
        plt.subplot(1, 3, 3)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted')
        plt.hist(y_true, bins=30, alpha=0.7, label='Actual')
        plt.xlabel('AQI')
        plt.ylabel('Frequency')
        plt.title('Distribution of AQI Values')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/images/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_traffic_results(self, y_true, y_pred):
        """Create visualization for traffic model results"""
        plt.figure(figsize=(12, 4))
        
        # Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Congestion Level')
        plt.ylabel('Predicted Congestion Level')
        plt.title('Actual vs Predicted Congestion')
        
        # Residuals
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Congestion Level')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Distribution of predictions
        plt.subplot(1, 3, 3)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted')
        plt.hist(y_true, bins=30, alpha=0.7, label='Actual')
        plt.xlabel('Congestion Level')
        plt.ylabel('Frequency')
        plt.title('Distribution of Congestion Levels')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/images/actual_vs_predicted_congestion.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_housing_results(self, y_true, y_pred):
        """Create visualization for housing model results"""
        plt.figure(figsize=(12, 4))
        
        # Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Housing Price')
        plt.ylabel('Predicted Housing Price')
        plt.title('Actual vs Predicted Housing Price')
        
        # Residuals
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Housing Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Distribution of predictions
        plt.subplot(1, 3, 3)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted')
        plt.hist(y_true, bins=30, alpha=0.7, label='Actual')
        plt.xlabel('Housing Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of Housing Prices')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/images/housing_demand_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_summary(self, results):
        """Create a summary of all model performances"""
        logger.info("\n" + "="*60)
        logger.info("MODEL TRAINING SUMMARY")
        logger.info("="*60)
        
        for model_name, result in results.items():
            logger.info(f"\n{model_name.upper()} MODEL:")
            logger.info(f"  Mean Squared Error: {result['mse']:.2f}")
            logger.info(f"  R² Score: {result['r2']:.3f}")
            logger.info(f"  Mean Absolute Error: {result['mae']:.2f}")
        
        # Save summary to file
        with open('model_training_summary.txt', 'w') as f:
            f.write("Smart Urban Settlement Management System - Model Training Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name.upper()} MODEL:\n")
                f.write(f"  Mean Squared Error: {result['mse']:.2f}\n")
                f.write(f"  R² Score: {result['r2']:.3f}\n")
                f.write(f"  Mean Absolute Error: {result['mae']:.2f}\n\n")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("Starting model training process...")
        
        # Generate sample data if it doesn't exist
        if not os.path.exists(f'{self.data_dir}/air_quality_data.csv'):
            self.generate_sample_data()
        
        # Train all models
        results = {}
        
        try:
            results['air_quality'] = self.train_air_quality_model()
            results['traffic'] = self.train_traffic_model()
            results['housing'] = self.train_housing_model()
            
            # Create summary
            self.create_model_summary(results)
            
            logger.info("✅ All models trained successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during model training: {e}")
            raise

def main():
    """Main function to run model training"""
    print("🤖 Smart Urban Settlement Management System - Model Training")
    print("="*60)
    
    # Create trainer instance
    trainer = ModelTrainer()
    
    try:
        # Train all models
        results = trainer.train_all_models()
        
        print("\n🎉 Model training completed successfully!")
        print("📁 Models saved in 'models/' directory")
        print("📊 Visualizations saved in 'static/images/' directory")
        print("📋 Training summary saved as 'model_training_summary.txt'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
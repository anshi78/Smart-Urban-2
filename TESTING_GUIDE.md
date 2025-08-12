# Testing Guide - Smart Urban Settlement Management System

## 🧪 Testing Overview

This guide provides comprehensive testing strategies and examples for the Smart Urban Settlement Management System. Testing is crucial for ensuring reliability, performance, and maintainability of the application.

## 📋 Testing Strategy

### Testing Pyramid

```
    /\
   /  \     E2E Tests (Few)
  /____\    Integration Tests (Some)
 /______\   Unit Tests (Many)
```

1. **Unit Tests** - Test individual functions and components
2. **Integration Tests** - Test interactions between components
3. **End-to-End Tests** - Test complete user workflows
4. **Performance Tests** - Test system performance under load

## 🧩 Unit Testing

### Test Structure

```python
# tests/test_models.py
import unittest
import numpy as np
import joblib
from unittest.mock import patch, MagicMock

class TestAirQualityModel(unittest.TestCase):
    """Test cases for air quality prediction model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = joblib.load('models/air_quality_model.pkl')
        self.country_encoder = joblib.load('models/label_encoder_country.pkl')
        self.status_encoder = joblib.load('models/label_encoder_status.pkl')
    
    def test_model_loaded_successfully(self):
        """Test that model loads without errors"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.country_encoder)
        self.assertIsNotNone(self.status_encoder)
    
    def test_prediction_with_valid_input(self):
        """Test prediction with valid input data"""
        # Prepare test data
        country = "United States"
        status = "Developed"
        year = 2024.0
        month = 6.0
        day = 15.0
        
        # Encode categorical variables
        country_encoded = self.country_encoder.transform([country])[0]
        status_encoded = self.status_encoder.transform([status])[0]
        
        # Create feature array
        features = np.array([[country_encoded, status_encoded, year, month, day]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Assertions
        self.assertIsInstance(prediction, (int, float))
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 500)  # Reasonable AQI range
    
    def test_prediction_with_invalid_country(self):
        """Test prediction with invalid country"""
        with self.assertRaises(ValueError):
            self.country_encoder.transform(["InvalidCountry"])
    
    def test_prediction_with_invalid_status(self):
        """Test prediction with invalid status"""
        with self.assertRaises(ValueError):
            self.status_encoder.transform(["InvalidStatus"])

class TestTrafficModel(unittest.TestCase):
    """Test cases for traffic congestion prediction model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = joblib.load('models/traffic_congestion_model.pkl')
    
    def test_model_loaded_successfully(self):
        """Test that model loads without errors"""
        self.assertIsNotNone(self.model)
    
    def test_prediction_with_valid_input(self):
        """Test prediction with valid input data"""
        # Test data
        hour = 14  # 2 PM
        day_of_week = 2  # Wednesday
        
        features = np.array([[hour, day_of_week]])
        prediction = self.model.predict(features)[0]
        
        # Assertions
        self.assertIsInstance(prediction, (int, float))
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 100)  # Reasonable congestion range
    
    def test_prediction_edge_cases(self):
        """Test prediction with edge case inputs"""
        test_cases = [
            (0, 0),   # Midnight on Monday
            (23, 6),  # 11 PM on Sunday
            (12, 3),  # Noon on Thursday
        ]
        
        for hour, day in test_cases:
            with self.subTest(hour=hour, day=day):
                features = np.array([[hour, day]])
                prediction = self.model.predict(features)[0]
                self.assertIsInstance(prediction, (int, float))

class TestHousingModel(unittest.TestCase):
    """Test cases for housing allocation prediction model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = joblib.load('models/housing_model.pkl')
    
    def test_model_loaded_successfully(self):
        """Test that model loads without errors"""
        self.assertIsNotNone(self.model)
    
    def test_prediction_with_valid_input(self):
        """Test prediction with valid input data"""
        # Test data
        population_density = 5000
        average_income = 75000
        
        features = np.array([[population_density, average_income]])
        prediction = self.model.predict(features)[0]
        
        # Assertions
        self.assertIsInstance(prediction, (int, float))
        self.assertGreaterEqual(prediction, 0)
    
    def test_prediction_with_different_scenarios(self):
        """Test prediction with different demographic scenarios"""
        test_cases = [
            (1000, 30000),   # Low density, low income
            (10000, 100000), # High density, high income
            (5000, 50000),   # Medium density, medium income
        ]
        
        for density, income in test_cases:
            with self.subTest(density=density, income=income):
                features = np.array([[density, income]])
                prediction = self.model.predict(features)[0]
                self.assertIsInstance(prediction, (int, float))

if __name__ == '__main__':
    unittest.main()
```

### Utility Function Tests

```python
# tests/test_utils.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_categorize_aqi_good(self):
        """Test AQI categorization for good air quality"""
        from app import categorize_aqi
        
        test_values = [0, 25, 50]
        for value in test_values:
            with self.subTest(value=value):
                category = categorize_aqi(value)
                self.assertEqual(category, "Good")
    
    def test_categorize_aqi_moderate(self):
        """Test AQI categorization for moderate air quality"""
        from app import categorize_aqi
        
        test_values = [51, 75, 100]
        for value in test_values:
            with self.subTest(value=value):
                category = categorize_aqi(value)
                self.assertEqual(category, "Moderate")
    
    def test_categorize_aqi_bad(self):
        """Test AQI categorization for bad air quality"""
        from app import categorize_aqi
        
        test_values = [101, 150, 200, 300]
        for value in test_values:
            with self.subTest(value=value):
                category = categorize_aqi(value)
                self.assertEqual(category, "Bad")
    
    def test_validate_input_parameters(self):
        """Test input parameter validation"""
        from app import validate_input_parameters
        
        # Valid inputs
        valid_data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        result = validate_input_parameters(valid_data)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        
        # Invalid inputs
        invalid_data = {
            'Country': '',
            'Status': 'Invalid',
            'year': 'invalid',
            'month': '13',
            'day': '32'
        }
        
        result = validate_input_parameters(invalid_data)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_preprocess_features(self):
        """Test feature preprocessing"""
        from app import preprocess_features
        
        # Test data
        country = "United States"
        status = "Developed"
        year = 2024.0
        month = 6.0
        day = 15.0
        
        features = preprocess_features(country, status, year, month, day)
        
        # Assertions
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (1, 5))
        self.assertTrue(np.all(np.isfinite(features)))

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling functions"""
    
    def test_handle_model_error(self):
        """Test model error handling"""
        from app import handle_model_error
        
        # Test with ValueError
        error = ValueError("Invalid input")
        result = handle_model_error(error)
        
        self.assertIn('error', result)
        self.assertEqual(result['status'], 'error')
    
    def test_handle_validation_error(self):
        """Test validation error handling"""
        from app import handle_validation_error
        
        errors = ["Invalid country", "Invalid year"]
        result = handle_validation_error(errors)
        
        self.assertIn('errors', result)
        self.assertEqual(len(result['errors']), 2)

if __name__ == '__main__':
    unittest.main()
```

## 🔗 Integration Testing

### API Integration Tests

```python
# tests/test_integration.py
import unittest
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"
        self.session = requests.Session()
        
        # Wait for server to be ready
        self.wait_for_server()
    
    def wait_for_server(self, timeout=30):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/")
                if response.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError("Server not ready within timeout")
    
    def test_dashboard_access(self):
        """Test dashboard accessibility"""
        response = self.session.get(f"{self.base_url}/")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Smart Urban Settlement Management', response.text)
        self.assertIn('Air Quality Prediction', response.text)
        self.assertIn('Traffic Congestion Prediction', response.text)
        self.assertIn('Affordable Housing Prediction', response.text)
    
    def test_air_quality_prediction_workflow(self):
        """Test complete air quality prediction workflow"""
        # 1. Access dashboard
        response = self.session.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        
        # 2. Submit prediction request
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/air_quality",
            data=data
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted AQI', response.text)
    
    def test_traffic_prediction_workflow(self):
        """Test complete traffic prediction workflow"""
        data = {
            'hour': '14',
            'day_of_week': '2'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/traffic",
            data=data
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Traffic Congestion Level', response.text)
    
    def test_housing_prediction_workflow(self):
        """Test complete housing prediction workflow"""
        data = {
            'population_density': '5000',
            'average_income': '75000'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/housing",
            data=data
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted Housing Price', response.text)
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs"""
        # Test invalid country
        data = {
            'Country': 'InvalidCountry',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict/air_quality",
            data=data
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('Invalid category input', response.text)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        def make_request():
            data = {
                'Country': 'United States',
                'Status': 'Developed',
                'year': '2024',
                'month': '6',
                'day': '15'
            }
            response = self.session.post(
                f"{self.base_url}/predict/air_quality",
                data=data
            )
            return response.status_code
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        for status_code in results:
            self.assertEqual(status_code, 200)
    
    def test_response_time(self):
        """Test response time performance"""
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/predict/air_quality",
            data=data
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 5.0)  # Should respond within 5 seconds

class TestDataFlow(unittest.TestCase):
    """Test data flow through the system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"
        self.session = requests.Session()
    
    def test_data_consistency(self):
        """Test data consistency across multiple requests"""
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        # Make multiple requests with same data
        responses = []
        for _ in range(5):
            response = self.session.post(
                f"{self.base_url}/predict/air_quality",
                data=data
            )
            responses.append(response.text)
        
        # All responses should be identical
        first_response = responses[0]
        for response in responses[1:]:
            self.assertEqual(response, first_response)
    
    def test_model_persistence(self):
        """Test that models persist across requests"""
        # This test ensures models are loaded once and reused
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        # Make requests to all endpoints
        endpoints = [
            '/predict/air_quality',
            '/predict/traffic',
            '/predict/housing'
        ]
        
        for endpoint in endpoints:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=data
            )
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
```

## 🚀 Performance Testing

### Load Testing

```python
# tests/test_performance.py
import unittest
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class TestPerformance(unittest.TestCase):
    """Performance tests for the application"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"
        self.session = requests.Session()
    
    def test_single_request_performance(self):
        """Test performance of single requests"""
        endpoints = [
            ('/predict/air_quality', {
                'Country': 'United States',
                'Status': 'Developed',
                'year': '2024',
                'month': '6',
                'day': '15'
            }),
            ('/predict/traffic', {
                'hour': '14',
                'day_of_week': '2'
            }),
            ('/predict/housing', {
                'population_density': '5000',
                'average_income': '75000'
            })
        ]
        
        for endpoint, data in endpoints:
            with self.subTest(endpoint=endpoint):
                # Measure response time
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    data=data
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Assertions
                self.assertEqual(response.status_code, 200)
                self.assertLess(response_time, 2.0)  # Should respond within 2 seconds
    
    def test_concurrent_load_performance(self):
        """Test performance under concurrent load"""
        def make_request(endpoint, data):
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data=data
            )
            end_time = time.time()
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'success': response.status_code == 200
            }
        
        # Test data
        test_cases = [
            ('/predict/air_quality', {
                'Country': 'United States',
                'Status': 'Developed',
                'year': '2024',
                'month': '6',
                'day': '15'
            }),
            ('/predict/traffic', {
                'hour': '14',
                'day_of_week': '2'
            }),
            ('/predict/housing', {
                'population_density': '5000',
                'average_income': '75000'
            })
        ]
        
        # Test different load levels
        load_levels = [10, 25, 50, 100]
        
        for load in load_levels:
            with self.subTest(load=load):
                results = []
                
                with ThreadPoolExecutor(max_workers=load) as executor:
                    # Submit requests for all endpoints
                    futures = []
                    for endpoint, data in test_cases:
                        for _ in range(load // len(test_cases)):
                            future = executor.submit(make_request, endpoint, data)
                            futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        results.append(future.result())
                
                # Analyze results
                successful_requests = [r for r in results if r['success']]
                response_times = [r['response_time'] for r in successful_requests]
                
                # Assertions
                success_rate = len(successful_requests) / len(results)
                avg_response_time = statistics.mean(response_times) if response_times else 0
                max_response_time = max(response_times) if response_times else 0
                
                self.assertGreaterEqual(success_rate, 0.95)  # 95% success rate
                self.assertLess(avg_response_time, 3.0)  # Average < 3 seconds
                self.assertLess(max_response_time, 10.0)  # Max < 10 seconds
    
    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple requests
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        for _ in range(100):
            response = self.session.post(
                f"{self.base_url}/predict/air_quality",
                data=data
            )
            self.assertEqual(response.status_code, 200)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        self.assertLess(memory_increase, 100)
    
    def test_stress_test(self):
        """Stress test the application"""
        def stress_worker():
            data = {
                'Country': 'United States',
                'Status': 'Developed',
                'year': '2024',
                'month': '6',
                'day': '15'
            }
            
            for _ in range(50):  # 50 requests per worker
                try:
                    response = self.session.post(
                        f"{self.base_url}/predict/air_quality",
                        data=data,
                        timeout=10
                    )
                    if response.status_code != 200:
                        return False
                except:
                    return False
            return True
        
        # Run stress test with multiple workers
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker) for _ in range(num_workers)]
            results = [future.result() for future in futures]
        
        # All workers should complete successfully
        success_rate = sum(results) / len(results)
        self.assertGreaterEqual(success_rate, 0.9)  # 90% success rate

if __name__ == '__main__':
    unittest.main()
```

## 🔍 End-to-End Testing

### User Workflow Tests

```python
# tests/test_e2e.py
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests using Selenium"""
    
    def setUp(self):
        """Set up test fixtures"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("http://localhost:5000")
        self.wait = WebDriverWait(self.driver, 10)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.driver.quit()
    
    def test_air_quality_prediction_workflow(self):
        """Test complete air quality prediction user workflow"""
        # Wait for page to load
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        
        # Verify page title
        title = self.driver.find_element(By.TAG_NAME, "h1")
        self.assertIn("Smart Urban Settlement Management", title.text)
        
        # Find air quality form
        country_input = self.driver.find_element(By.NAME, "Country")
        status_input = self.driver.find_element(By.NAME, "Status")
        year_input = self.driver.find_element(By.NAME, "year")
        month_input = self.driver.find_element(By.NAME, "month")
        day_input = self.driver.find_element(By.NAME, "day")
        
        # Fill form
        country_input.send_keys("United States")
        status_input.send_keys("Developed")
        year_input.send_keys("2024")
        month_input.send_keys("6")
        day_input.send_keys("15")
        
        # Submit form
        submit_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
        submit_button.click()
        
        # Wait for prediction result
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Predicted AQI')]")))
        
        # Verify result
        result = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Predicted AQI')]")
        self.assertIn("Predicted AQI", result.text)
    
    def test_traffic_prediction_workflow(self):
        """Test complete traffic prediction user workflow"""
        # Find traffic form
        hour_input = self.driver.find_element(By.NAME, "hour")
        day_of_week_input = self.driver.find_element(By.NAME, "day_of_week")
        
        # Fill form
        hour_input.send_keys("14")
        day_of_week_input.send_keys("2")
        
        # Submit form
        submit_buttons = self.driver.find_elements(By.XPATH, "//button[@type='submit']")
        submit_buttons[1].click()  # Second form
        
        # Wait for prediction result
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Traffic Congestion Level')]")))
        
        # Verify result
        result = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Traffic Congestion Level')]")
        self.assertIn("Traffic Congestion Level", result.text)
    
    def test_housing_prediction_workflow(self):
        """Test complete housing prediction user workflow"""
        # Find housing form
        population_density_input = self.driver.find_element(By.NAME, "population_density")
        average_income_input = self.driver.find_element(By.NAME, "average_income")
        
        # Fill form
        population_density_input.send_keys("5000")
        average_income_input.send_keys("75000")
        
        # Submit form
        submit_buttons = self.driver.find_elements(By.XPATH, "//button[@type='submit']")
        submit_buttons[2].click()  # Third form
        
        # Wait for prediction result
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Predicted Housing Price')]")))
        
        # Verify result
        result = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Predicted Housing Price')]")
        self.assertIn("Predicted Housing Price", result.text)
    
    def test_error_handling(self):
        """Test error handling in UI"""
        # Try to submit empty form
        submit_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
        submit_button.click()
        
        # Should stay on same page (no navigation)
        current_url = self.driver.current_url
        self.assertIn("localhost:5000", current_url)
    
    def test_responsive_design(self):
        """Test responsive design on different screen sizes"""
        # Test mobile viewport
        self.driver.set_window_size(375, 667)  # iPhone 6/7/8
        self.driver.refresh()
        
        # Verify elements are still accessible
        title = self.driver.find_element(By.TAG_NAME, "h1")
        self.assertIn("Smart Urban Settlement Management", title.text)
        
        # Test tablet viewport
        self.driver.set_window_size(768, 1024)  # iPad
        self.driver.refresh()
        
        # Verify elements are still accessible
        title = self.driver.find_element(By.TAG_NAME, "h1")
        self.assertIn("Smart Urban Settlement Management", title.text)

if __name__ == '__main__':
    unittest.main()
```

## 🛠️ Test Automation

### Test Runner Script

```python
# run_tests.py
#!/usr/bin/env python3
"""
Test runner script for Smart Urban Settlement Management System
"""

import unittest
import sys
import os
import time
import subprocess
from pathlib import Path

def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running Unit Tests...")
    loader = unittest.TestLoader()
    start_dir = Path('tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests"""
    print("🔗 Running Integration Tests...")
    loader = unittest.TestLoader()
    start_dir = Path('tests')
    suite = loader.discover(start_dir, pattern='test_integration.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests():
    """Run performance tests"""
    print("🚀 Running Performance Tests...")
    loader = unittest.TestLoader()
    start_dir = Path('tests')
    suite = loader.discover(start_dir, pattern='test_performance.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_e2e_tests():
    """Run end-to-end tests"""
    print("🌐 Running End-to-End Tests...")
    loader = unittest.TestLoader()
    start_dir = Path('tests')
    suite = loader.discover(start_dir, pattern='test_e2e.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def check_server_running():
    """Check if the Flask server is running"""
    try:
        import requests
        response = requests.get('http://localhost:5000', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the Flask server"""
    print("🚀 Starting Flask server...")
    process = subprocess.Popen([
        sys.executable, 'app.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    return process

def stop_server(process):
    """Stop the Flask server"""
    if process:
        process.terminate()
        process.wait()

def main():
    """Main test runner"""
    print("🧪 Smart Urban Settlement Management System - Test Suite")
    print("=" * 60)
    
    # Check if server is running
    server_process = None
    if not check_server_running():
        print("⚠️  Flask server not running. Starting server...")
        server_process = start_server()
        
        # Wait for server to be ready
        for i in range(30):
            if check_server_running():
                print("✅ Server is ready!")
                break
            time.sleep(1)
        else:
            print("❌ Failed to start server")
            return False
    
    # Run tests
    test_results = {}
    
    # Unit tests
    test_results['unit'] = run_unit_tests()
    
    # Integration tests
    test_results['integration'] = run_integration_tests()
    
    # Performance tests
    test_results['performance'] = run_performance_tests()
    
    # E2E tests (optional)
    if '--e2e' in sys.argv:
        test_results['e2e'] = run_e2e_tests()
    
    # Stop server if we started it
    if server_process:
        stop_server(server_process)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_type, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_type.title()} Tests: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"\nOverall Result: {overall_status}")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov selenium
    
    - name: Run unit tests
      run: |
        python -m pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Start Flask server
      run: |
        python app.py &
        sleep 10
    
    - name: Run integration tests
      run: |
        python -m pytest tests/test_integration.py -v
    
    - name: Run performance tests
      run: |
        python -m pytest tests/test_performance.py -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## 📊 Test Coverage

### Coverage Configuration

```ini
# .coveragerc
[run]
source = app.py
omit = 
    */tests/*
    */venv/*
    */env/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

### Coverage Report

```python
# generate_coverage_report.py
#!/usr/bin/env python3
"""
Generate coverage report for the project
"""

import coverage
import os
import sys

def generate_coverage_report():
    """Generate coverage report"""
    # Start coverage measurement
    cov = coverage.Coverage()
    cov.start()
    
    # Import and run tests
    import unittest
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Generate reports
    cov.report()
    cov.html_report(directory='htmlcov')
    cov.xml_report(outfile='coverage.xml')
    
    # Print summary
    total_coverage = cov.report()
    print(f"\nTotal Coverage: {total_coverage:.1f}%")
    
    return total_coverage

if __name__ == '__main__':
    coverage = generate_coverage_report()
    sys.exit(0 if coverage >= 80 else 1)  # Exit with error if coverage < 80%
```

## 🎯 Testing Best Practices

### 1. Test Organization

- **Test files should mirror the structure of the source code**
- **Use descriptive test names that explain what is being tested**
- **Group related tests in test classes**
- **Use setUp() and tearDown() methods for common test fixtures**

### 2. Test Data Management

```python
# tests/test_data.py
"""Test data fixtures"""

AIR_QUALITY_TEST_DATA = [
    {
        'country': 'United States',
        'status': 'Developed',
        'year': 2024,
        'month': 6,
        'day': 15,
        'expected_category': 'Good'
    },
    {
        'country': 'India',
        'status': 'Developing',
        'year': 2024,
        'month': 12,
        'day': 25,
        'expected_category': 'Moderate'
    }
]

TRAFFIC_TEST_DATA = [
    {
        'hour': 8,
        'day_of_week': 1,  # Monday
        'expected_severity': 'High'
    },
    {
        'hour': 2,
        'day_of_week': 6,  # Sunday
        'expected_severity': 'Low'
    }
]

HOUSING_TEST_DATA = [
    {
        'population_density': 1000,
        'average_income': 30000,
        'expected_affordability': 'Affordable'
    },
    {
        'population_density': 15000,
        'average_income': 120000,
        'expected_affordability': 'Not Affordable'
    }
]
```

### 3. Mocking and Stubbing

```python
# tests/test_mocks.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestWithMocks(unittest.TestCase):
    """Tests using mocks and stubs"""
    
    @patch('app.air_quality_model')
    def test_prediction_with_mock_model(self, mock_model):
        """Test prediction using mocked model"""
        # Setup mock
        mock_model.predict.return_value = np.array([45.5])
        
        # Test prediction
        from app import predict_air_quality
        result = predict_air_quality('United States', 'Developed', 2024, 6, 15)
        
        # Verify mock was called
        mock_model.predict.assert_called_once()
        
        # Verify result
        self.assertEqual(result['prediction'], 45.5)
        self.assertEqual(result['category'], 'Good')
    
    @patch('requests.post')
    def test_api_call_with_mock(self, mock_post):
        """Test API call using mocked requests"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Predicted AQI: 45.23 (Good)"
        mock_post.return_value = mock_response
        
        # Test API call
        import requests
        response = requests.post('http://localhost:5000/predict/air_quality', 
                               data={'Country': 'United States'})
        
        # Verify mock was called
        mock_post.assert_called_once()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted AQI', response.text)
```

### 4. Test Documentation

```python
# tests/test_documentation.py
"""
Test documentation examples for the Smart Urban Settlement Management System.

This module contains comprehensive tests that serve as documentation for how
to use the system's prediction APIs and verify their behavior.

Examples:
    >>> from tests.test_documentation import TestDocumentation
    >>> test = TestDocumentation()
    >>> test.test_air_quality_prediction_example()
    >>> test.test_traffic_prediction_example()
    >>> test.test_housing_prediction_example()
"""

import unittest
import requests

class TestDocumentation(unittest.TestCase):
    """
    Documentation tests that serve as examples for API usage.
    
    These tests demonstrate how to use the prediction APIs and what
    responses to expect. They serve as both tests and documentation.
    """
    
    def setUp(self):
        """Set up test fixtures for documentation examples."""
        self.base_url = "http://localhost:5000"
    
    def test_air_quality_prediction_example(self):
        """
        Example: How to predict air quality.
        
        This test demonstrates how to use the air quality prediction API
        and what response format to expect.
        
        Steps:
        1. Prepare input data with country, status, and date
        2. Send POST request to /predict/air_quality
        3. Parse the response to get AQI value and category
        
        Expected Response:
        - Status code: 200
        - Content: HTML with prediction result
        - Format: "Predicted AQI: {value} ({category})"
        """
        # Prepare test data
        data = {
            'Country': 'United States',
            'Status': 'Developed',
            'year': '2024',
            'month': '6',
            'day': '15'
        }
        
        # Make API call
        response = requests.post(f"{self.base_url}/predict/air_quality", data=data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted AQI', response.text)
        
        # Parse result (example of how to extract information)
        if 'Predicted AQI:' in response.text:
            # Extract AQI value and category
            result_text = response.text
            # This is a simplified example - in practice you'd use proper HTML parsing
            self.assertIn('(', result_text)  # Should contain category in parentheses
    
    def test_traffic_prediction_example(self):
        """
        Example: How to predict traffic congestion.
        
        This test demonstrates how to use the traffic prediction API
        and what response format to expect.
        
        Steps:
        1. Prepare input data with hour and day of week
        2. Send POST request to /predict/traffic
        3. Parse the response to get congestion level and severity
        
        Expected Response:
        - Status code: 200
        - Content: HTML with prediction result
        - Format: "Traffic Congestion Level: {value} ({severity})"
        """
        # Prepare test data
        data = {
            'hour': '14',  # 2 PM
            'day_of_week': '2'  # Wednesday
        }
        
        # Make API call
        response = requests.post(f"{self.base_url}/predict/traffic", data=data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn('Traffic Congestion Level', response.text)
    
    def test_housing_prediction_example(self):
        """
        Example: How to predict housing prices.
        
        This test demonstrates how to use the housing prediction API
        and what response format to expect.
        
        Steps:
        1. Prepare input data with population density and average income
        2. Send POST request to /predict/housing
        3. Parse the response to get predicted price and affordability
        
        Expected Response:
        - Status code: 200
        - Content: HTML with prediction result
        - Format: "Predicted Housing Price: {price} ({affordability})"
        """
        # Prepare test data
        data = {
            'population_density': '5000',
            'average_income': '75000'
        }
        
        # Make API call
        response = requests.post(f"{self.base_url}/predict/housing", data=data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn('Predicted Housing Price', response.text)

if __name__ == '__main__':
    unittest.main()
```

This comprehensive testing guide provides everything needed to implement thorough testing for your Smart Urban Settlement Management System. The guide covers unit testing, integration testing, performance testing, end-to-end testing, and test automation, making your project robust and reliable for the hackathon.
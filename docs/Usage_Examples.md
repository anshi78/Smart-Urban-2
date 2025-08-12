# Usage Examples & Tutorials

## Overview

This document provides comprehensive examples and tutorials for using the Smart Urban Settlement Management System. It includes practical scenarios, step-by-step guides, and advanced usage patterns for hackathon participants and urban planners.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Web Interface Tutorials](#web-interface-tutorials)
3. [API Usage Examples](#api-usage-examples)
4. [Python SDK Examples](#python-sdk-examples)
5. [Real-World Scenarios](#real-world-scenarios)
6. [Integration Examples](#integration-examples)
7. [Advanced Use Cases](#advanced-use-cases)

---

## Quick Start Examples

### 1. Basic Air Quality Prediction

#### Using Web Interface
1. Open your browser to `http://localhost:5000`
2. Navigate to the "Air Quality Prediction" section
3. Fill in the form:
   - **Country**: India
   - **Status**: Good
   - **Year**: 2024
   - **Month**: 7
   - **Day**: 15
4. Click "Get Prediction"
5. View the result: "Predicted AQI: 45.32 (Good)"

#### Using cURL
```bash
curl -X POST http://localhost:5000/predict/air_quality \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Country=India&Status=Good&year=2024&month=7&day=15"
```

### 2. Basic Traffic Prediction

#### Using Web Interface
1. Navigate to "Traffic Congestion Prediction" section
2. Fill in the form:
   - **Hour**: 17 (5 PM)
   - **Day of Week**: 4 (Friday)
3. Click "Get Prediction"
4. View result: "Traffic Congestion Level: 67.45 (High)"

#### Using cURL
```bash
curl -X POST http://localhost:5000/predict/traffic \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "hour=17&day_of_week=4"
```

### 3. Basic Housing Price Prediction

#### Using Web Interface
1. Navigate to "Affordable Housing Prediction" section
2. Fill in the form:
   - **Population Density**: 25000
   - **Average Income**: 75000
3. Click "Get Prediction"
4. View result: "Predicted Housing Price: 45000.00 (Affordable)"

#### Using cURL
```bash
curl -X POST http://localhost:5000/predict/housing \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "population_density=25000&average_income=75000"
```

---

## Web Interface Tutorials

### Tutorial 1: Urban Planning Assessment

**Scenario**: You're planning a new residential area and need to assess environmental and infrastructure conditions.

#### Step 1: Assess Air Quality
1. Navigate to Air Quality Prediction
2. Enter location details:
   - Country: "United States"
   - Status: "Moderate"
   - Year: 2024, Month: 8, Day: 1
3. Record the AQI prediction
4. Note the air quality category for planning decisions

#### Step 2: Analyze Traffic Patterns
1. Navigate to Traffic Congestion Prediction
2. Test different times:
   - **Morning rush**: Hour=8, Day=1 (Monday)
   - **Evening rush**: Hour=17, Day=1 (Monday)
   - **Weekend**: Hour=14, Day=6 (Saturday)
3. Compare congestion levels for optimal planning

#### Step 3: Evaluate Housing Affordability
1. Navigate to Housing Price Prediction
2. Test different demographic scenarios:
   - **Low density**: Population=15000, Income=50000
   - **High density**: Population=50000, Income=80000
3. Compare affordability for different income brackets

### Tutorial 2: Daily Operations Monitoring

**Scenario**: Monitor daily urban conditions for operational decisions.

#### Morning Assessment (7 AM)
```bash
# Check air quality for the day
curl -X POST http://localhost:5000/predict/air_quality \
  -d "Country=India&Status=Good&year=2024&month=7&day=15"

# Check traffic for morning commute
curl -X POST http://localhost:5000/predict/traffic \
  -d "hour=7&day_of_week=1"
```

#### Evening Planning (5 PM)
```bash
# Check evening traffic conditions
curl -X POST http://localhost:5000/predict/traffic \
  -d "hour=17&day_of_week=1"

# Assess housing market conditions
curl -X POST http://localhost:5000/predict/housing \
  -d "population_density=30000&average_income=65000"
```

---

## API Usage Examples

### 1. Python Requests Library

#### Complete Example Script
```python
import requests
import json
from datetime import datetime

class UrbanDataAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def predict_air_quality(self, country, status, year, month, day):
        """Predict air quality index"""
        url = f"{self.base_url}/predict/air_quality"
        data = {
            'Country': country,
            'Status': status,
            'year': year,
            'month': month,
            'day': day
        }
        
        response = requests.post(url, data=data)
        return self._parse_response(response, 'air_quality')
    
    def predict_traffic(self, hour, day_of_week):
        """Predict traffic congestion"""
        url = f"{self.base_url}/predict/traffic"
        data = {
            'hour': hour,
            'day_of_week': day_of_week
        }
        
        response = requests.post(url, data=data)
        return self._parse_response(response, 'traffic')
    
    def predict_housing(self, population_density, average_income):
        """Predict housing prices"""
        url = f"{self.base_url}/predict/housing"
        data = {
            'population_density': population_density,
            'average_income': average_income
        }
        
        response = requests.post(url, data=data)
        return self._parse_response(response, 'housing')
    
    def _parse_response(self, response, prediction_type):
        """Parse HTML response to extract prediction"""
        if response.status_code == 200:
            # Simple parsing - in practice, you might want to use BeautifulSoup
            text = response.text
            if prediction_type == 'air_quality' and 'Predicted AQI:' in text:
                # Extract AQI value and category
                import re
                match = re.search(r'Predicted AQI: ([\d.]+) \((\w+)\)', text)
                if match:
                    return {
                        'value': float(match.group(1)),
                        'category': match.group(2),
                        'type': 'air_quality'
                    }
            elif prediction_type == 'traffic' and 'Traffic Congestion Level:' in text:
                match = re.search(r'Traffic Congestion Level: ([\d.]+) \((\w+)\)', text)
                if match:
                    return {
                        'value': float(match.group(1)),
                        'level': match.group(2),
                        'type': 'traffic'
                    }
            elif prediction_type == 'housing' and 'Predicted Housing Price:' in text:
                match = re.search(r'Predicted Housing Price: ([\d.]+) \(([^)]+)\)', text)
                if match:
                    return {
                        'value': float(match.group(1)),
                        'affordability': match.group(2),
                        'type': 'housing'
                    }
        
        return {'error': f'Failed to get prediction: {response.status_code}'}

# Usage example
if __name__ == "__main__":
    api = UrbanDataAPI()
    
    # Predict air quality
    air_result = api.predict_air_quality('India', 'Good', 2024, 7, 15)
    print(f"Air Quality: {air_result}")
    
    # Predict traffic
    traffic_result = api.predict_traffic(17, 4)
    print(f"Traffic: {traffic_result}")
    
    # Predict housing
    housing_result = api.predict_housing(25000, 75000)
    print(f"Housing: {housing_result}")
```

### 2. JavaScript/Node.js Example

#### Using Fetch API
```javascript
class UrbanDataAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async predictAirQuality(country, status, year, month, day) {
        const url = `${this.baseUrl}/predict/air_quality`;
        const formData = new URLSearchParams({
            Country: country,
            Status: status,
            year: year,
            month: month,
            day: day
        });
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            });
            
            const html = await response.text();
            return this.parseAirQualityResponse(html);
        } catch (error) {
            return { error: error.message };
        }
    }
    
    async predictTraffic(hour, dayOfWeek) {
        const url = `${this.baseUrl}/predict/traffic`;
        const formData = new URLSearchParams({
            hour: hour,
            day_of_week: dayOfWeek
        });
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            });
            
            const html = await response.text();
            return this.parseTrafficResponse(html);
        } catch (error) {
            return { error: error.message };
        }
    }
    
    async predictHousing(populationDensity, averageIncome) {
        const url = `${this.baseUrl}/predict/housing`;
        const formData = new URLSearchParams({
            population_density: populationDensity,
            average_income: averageIncome
        });
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            });
            
            const html = await response.text();
            return this.parseHousingResponse(html);
        } catch (error) {
            return { error: error.message };
        }
    }
    
    parseAirQualityResponse(html) {
        const match = html.match(/Predicted AQI: ([\d.]+) \((\w+)\)/);
        if (match) {
            return {
                value: parseFloat(match[1]),
                category: match[2],
                type: 'air_quality'
            };
        }
        return { error: 'Could not parse air quality response' };
    }
    
    parseTrafficResponse(html) {
        const match = html.match(/Traffic Congestion Level: ([\d.]+) \((\w+)\)/);
        if (match) {
            return {
                value: parseFloat(match[1]),
                level: match[2],
                type: 'traffic'
            };
        }
        return { error: 'Could not parse traffic response' };
    }
    
    parseHousingResponse(html) {
        const match = html.match(/Predicted Housing Price: ([\d.]+) \(([^)]+)\)/);
        if (match) {
            return {
                value: parseFloat(match[1]),
                affordability: match[2],
                type: 'housing'
            };
        }
        return { error: 'Could not parse housing response' };
    }
}

// Usage example
async function main() {
    const api = new UrbanDataAPI();
    
    try {
        // Predict air quality
        const airResult = await api.predictAirQuality('India', 'Good', 2024, 7, 15);
        console.log('Air Quality:', airResult);
        
        // Predict traffic
        const trafficResult = await api.predictTraffic(17, 4);
        console.log('Traffic:', trafficResult);
        
        // Predict housing
        const housingResult = await api.predictHousing(25000, 75000);
        console.log('Housing:', housingResult);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

---

## Python SDK Examples

### 1. Direct Model Access

#### Advanced Python Integration
```python
import sys
sys.path.append('src')

from air_quality import predict_aqi
from traffic_congestion import predict_traffic_congestion
from house_price import predict_housing_demand
import pandas as pd

class SmartUrbanSDK:
    """Python SDK for Smart Urban Settlement Management"""
    
    def __init__(self):
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load all ML models"""
        try:
            # Models are loaded automatically when importing modules
            self.models_loaded = True
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def get_air_quality_prediction(self, country, status, year, month, day):
        """Get air quality prediction with enhanced features"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            aqi_value, aqi_category = predict_aqi(country, status, year, month, day)
            
            # Enhanced response with recommendations
            recommendations = self._get_air_quality_recommendations(aqi_value, aqi_category)
            
            return {
                "aqi_value": round(aqi_value, 2),
                "category": aqi_category,
                "recommendations": recommendations,
                "health_impact": self._get_health_impact(aqi_value),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_traffic_prediction(self, hour, day_of_week):
        """Get traffic prediction with enhanced features"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            congestion_level = predict_traffic_congestion(hour, day_of_week)
            category = "High" if congestion_level >= 50 else "Low"
            
            # Enhanced response with travel recommendations
            recommendations = self._get_traffic_recommendations(hour, day_of_week, congestion_level)
            
            return {
                "congestion_level": round(congestion_level, 2),
                "category": category,
                "recommendations": recommendations,
                "alternative_routes": self._suggest_alternative_times(hour, day_of_week),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_housing_prediction(self, population_density, average_income):
        """Get housing prediction with enhanced features"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            # Create DataFrame for prediction
            new_data = pd.DataFrame({
                'population_density': [population_density],
                'avg_area_income': [average_income]
            })
            
            prices, affordability = predict_housing_demand(new_data)
            
            if prices is not None and len(prices) > 0:
                price = prices[0]
                afford_status = affordability[0]
                
                # Enhanced response with market analysis
                market_analysis = self._get_market_analysis(population_density, average_income, price)
                
                return {
                    "predicted_price": round(price, 2),
                    "affordability": afford_status,
                    "market_analysis": market_analysis,
                    "investment_potential": self._assess_investment_potential(price, average_income),
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            else:
                return {"error": "Prediction failed"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_comprehensive_assessment(self, location_data):
        """Get comprehensive urban assessment"""
        results = {}
        
        # Air quality assessment
        if all(key in location_data for key in ['country', 'status', 'year', 'month', 'day']):
            results['air_quality'] = self.get_air_quality_prediction(
                location_data['country'], location_data['status'],
                location_data['year'], location_data['month'], location_data['day']
            )
        
        # Traffic assessment for different times
        if 'assess_traffic' in location_data:
            results['traffic'] = {}
            peak_hours = [7, 8, 17, 18]  # Morning and evening rush
            for hour in peak_hours:
                for day in range(5):  # Weekdays
                    key = f"weekday_{hour}h"
                    results['traffic'][key] = self.get_traffic_prediction(hour, day)
        
        # Housing assessment
        if all(key in location_data for key in ['population_density', 'average_income']):
            results['housing'] = self.get_housing_prediction(
                location_data['population_density'], location_data['average_income']
            )
        
        # Overall livability score
        results['livability_score'] = self._calculate_livability_score(results)
        
        return results
    
    def _get_air_quality_recommendations(self, aqi_value, category):
        """Generate air quality recommendations"""
        if category == "Good":
            return [
                "Excellent air quality conditions",
                "Perfect for outdoor activities",
                "No health precautions needed"
            ]
        elif category == "Moderate":
            return [
                "Acceptable air quality for most people",
                "Sensitive individuals should limit prolonged outdoor exertion",
                "Consider indoor activities during peak pollution hours"
            ]
        else:  # Bad
            return [
                "Poor air quality conditions",
                "Everyone should limit outdoor activities",
                "Use air purifiers indoors",
                "Wear masks when going outside"
            ]
    
    def _get_health_impact(self, aqi_value):
        """Assess health impact based on AQI"""
        if aqi_value <= 50:
            return "Minimal health impact"
        elif aqi_value <= 100:
            return "Minor health concerns for sensitive groups"
        elif aqi_value <= 150:
            return "Health concerns for sensitive groups"
        else:
            return "Significant health concerns for all groups"
    
    def _get_traffic_recommendations(self, hour, day_of_week, congestion_level):
        """Generate traffic recommendations"""
        recommendations = []
        
        if congestion_level >= 70:
            recommendations.extend([
                "Heavy traffic expected",
                "Consider using public transportation",
                "Allow extra travel time",
                "Work from home if possible"
            ])
        elif congestion_level >= 50:
            recommendations.extend([
                "Moderate traffic expected",
                "Consider alternative routes",
                "Allow additional 15-20 minutes"
            ])
        else:
            recommendations.extend([
                "Light traffic conditions",
                "Good time for travel",
                "Normal travel times expected"
            ])
        
        return recommendations
    
    def _suggest_alternative_times(self, hour, day_of_week):
        """Suggest alternative travel times"""
        # Simple logic - suggest earlier or later times
        alternatives = []
        
        if 7 <= hour <= 9:  # Morning rush
            alternatives = [6, 10, 11]
        elif 17 <= hour <= 19:  # Evening rush
            alternatives = [15, 16, 20, 21]
        else:
            alternatives = [hour]
        
        return [f"{alt}:00" for alt in alternatives]
    
    def _get_market_analysis(self, population_density, average_income, predicted_price):
        """Analyze housing market conditions"""
        analysis = {}
        
        # Price per income ratio
        price_income_ratio = predicted_price / average_income
        
        if price_income_ratio <= 3:
            analysis['affordability_level'] = "Highly Affordable"
        elif price_income_ratio <= 5:
            analysis['affordability_level'] = "Moderately Affordable"
        else:
            analysis['affordability_level'] = "Low Affordability"
        
        # Density analysis
        if population_density > 40000:
            analysis['area_type'] = "High Density Urban"
            analysis['characteristics'] = ["Urban amenities", "Public transport access", "Higher prices"]
        elif population_density > 20000:
            analysis['area_type'] = "Medium Density Suburban"
            analysis['characteristics'] = ["Balanced lifestyle", "Moderate prices", "Good infrastructure"]
        else:
            analysis['area_type'] = "Low Density Rural"
            analysis['characteristics'] = ["Peaceful environment", "Lower prices", "Limited amenities"]
        
        return analysis
    
    def _assess_investment_potential(self, price, income):
        """Assess investment potential"""
        ratio = price / income
        
        if ratio < 2:
            return "High - Undervalued market"
        elif ratio < 4:
            return "Moderate - Balanced market"
        else:
            return "Low - Overvalued market"
    
    def _calculate_livability_score(self, results):
        """Calculate overall livability score"""
        score = 0
        factors = 0
        
        # Air quality score
        if 'air_quality' in results and 'aqi_value' in results['air_quality']:
            aqi = results['air_quality']['aqi_value']
            if aqi <= 50:
                score += 30
            elif aqi <= 100:
                score += 20
            else:
                score += 10
            factors += 1
        
        # Traffic score
        if 'traffic' in results:
            # Average traffic congestion
            traffic_scores = []
            for traffic_data in results['traffic'].values():
                if 'congestion_level' in traffic_data:
                    level = traffic_data['congestion_level']
                    if level < 30:
                        traffic_scores.append(30)
                    elif level < 60:
                        traffic_scores.append(20)
                    else:
                        traffic_scores.append(10)
            
            if traffic_scores:
                score += sum(traffic_scores) / len(traffic_scores)
                factors += 1
        
        # Housing score
        if 'housing' in results and 'affordability' in results['housing']:
            if results['housing']['affordability'] == "Affordable":
                score += 40
            else:
                score += 20
            factors += 1
        
        if factors > 0:
            return round(score / factors, 1)
        else:
            return 0

# Usage Examples
if __name__ == "__main__":
    sdk = SmartUrbanSDK()
    
    # Example 1: Single predictions
    print("=== Air Quality Prediction ===")
    air_result = sdk.get_air_quality_prediction('India', 'Good', 2024, 7, 15)
    print(json.dumps(air_result, indent=2))
    
    print("\n=== Traffic Prediction ===")
    traffic_result = sdk.get_traffic_prediction(17, 4)
    print(json.dumps(traffic_result, indent=2))
    
    print("\n=== Housing Prediction ===")
    housing_result = sdk.get_housing_prediction(25000, 75000)
    print(json.dumps(housing_result, indent=2))
    
    # Example 2: Comprehensive assessment
    print("\n=== Comprehensive Assessment ===")
    location_data = {
        'country': 'India',
        'status': 'Good',
        'year': 2024,
        'month': 7,
        'day': 15,
        'population_density': 25000,
        'average_income': 75000,
        'assess_traffic': True
    }
    
    comprehensive_result = sdk.get_comprehensive_assessment(location_data)
    print(json.dumps(comprehensive_result, indent=2))
```

---

## Real-World Scenarios

### Scenario 1: New Residential Development Planning

**Context**: A city planner wants to assess the viability of a new residential development.

#### Step-by-Step Analysis

```python
# Development planning assessment
def assess_development_site(sdk, site_data):
    """Comprehensive site assessment for development"""
    
    assessment = {}
    
    # 1. Air Quality Assessment
    print("Analyzing air quality conditions...")
    air_quality = sdk.get_air_quality_prediction(
        site_data['country'], 
        site_data['current_air_status'],
        2024, 7, 15  # Current date
    )
    assessment['air_quality'] = air_quality
    
    # 2. Traffic Impact Analysis
    print("Analyzing traffic patterns...")
    traffic_analysis = {}
    
    # Peak hours analysis
    peak_times = [
        (7, "morning_rush"), (8, "morning_rush"),
        (17, "evening_rush"), (18, "evening_rush")
    ]
    
    for hour, period in peak_times:
        for day in range(5):  # Weekdays
            key = f"{period}_day{day}_hour{hour}"
            traffic_analysis[key] = sdk.get_traffic_prediction(hour, day)
    
    assessment['traffic_analysis'] = traffic_analysis
    
    # 3. Housing Market Analysis
    print("Analyzing housing market...")
    housing_scenarios = []
    
    # Different income brackets
    income_levels = [40000, 60000, 80000, 100000]
    for income in income_levels:
        housing_result = sdk.get_housing_prediction(
            site_data['population_density'], 
            income
        )
        housing_scenarios.append({
            'income_level': income,
            'prediction': housing_result
        })
    
    assessment['housing_scenarios'] = housing_scenarios
    
    # 4. Generate recommendations
    assessment['recommendations'] = generate_development_recommendations(assessment)
    
    return assessment

def generate_development_recommendations(assessment):
    """Generate development recommendations based on assessment"""
    recommendations = []
    
    # Air quality recommendations
    air_quality = assessment.get('air_quality', {})
    if air_quality.get('category') == 'Good':
        recommendations.append("✓ Excellent air quality - suitable for residential development")
    elif air_quality.get('category') == 'Moderate':
        recommendations.append("⚠ Moderate air quality - consider air filtration systems")
    else:
        recommendations.append("✗ Poor air quality - reconsider location or implement mitigation")
    
    # Traffic recommendations
    traffic_data = assessment.get('traffic_analysis', {})
    high_congestion_count = sum(1 for data in traffic_data.values() 
                               if data.get('category') == 'High')
    
    if high_congestion_count > len(traffic_data) * 0.7:
        recommendations.append("✗ High traffic congestion - improve public transport access")
    elif high_congestion_count > len(traffic_data) * 0.3:
        recommendations.append("⚠ Moderate traffic - plan for adequate parking and transit")
    else:
        recommendations.append("✓ Low traffic congestion - good accessibility")
    
    # Housing affordability recommendations
    housing_scenarios = assessment.get('housing_scenarios', [])
    affordable_count = sum(1 for scenario in housing_scenarios 
                          if scenario['prediction'].get('affordability') == 'Affordable')
    
    if affordable_count >= len(housing_scenarios) * 0.7:
        recommendations.append("✓ Highly affordable area - good for diverse income groups")
    elif affordable_count >= len(housing_scenarios) * 0.3:
        recommendations.append("⚠ Moderately affordable - consider mixed-income housing")
    else:
        recommendations.append("✗ Low affordability - focus on luxury or assisted housing")
    
    return recommendations

# Usage example
sdk = SmartUrbanSDK()

site_data = {
    'country': 'India',
    'current_air_status': 'Good',
    'population_density': 30000,
    'location_name': 'Proposed Residential Site A'
}

development_assessment = assess_development_site(sdk, site_data)
print("Development Assessment Results:")
print(json.dumps(development_assessment, indent=2))
```

### Scenario 2: Daily Urban Operations Dashboard

**Context**: City operations center needs real-time urban condition monitoring.

#### Dashboard Implementation

```python
import time
from datetime import datetime, timedelta

class UrbanOperationsDashboard:
    """Real-time urban operations monitoring"""
    
    def __init__(self, sdk):
        self.sdk = sdk
        self.monitoring_areas = []
        self.alerts = []
    
    def add_monitoring_area(self, area_config):
        """Add area to monitoring list"""
        self.monitoring_areas.append(area_config)
    
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle for all areas"""
        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'areas': {},
            'alerts': [],
            'summary': {}
        }
        
        for area in self.monitoring_areas:
            area_name = area['name']
            print(f"Monitoring {area_name}...")
            
            area_assessment = self._assess_area(area)
            cycle_results['areas'][area_name] = area_assessment
            
            # Check for alerts
            alerts = self._check_alerts(area_name, area_assessment)
            cycle_results['alerts'].extend(alerts)
        
        # Generate summary
        cycle_results['summary'] = self._generate_summary(cycle_results['areas'])
        
        return cycle_results
    
    def _assess_area(self, area_config):
        """Assess single area"""
        current_time = datetime.now()
        
        assessment = {
            'air_quality': None,
            'traffic': None,
            'housing': None,
            'status': 'unknown'
        }
        
        # Air quality assessment
        if 'air_quality' in area_config:
            aq_config = area_config['air_quality']
            assessment['air_quality'] = self.sdk.get_air_quality_prediction(
                aq_config['country'],
                aq_config['status'],
                current_time.year,
                current_time.month,
                current_time.day
            )
        
        # Traffic assessment
        if 'monitor_traffic' in area_config and area_config['monitor_traffic']:
            assessment['traffic'] = self.sdk.get_traffic_prediction(
                current_time.hour,
                current_time.weekday()
            )
        
        # Housing market check (daily)
        if 'housing' in area_config:
            housing_config = area_config['housing']
            assessment['housing'] = self.sdk.get_housing_prediction(
                housing_config['population_density'],
                housing_config['average_income']
            )
        
        # Overall status
        assessment['status'] = self._determine_area_status(assessment)
        
        return assessment
    
    def _determine_area_status(self, assessment):
        """Determine overall area status"""
        status_scores = []
        
        # Air quality status
        if assessment['air_quality']:
            aqi_category = assessment['air_quality'].get('category', 'Unknown')
            if aqi_category == 'Good':
                status_scores.append(3)
            elif aqi_category == 'Moderate':
                status_scores.append(2)
            else:
                status_scores.append(1)
        
        # Traffic status
        if assessment['traffic']:
            traffic_category = assessment['traffic'].get('category', 'Unknown')
            if traffic_category == 'Low':
                status_scores.append(3)
            else:
                status_scores.append(1)
        
        # Overall status
        if not status_scores:
            return 'unknown'
        
        avg_score = sum(status_scores) / len(status_scores)
        
        if avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'moderate'
        else:
            return 'poor'
    
    def _check_alerts(self, area_name, assessment):
        """Check for alert conditions"""
        alerts = []
        
        # Air quality alerts
        if assessment['air_quality']:
            aqi_value = assessment['air_quality'].get('aqi_value', 0)
            if aqi_value > 100:
                alerts.append({
                    'type': 'air_quality',
                    'severity': 'high',
                    'area': area_name,
                    'message': f'Poor air quality detected: AQI {aqi_value}',
                    'timestamp': datetime.now().isoformat()
                })
            elif aqi_value > 50:
                alerts.append({
                    'type': 'air_quality',
                    'severity': 'medium',
                    'area': area_name,
                    'message': f'Moderate air quality: AQI {aqi_value}',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Traffic alerts
        if assessment['traffic']:
            congestion_level = assessment['traffic'].get('congestion_level', 0)
            if congestion_level > 70:
                alerts.append({
                    'type': 'traffic',
                    'severity': 'high',
                    'area': area_name,
                    'message': f'Heavy traffic congestion: Level {congestion_level}',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _generate_summary(self, areas_data):
        """Generate monitoring summary"""
        summary = {
            'total_areas': len(areas_data),
            'status_distribution': {'good': 0, 'moderate': 0, 'poor': 0, 'unknown': 0},
            'average_aqi': 0,
            'average_traffic': 0,
            'recommendations': []
        }
        
        aqi_values = []
        traffic_values = []
        
        for area_data in areas_data.values():
            status = area_data.get('status', 'unknown')
            summary['status_distribution'][status] += 1
            
            if area_data.get('air_quality') and 'aqi_value' in area_data['air_quality']:
                aqi_values.append(area_data['air_quality']['aqi_value'])
            
            if area_data.get('traffic') and 'congestion_level' in area_data['traffic']:
                traffic_values.append(area_data['traffic']['congestion_level'])
        
        # Calculate averages
        if aqi_values:
            summary['average_aqi'] = round(sum(aqi_values) / len(aqi_values), 2)
        
        if traffic_values:
            summary['average_traffic'] = round(sum(traffic_values) / len(traffic_values), 2)
        
        # Generate recommendations
        if summary['average_aqi'] > 75:
            summary['recommendations'].append("Consider air quality improvement measures")
        
        if summary['average_traffic'] > 60:
            summary['recommendations'].append("Implement traffic management strategies")
        
        return summary

# Usage example
sdk = SmartUrbanSDK()
dashboard = UrbanOperationsDashboard(sdk)

# Add monitoring areas
dashboard.add_monitoring_area({
    'name': 'Downtown District',
    'air_quality': {'country': 'India', 'status': 'Moderate'},
    'monitor_traffic': True,
    'housing': {'population_density': 45000, 'average_income': 65000}
})

dashboard.add_monitoring_area({
    'name': 'Residential Zone A',
    'air_quality': {'country': 'India', 'status': 'Good'},
    'monitor_traffic': True,
    'housing': {'population_density': 25000, 'average_income': 55000}
})

dashboard.add_monitoring_area({
    'name': 'Industrial Area',
    'air_quality': {'country': 'India', 'status': 'Poor'},
    'monitor_traffic': True
})

# Run monitoring cycle
monitoring_results = dashboard.run_monitoring_cycle()
print("Urban Operations Dashboard Results:")
print(json.dumps(monitoring_results, indent=2))
```

### Scenario 3: Environmental Impact Assessment

**Context**: Environmental consultant needs to assess the impact of a proposed development on urban conditions.

```python
def environmental_impact_assessment(sdk, baseline_data, proposed_changes):
    """Comprehensive environmental impact assessment"""
    
    assessment = {
        'baseline': {},
        'projected': {},
        'impact_analysis': {},
        'mitigation_recommendations': []
    }
    
    # Baseline assessment
    print("Establishing baseline conditions...")
    assessment['baseline'] = {
        'air_quality': sdk.get_air_quality_prediction(
            baseline_data['country'],
            baseline_data['current_air_status'],
            2024, 7, 15
        ),
        'traffic': sdk.get_traffic_prediction(
            baseline_data['peak_hour'],
            baseline_data['peak_day']
        ),
        'housing': sdk.get_housing_prediction(
            baseline_data['current_population_density'],
            baseline_data['current_average_income']
        )
    }
    
    # Projected conditions after development
    print("Projecting post-development conditions...")
    projected_density = baseline_data['current_population_density'] + proposed_changes['population_increase']
    projected_income = baseline_data['current_average_income'] * (1 + proposed_changes['income_change_percent'] / 100)
    
    # Air quality might change due to increased activity
    projected_air_status = baseline_data['current_air_status']
    if proposed_changes['development_type'] == 'industrial':
        if projected_air_status == 'Good':
            projected_air_status = 'Moderate'
        elif projected_air_status == 'Moderate':
            projected_air_status = 'Poor'
    
    assessment['projected'] = {
        'air_quality': sdk.get_air_quality_prediction(
            baseline_data['country'],
            projected_air_status,
            2024, 7, 15
        ),
        'traffic': sdk.get_traffic_prediction(
            baseline_data['peak_hour'],
            baseline_data['peak_day']
        ),  # Traffic model doesn't account for density changes directly
        'housing': sdk.get_housing_prediction(
            projected_density,
            projected_income
        )
    }
    
    # Impact analysis
    assessment['impact_analysis'] = analyze_impacts(assessment['baseline'], assessment['projected'])
    
    # Mitigation recommendations
    assessment['mitigation_recommendations'] = generate_mitigation_recommendations(
        assessment['impact_analysis'], 
        proposed_changes
    )
    
    return assessment

def analyze_impacts(baseline, projected):
    """Analyze the differences between baseline and projected conditions"""
    impacts = {}
    
    # Air quality impact
    baseline_aqi = baseline['air_quality'].get('aqi_value', 0)
    projected_aqi = projected['air_quality'].get('aqi_value', 0)
    
    impacts['air_quality'] = {
        'baseline_aqi': baseline_aqi,
        'projected_aqi': projected_aqi,
        'change': projected_aqi - baseline_aqi,
        'impact_level': 'positive' if projected_aqi < baseline_aqi else 'negative' if projected_aqi > baseline_aqi else 'neutral'
    }
    
    # Housing market impact
    baseline_price = baseline['housing'].get('predicted_price', 0)
    projected_price = projected['housing'].get('predicted_price', 0)
    
    impacts['housing'] = {
        'baseline_price': baseline_price,
        'projected_price': projected_price,
        'price_change_percent': ((projected_price - baseline_price) / baseline_price * 100) if baseline_price > 0 else 0,
        'affordability_change': {
            'baseline': baseline['housing'].get('affordability', 'Unknown'),
            'projected': projected['housing'].get('affordability', 'Unknown')
        }
    }
    
    # Traffic impact (simplified - would need more complex modeling in reality)
    baseline_traffic = baseline['traffic'].get('congestion_level', 0)
    projected_traffic = projected['traffic'].get('congestion_level', 0)
    
    impacts['traffic'] = {
        'baseline_congestion': baseline_traffic,
        'projected_congestion': projected_traffic,
        'estimated_increase': 10,  # Simplified assumption
        'impact_level': 'moderate'
    }
    
    return impacts

def generate_mitigation_recommendations(impacts, proposed_changes):
    """Generate mitigation recommendations based on impact analysis"""
    recommendations = []
    
    # Air quality mitigation
    if impacts['air_quality']['impact_level'] == 'negative':
        recommendations.extend([
            "Implement air quality monitoring systems",
            "Require green building standards",
            "Increase green spaces by 20%",
            "Consider electric vehicle charging infrastructure"
        ])
    
    # Housing market mitigation
    price_change = impacts['housing']['price_change_percent']
    if price_change > 10:
        recommendations.extend([
            "Include affordable housing units (20% minimum)",
            "Implement inclusionary zoning policies",
            "Provide first-time buyer assistance programs"
        ])
    
    # Traffic mitigation
    if impacts['traffic']['impact_level'] in ['moderate', 'high']:
        recommendations.extend([
            "Improve public transportation access",
            "Implement traffic impact fees",
            "Design pedestrian and cycling infrastructure",
            "Consider traffic calming measures"
        ])
    
    # Development-specific recommendations
    if proposed_changes['development_type'] == 'residential':
        recommendations.extend([
            "Ensure adequate parking (1.5 spaces per unit)",
            "Include community amenities",
            "Plan for school capacity increases"
        ])
    elif proposed_changes['development_type'] == 'commercial':
        recommendations.extend([
            "Implement stormwater management",
            "Require LEED certification",
            "Plan for increased service delivery needs"
        ])
    
    return recommendations

# Usage example
sdk = SmartUrbanSDK()

baseline_data = {
    'country': 'India',
    'current_air_status': 'Good',
    'current_population_density': 20000,
    'current_average_income': 60000,
    'peak_hour': 17,
    'peak_day': 4
}

proposed_changes = {
    'development_type': 'residential',
    'population_increase': 5000,
    'income_change_percent': 15,
    'development_size_acres': 50
}

eia_results = environmental_impact_assessment(sdk, baseline_data, proposed_changes)
print("Environmental Impact Assessment:")
print(json.dumps(eia_results, indent=2))
```

---

## Integration Examples

### 1. Integration with GIS Systems

```python
import geopandas as gpd
from shapely.geometry import Point
import folium

class GISIntegration:
    """Integration with Geographic Information Systems"""
    
    def __init__(self, sdk):
        self.sdk = sdk
        self.map_data = {}
    
    def create_prediction_map(self, coordinates_list, map_center, zoom_start=10):
        """Create interactive map with predictions"""
        
        # Create base map
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        for coord_data in coordinates_list:
            lat, lon = coord_data['lat'], coord_data['lon']
            
            # Get predictions for this location
            predictions = self._get_location_predictions(coord_data)
            
            # Create popup content
            popup_content = self._create_popup_content(predictions, coord_data)
            
            # Determine marker color based on conditions
            marker_color = self._get_marker_color(predictions)
            
            # Add marker to map
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Location: {coord_data.get('name', 'Unknown')}",
                icon=folium.Icon(color=marker_color)
            ).add_to(m)
        
        return m
    
    def _get_location_predictions(self, coord_data):
        """Get all predictions for a location"""
        predictions = {}
        
        # Air quality prediction
        if 'air_quality_data' in coord_data:
            aq_data = coord_data['air_quality_data']
            predictions['air_quality'] = self.sdk.get_air_quality_prediction(
                aq_data['country'], aq_data['status'],
                aq_data['year'], aq_data['month'], aq_data['day']
            )
        
        # Traffic prediction (for peak hour)
        predictions['traffic'] = self.sdk.get_traffic_prediction(17, 4)  # Friday 5 PM
        
        # Housing prediction
        if 'demographics' in coord_data:
            demo = coord_data['demographics']
            predictions['housing'] = self.sdk.get_housing_prediction(
                demo['population_density'], demo['average_income']
            )
        
        return predictions
    
    def _create_popup_content(self, predictions, coord_data):
        """Create HTML content for map popup"""
        content = f"<h4>{coord_data.get('name', 'Location')}</h4>"
        
        if 'air_quality' in predictions and 'aqi_value' in predictions['air_quality']:
            aq = predictions['air_quality']
            content += f"""
            <p><strong>Air Quality:</strong><br>
            AQI: {aq['aqi_value']} ({aq['category']})</p>
            """
        
        if 'traffic' in predictions and 'congestion_level' in predictions['traffic']:
            traffic = predictions['traffic']
            content += f"""
            <p><strong>Traffic (Peak Hour):</strong><br>
            Congestion: {traffic['congestion_level']} ({traffic['category']})</p>
            """
        
        if 'housing' in predictions and 'predicted_price' in predictions['housing']:
            housing = predictions['housing']
            content += f"""
            <p><strong>Housing:</strong><br>
            Price: ${housing['predicted_price']:,.0f} ({housing['affordability']})</p>
            """
        
        return content
    
    def _get_marker_color(self, predictions):
        """Determine marker color based on overall conditions"""
        score = 0
        factors = 0
        
        # Air quality score
        if 'air_quality' in predictions and 'category' in predictions['air_quality']:
            category = predictions['air_quality']['category']
            if category == 'Good':
                score += 3
            elif category == 'Moderate':
                score += 2
            else:
                score += 1
            factors += 1
        
        # Traffic score
        if 'traffic' in predictions and 'category' in predictions['traffic']:
            category = predictions['traffic']['category']
            if category == 'Low':
                score += 3
            else:
                score += 1
            factors += 1
        
        # Housing score
        if 'housing' in predictions and 'affordability' in predictions['housing']:
            affordability = predictions['housing']['affordability']
            if affordability == 'Affordable':
                score += 3
            else:
                score += 1
            factors += 1
        
        if factors == 0:
            return 'gray'
        
        avg_score = score / factors
        
        if avg_score >= 2.5:
            return 'green'
        elif avg_score >= 2:
            return 'orange'
        else:
            return 'red'

# Usage example
sdk = SmartUrbanSDK()
gis_integration = GISIntegration(sdk)

# Sample locations with coordinates and data
locations = [
    {
        'name': 'Downtown Area',
        'lat': 28.6139, 'lon': 77.2090,  # New Delhi coordinates
        'air_quality_data': {'country': 'India', 'status': 'Moderate', 'year': 2024, 'month': 7, 'day': 15},
        'demographics': {'population_density': 45000, 'average_income': 70000}
    },
    {
        'name': 'Residential Zone',
        'lat': 28.7041, 'lon': 77.1025,
        'air_quality_data': {'country': 'India', 'status': 'Good', 'year': 2024, 'month': 7, 'day': 15},
        'demographics': {'population_density': 25000, 'average_income': 55000}
    },
    {
        'name': 'Industrial Area',
        'lat': 28.5355, 'lon': 77.3910,
        'air_quality_data': {'country': 'India', 'status': 'Poor', 'year': 2024, 'month': 7, 'day': 15},
        'demographics': {'population_density': 35000, 'average_income': 45000}
    }
]

# Create interactive map
prediction_map = gis_integration.create_prediction_map(
    locations, 
    map_center=[28.6139, 77.2090], 
    zoom_start=11
)

# Save map
prediction_map.save('urban_predictions_map.html')
print("Interactive map saved as 'urban_predictions_map.html'")
```

### 2. Integration with Data Pipelines

```python
import schedule
import time
import logging
from datetime import datetime
import sqlite3
import pandas as pd

class DataPipelineIntegration:
    """Integration with automated data pipelines"""
    
    def __init__(self, sdk, db_path='urban_predictions.db'):
        self.sdk = sdk
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
    
    def setup_database(self):
        """Initialize database for storing predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location_id TEXT,
                prediction_type TEXT,
                input_data TEXT,
                output_data TEXT,
                confidence_score REAL
            )
        ''')
        
        # Create locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                location_id TEXT PRIMARY KEY,
                name TEXT,
                latitude REAL,
                longitude REAL,
                configuration TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_logging(self):
        """Setup logging for pipeline operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('urban_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_monitoring_location(self, location_id, name, latitude, longitude, config):
        """Add location to monitoring pipeline"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO locations 
            (location_id, name, latitude, longitude, configuration)
            VALUES (?, ?, ?, ?, ?)
        ''', (location_id, name, latitude, longitude, json.dumps(config)))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Added monitoring location: {name} ({location_id})")
    
    def run_prediction_pipeline(self):
        """Run prediction pipeline for all locations"""
        self.logger.info("Starting prediction pipeline...")
        
        # Get all monitoring locations
        conn = sqlite3.connect(self.db_path)
        locations_df = pd.read_sql_query('SELECT * FROM locations', conn)
        conn.close()
        
        predictions_made = 0
        
        for _, location in locations_df.iterrows():
            try:
                config = json.loads(location['configuration'])
                location_id = location['location_id']
                
                # Run predictions for this location
                location_predictions = self._run_location_predictions(location_id, config)
                
                # Store predictions in database
                self._store_predictions(location_id, location_predictions)
                
                predictions_made += len(location_predictions)
                
            except Exception as e:
                self.logger.error(f"Error processing location {location['location_id']}: {e}")
        
        self.logger.info(f"Pipeline completed. Made {predictions_made} predictions.")
    
    def _run_location_predictions(self, location_id, config):
        """Run all configured predictions for a location"""
        predictions = []
        current_time = datetime.now()
        
        # Air quality prediction
        if 'air_quality' in config:
            aq_config = config['air_quality']
            try:
                result = self.sdk.get_air_quality_prediction(
                    aq_config['country'],
                    aq_config['status'],
                    current_time.year,
                    current_time.month,
                    current_time.day
                )
                
                predictions.append({
                    'type': 'air_quality',
                    'input': aq_config,
                    'output': result,
                    'confidence': self._calculate_confidence(result, 'air_quality')
                })
                
            except Exception as e:
                self.logger.error(f"Air quality prediction failed for {location_id}: {e}")
        
        # Traffic prediction
        if 'traffic' in config and config['traffic']:
            try:
                result = self.sdk.get_traffic_prediction(
                    current_time.hour,
                    current_time.weekday()
                )
                
                predictions.append({
                    'type': 'traffic',
                    'input': {'hour': current_time.hour, 'day_of_week': current_time.weekday()},
                    'output': result,
                    'confidence': self._calculate_confidence(result, 'traffic')
                })
                
            except Exception as e:
                self.logger.error(f"Traffic prediction failed for {location_id}: {e}")
        
        # Housing prediction
        if 'housing' in config:
            housing_config = config['housing']
            try:
                result = self.sdk.get_housing_prediction(
                    housing_config['population_density'],
                    housing_config['average_income']
                )
                
                predictions.append({
                    'type': 'housing',
                    'input': housing_config,
                    'output': result,
                    'confidence': self._calculate_confidence(result, 'housing')
                })
                
            except Exception as e:
                self.logger.error(f"Housing prediction failed for {location_id}: {e}")
        
        return predictions
    
    def _calculate_confidence(self, result, prediction_type):
        """Calculate confidence score for prediction"""
        # Simplified confidence calculation
        if 'error' in result:
            return 0.0
        
        # Base confidence scores
        base_confidence = {
            'air_quality': 0.85,
            'traffic': 0.75,
            'housing': 0.80
        }
        
        return base_confidence.get(prediction_type, 0.70)
    
    def _store_predictions(self, location_id, predictions):
        """Store predictions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for prediction in predictions:
            cursor.execute('''
                INSERT INTO predictions 
                (location_id, prediction_type, input_data, output_data, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                location_id,
                prediction['type'],
                json.dumps(prediction['input']),
                json.dumps(prediction['output']),
                prediction['confidence']
            ))
        
        conn.commit()
        conn.close()
    
    def generate_daily_report(self):
        """Generate daily summary report"""
        self.logger.info("Generating daily report...")
        
        # Get today's predictions
        conn = sqlite3.connect(self.db_path)
        today = datetime.now().strftime('%Y-%m-%d')
        
        query = '''
            SELECT p.*, l.name as location_name
            FROM predictions p
            JOIN locations l ON p.location_id = l.location_id
            WHERE DATE(p.timestamp) = ?
        '''
        
        predictions_df = pd.read_sql_query(query, conn, params=[today])
        conn.close()
        
        if predictions_df.empty:
            self.logger.info("No predictions found for today.")
            return
        
        # Generate summary statistics
        summary = {
            'date': today,
            'total_predictions': len(predictions_df),
            'locations_monitored': predictions_df['location_id'].nunique(),
            'average_confidence': predictions_df['confidence_score'].mean(),
            'prediction_types': predictions_df['prediction_type'].value_counts().to_dict()
        }
        
        # Alert conditions
        alerts = self._check_daily_alerts(predictions_df)
        
        # Save report
        report = {
            'summary': summary,
            'alerts': alerts,
            'detailed_data': predictions_df.to_dict('records')
        }
        
        report_filename = f'daily_report_{today}.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Daily report saved: {report_filename}")
        
        return report
    
    def _check_daily_alerts(self, predictions_df):
        """Check for alert conditions in daily predictions"""
        alerts = []
        
        for _, prediction in predictions_df.iterrows():
            output_data = json.loads(prediction['output_data'])
            
            # Air quality alerts
            if prediction['prediction_type'] == 'air_quality':
                aqi_value = output_data.get('aqi_value', 0)
                if aqi_value > 100:
                    alerts.append({
                        'type': 'air_quality_alert',
                        'location': prediction['location_name'],
                        'severity': 'high',
                        'message': f'Poor air quality: AQI {aqi_value}',
                        'timestamp': prediction['timestamp']
                    })
            
            # Traffic alerts
            elif prediction['prediction_type'] == 'traffic':
                congestion_level = output_data.get('congestion_level', 0)
                if congestion_level > 80:
                    alerts.append({
                        'type': 'traffic_alert',
                        'location': prediction['location_name'],
                        'severity': 'medium',
                        'message': f'Heavy traffic congestion: Level {congestion_level}',
                        'timestamp': prediction['timestamp']
                    })
            
            # Housing alerts (price increases)
            elif prediction['prediction_type'] == 'housing':
                # Would need historical data to detect price changes
                pass
        
        return alerts
    
    def setup_scheduled_pipeline(self):
        """Setup scheduled pipeline execution"""
        # Schedule predictions every hour
        schedule.every().hour.do(self.run_prediction_pipeline)
        
        # Schedule daily report at 6 AM
        schedule.every().day.at("06:00").do(self.generate_daily_report)
        
        self.logger.info("Scheduled pipeline setup completed.")
        self.logger.info("Predictions: Every hour")
        self.logger.info("Daily reports: 6:00 AM")
    
    def run_scheduler(self):
        """Run the scheduler (blocking)"""
        self.logger.info("Starting scheduler...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Usage example
sdk = SmartUrbanSDK()
pipeline = DataPipelineIntegration(sdk)

# Add monitoring locations
pipeline.add_monitoring_location(
    'downtown_01',
    'Downtown Business District',
    28.6139, 77.2090,
    {
        'air_quality': {'country': 'India', 'status': 'Moderate'},
        'traffic': True,
        'housing': {'population_density': 45000, 'average_income': 70000}
    }
)

pipeline.add_monitoring_location(
    'residential_01',
    'Residential Area North',
    28.7041, 77.1025,
    {
        'air_quality': {'country': 'India', 'status': 'Good'},
        'traffic': True,
        'housing': {'population_density': 25000, 'average_income': 55000}
    }
)

# Setup and run pipeline
pipeline.setup_scheduled_pipeline()

# For demonstration, run pipeline once manually
pipeline.run_prediction_pipeline()
daily_report = pipeline.generate_daily_report()

print("Pipeline integration completed. Check 'urban_pipeline.log' for details.")
```

---

## Advanced Use Cases

### 1. Machine Learning Model Ensemble

```python
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error

class AdvancedPredictionEnsemble:
    """Advanced ensemble methods for improved predictions"""
    
    def __init__(self, sdk):
        self.sdk = sdk
        self.ensemble_models = {}
        self.prediction_history = []
    
    def create_ensemble_prediction(self, prediction_type, inputs, historical_data=None):
        """Create ensemble prediction using multiple approaches"""
        
        if prediction_type == 'air_quality':
            return self._ensemble_air_quality(inputs, historical_data)
        elif prediction_type == 'traffic':
            return self._ensemble_traffic(inputs, historical_data)
        elif prediction_type == 'housing':
            return self._ensemble_housing(inputs, historical_data)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def _ensemble_air_quality(self, inputs, historical_data):
        """Ensemble air quality prediction"""
        predictions = []
        weights = []
        
        # Base model prediction
        base_result = self.sdk.get_air_quality_prediction(
            inputs['country'], inputs['status'],
            inputs['year'], inputs['month'], inputs['day']
        )
        
        if 'error' not in base_result:
            predictions.append(base_result['aqi_value'])
            weights.append(0.6)  # Base model weight
        
        # Seasonal adjustment
        seasonal_adjustment = self._calculate_seasonal_adjustment(
            inputs['month'], inputs['day'], 'air_quality'
        )
        
        if predictions:
            adjusted_prediction = predictions[0] * seasonal_adjustment
            predictions.append(adjusted_prediction)
            weights.append(0.3)
        
        # Historical trend adjustment
        if historical_data:
            trend_prediction = self._calculate_trend_prediction(historical_data, 'air_quality')
            if trend_prediction:
                predictions.append(trend_prediction)
                weights.append(0.1)
        
        # Calculate weighted ensemble prediction
        if predictions and weights:
            ensemble_value = np.average(predictions, weights=weights)
            
            # Determine category
            if ensemble_value <= 50:
                category = "Good"
            elif ensemble_value <= 100:
                category = "Moderate"
            else:
                category = "Bad"
            
            return {
                'aqi_value': round(ensemble_value, 2),
                'category': category,
                'ensemble_info': {
                    'base_prediction': predictions[0] if predictions else None,
                    'seasonal_adjustment': seasonal_adjustment,
                    'confidence': self._calculate_ensemble_confidence(predictions, weights)
                }
            }
        
        return base_result
    
    def _ensemble_traffic(self, inputs, historical_data):
        """Ensemble traffic prediction"""
        predictions = []
        weights = []
        
        # Base model prediction
        base_result = self.sdk.get_traffic_prediction(inputs['hour'], inputs['day_of_week'])
        
        if 'error' not in base_result:
            predictions.append(base_result['congestion_level'])
            weights.append(0.5)
        
        # Time-of-day pattern adjustment
        time_adjustment = self._calculate_time_pattern_adjustment(
            inputs['hour'], inputs['day_of_week']
        )
        
        if predictions:
            adjusted_prediction = predictions[0] * time_adjustment
            predictions.append(adjusted_prediction)
            weights.append(0.3)
        
        # Weather/event adjustment (simplified)
        event_adjustment = self._calculate_event_adjustment(inputs.get('special_events', []))
        if predictions:
            event_adjusted = predictions[0] * event_adjustment
            predictions.append(event_adjusted)
            weights.append(0.2)
        
        # Calculate ensemble prediction
        if predictions and weights:
            ensemble_value = np.average(predictions, weights=weights)
            category = "High" if ensemble_value >= 50 else "Low"
            
            return {
                'congestion_level': round(ensemble_value, 2),
                'category': category,
                'ensemble_info': {
                    'base_prediction': predictions[0] if predictions else None,
                    'time_adjustment': time_adjustment,
                    'event_adjustment': event_adjustment,
                    'confidence': self._calculate_ensemble_confidence(predictions, weights)
                }
            }
        
        return base_result
    
    def _ensemble_housing(self, inputs, historical_data):
        """Ensemble housing prediction"""
        predictions = []
        weights = []
        
        # Base model prediction
        base_result = self.sdk.get_housing_prediction(
            inputs['population_density'], inputs['average_income']
        )
        
        if 'error' not in base_result:
            predictions.append(base_result['predicted_price'])
            weights.append(0.5)
        
        # Market trend adjustment
        market_adjustment = self._calculate_market_trend_adjustment(
            inputs['population_density'], inputs['average_income']
        )
        
        if predictions:
            adjusted_prediction = predictions[0] * market_adjustment
            predictions.append(adjusted_prediction)
            weights.append(0.3)
        
        # Economic indicator adjustment
        economic_adjustment = self._calculate_economic_adjustment(inputs)
        if predictions:
            econ_adjusted = predictions[0] * economic_adjustment
            predictions.append(econ_adjusted)
            weights.append(0.2)
        
        # Calculate ensemble prediction
        if predictions and weights:
            ensemble_value = np.average(predictions, weights=weights)
            affordability = "Affordable" if ensemble_value <= 50000 else "Not Affordable"
            
            return {
                'predicted_price': round(ensemble_value, 2),
                'affordability': affordability,
                'ensemble_info': {
                    'base_prediction': predictions[0] if predictions else None,
                    'market_adjustment': market_adjustment,
                    'economic_adjustment': economic_adjustment,
                    'confidence': self._calculate_ensemble_confidence(predictions, weights)
                }
            }
        
        return base_result
    
    def _calculate_seasonal_adjustment(self, month, day, prediction_type):
        """Calculate seasonal adjustment factor"""
        # Simplified seasonal patterns
        seasonal_patterns = {
            'air_quality': {
                'winter_months': [12, 1, 2],
                'summer_months': [6, 7, 8],
                'winter_factor': 1.2,  # Worse air quality in winter
                'summer_factor': 0.9   # Better air quality in summer
            }
        }
        
        pattern = seasonal_patterns.get(prediction_type, {})
        
        if month in pattern.get('winter_months', []):
            return pattern.get('winter_factor', 1.0)
        elif month in pattern.get('summer_months', []):
            return pattern.get('summer_factor', 1.0)
        else:
            return 1.0
    
    def _calculate_time_pattern_adjustment(self, hour, day_of_week):
        """Calculate time-based pattern adjustment"""
        # Rush hour adjustments
        if day_of_week < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                return 1.3
            elif 22 <= hour or hour <= 5:  # Night hours
                return 0.5
        else:  # Weekends
            if 10 <= hour <= 14:  # Weekend peak
                return 1.1
            else:
                return 0.8
        
        return 1.0
    
    def _calculate_event_adjustment(self, special_events):
        """Calculate adjustment for special events"""
        if not special_events:
            return 1.0
        
        # Simple event impact multipliers
        event_impacts = {
            'concert': 1.5,
            'sports_game': 1.8,
            'festival': 1.4,
            'construction': 1.3,
            'holiday': 0.7
        }
        
        max_impact = max([event_impacts.get(event, 1.0) for event in special_events])
        return max_impact
    
    def _calculate_market_trend_adjustment(self, population_density, average_income):
        """Calculate market trend adjustment"""
        # Simplified market trend logic
        if population_density > 40000 and average_income > 70000:
            return 1.15  # High demand areas
        elif population_density < 20000 and average_income < 50000:
            return 0.9   # Lower demand areas
        else:
            return 1.0   # Neutral
    
    def _calculate_economic_adjustment(self, inputs):
        """Calculate economic indicator adjustment"""
        # Simplified economic adjustment based on income levels
        income = inputs.get('average_income', 60000)
        
        if income > 80000:
            return 1.1  # Higher income drives prices up
        elif income < 40000:
            return 0.95  # Lower income drives prices down
        else:
            return 1.0
    
    def _calculate_trend_prediction(self, historical_data, prediction_type):
        """Calculate trend-based prediction"""
        if not historical_data or len(historical_data) < 3:
            return None
        
        # Simple linear trend calculation
        values = [item['value'] for item in historical_data[-5:]]  # Last 5 values
        if len(values) >= 2:
            trend = (values[-1] - values[0]) / len(values)
            return values[-1] + trend
        
        return None
    
    def _calculate_ensemble_confidence(self, predictions, weights):
        """Calculate confidence for ensemble prediction"""
        if not predictions or len(predictions) < 2:
            return 0.7  # Base confidence
        
        # Calculate variance in predictions
        variance = np.var(predictions)
        
        # Lower variance = higher confidence
        if variance < 10:
            return 0.95
        elif variance < 50:
            return 0.85
        elif variance < 100:
            return 0.75
        else:
            return 0.65

# Usage example
sdk = SmartUrbanSDK()
ensemble = AdvancedPredictionEnsemble(sdk)

# Air quality ensemble prediction
air_inputs = {
    'country': 'India',
    'status': 'Moderate',
    'year': 2024,
    'month': 12,  # Winter month
    'day': 15
}

historical_air_data = [
    {'date': '2024-12-10', 'value': 65},
    {'date': '2024-12-11', 'value': 68},
    {'date': '2024-12-12', 'value': 70},
    {'date': '2024-12-13', 'value': 72},
    {'date': '2024-12-14', 'value': 75}
]

ensemble_air_result = ensemble.create_ensemble_prediction(
    'air_quality', air_inputs, historical_air_data
)

print("Ensemble Air Quality Prediction:")
print(json.dumps(ensemble_air_result, indent=2))

# Traffic ensemble prediction
traffic_inputs = {
    'hour': 17,
    'day_of_week': 4,
    'special_events': ['sports_game']
}

ensemble_traffic_result = ensemble.create_ensemble_prediction(
    'traffic', traffic_inputs
)

print("\nEnsemble Traffic Prediction:")
print(json.dumps(ensemble_traffic_result, indent=2))

# Housing ensemble prediction
housing_inputs = {
    'population_density': 35000,
    'average_income': 85000
}

ensemble_housing_result = ensemble.create_ensemble_prediction(
    'housing', housing_inputs
)

print("\nEnsemble Housing Prediction:")
print(json.dumps(ensemble_housing_result, indent=2))
```

### 2. Real-time Monitoring and Alerting System

```python
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import threading
import queue

class RealTimeMonitoringSystem:
    """Real-time monitoring and alerting system"""
    
    def __init__(self, sdk):
        self.sdk = sdk
        self.monitoring_locations = {}
        self.alert_thresholds = {}
        self.alert_queue = queue.Queue()
        self.websocket_clients = set()
        self.is_monitoring = False
        
    def add_monitoring_location(self, location_id, config):
        """Add location to real-time monitoring"""
        self.monitoring_locations[location_id] = config
        print(f"Added monitoring for location: {location_id}")
    
    def set_alert_thresholds(self, location_id, thresholds):
        """Set alert thresholds for a location"""
        self.alert_thresholds[location_id] = thresholds
        print(f"Set alert thresholds for {location_id}: {thresholds}")
    
    def start_monitoring(self, interval_seconds=300):  # 5 minutes default
        """Start real-time monitoring"""
        self.is_monitoring = True
        
        def monitoring_loop():
            while self.is_monitoring:
                try:
                    self._run_monitoring_cycle()
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(30)  # Wait 30 seconds before retrying
        
        monitoring_thread = threading.Thread(target=monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        print(f"Real-time monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        print("Real-time monitoring stopped")
    
    def _run_monitoring_cycle(self):
        """Run single monitoring cycle for all locations"""
        cycle_timestamp = datetime.now()
        
        for location_id, config in self.monitoring_locations.items():
            try:
                # Get current predictions
                predictions = self._get_location_predictions(location_id, config)
                
                # Check for alerts
                alerts = self._check_location_alerts(location_id, predictions)
                
                # Queue alerts
                for alert in alerts:
                    self.alert_queue.put(alert)
                
                # Broadcast updates to websocket clients
                update_data = {
                    'type': 'location_update',
                    'timestamp': cycle_timestamp.isoformat(),
                    'location_id': location_id,
                    'predictions': predictions,
                    'alerts': alerts
                }
                
                self._broadcast_to_clients(update_data)
                
            except Exception as e:
                print(f"Error monitoring location {location_id}: {e}")
    
    def _get_location_predictions(self, location_id, config):
        """Get predictions for a location"""
        current_time = datetime.now()
        predictions = {}
        
        # Air quality prediction
        if 'air_quality' in config:
            aq_config = config['air_quality']
            predictions['air_quality'] = self.sdk.get_air_quality_prediction(
                aq_config['country'], aq_config['status'],
                current_time.year, current_time.month, current_time.day
            )
        
        # Traffic prediction
        if 'traffic' in config and config['traffic']:
            predictions['traffic'] = self.sdk.get_traffic_prediction(
                current_time.hour, current_time.weekday()
            )
        
        # Housing prediction (less frequent updates)
        if 'housing' in config and current_time.hour in [6, 12, 18]:  # 3 times a day
            housing_config = config['housing']
            predictions['housing'] = self.sdk.get_housing_prediction(
                housing_config['population_density'],
                housing_config['average_income']
            )
        
        return predictions
    
    def _check_location_alerts(self, location_id, predictions):
        """Check for alert conditions"""
        alerts = []
        thresholds = self.alert_thresholds.get(location_id, {})
        
        # Air quality alerts
        if 'air_quality' in predictions and 'air_quality' in thresholds:
            aq_result = predictions['air_quality']
            aq_thresholds = thresholds['air_quality']
            
            if 'aqi_value' in aq_result:
                aqi_value = aq_result['aqi_value']
                
                if aqi_value >= aq_thresholds.get('critical', 150):
                    alerts.append({
                        'type': 'air_quality',
                        'severity': 'critical',
                        'location_id': location_id,
                        'value': aqi_value,
                        'threshold': aq_thresholds['critical'],
                        'message': f'Critical air quality: AQI {aqi_value}',
                        'timestamp': datetime.now().isoformat()
                    })
                elif aqi_value >= aq_thresholds.get('warning', 100):
                    alerts.append({
                        'type': 'air_quality',
                        'severity': 'warning',
                        'location_id': location_id,
                        'value': aqi_value,
                        'threshold': aq_thresholds['warning'],
                        'message': f'Poor air quality: AQI {aqi_value}',
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Traffic alerts
        if 'traffic' in predictions and 'traffic' in thresholds:
            traffic_result = predictions['traffic']
            traffic_thresholds = thresholds['traffic']
            
            if 'congestion_level' in traffic_result:
                congestion = traffic_result['congestion_level']
                
                if congestion >= traffic_thresholds.get('critical', 90):
                    alerts.append({
                        'type': 'traffic',
                        'severity': 'critical',
                        'location_id': location_id,
                        'value': congestion,
                        'threshold': traffic_thresholds['critical'],
                        'message': f'Severe traffic congestion: Level {congestion}',
                        'timestamp': datetime.now().isoformat()
                    })
                elif congestion >= traffic_thresholds.get('warning', 70):
                    alerts.append({
                        'type': 'traffic',
                        'severity': 'warning',
                        'location_id': location_id,
                        'value': congestion,
                        'threshold': traffic_thresholds['warning'],
                        'message': f'Heavy traffic congestion: Level {congestion}',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def _broadcast_to_clients(self, data):
        """Broadcast data to all connected websocket clients"""
        if self.websocket_clients:
            message = json.dumps(data)
            disconnected_clients = set()
            
            for client in self.websocket_clients:
                try:
                    asyncio.create_task(client.send(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    print(f"Error broadcasting to client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    async def websocket_handler(self, websocket, path):
        """Handle websocket connections"""
        self.websocket_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            # Send current status to new client
            status_data = {
                'type': 'status',
                'monitoring_locations': list(self.monitoring_locations.keys()),
                'is_monitoring': self.is_monitoring,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(status_data))
            
            # Keep connection alive
            async for message in websocket:
                # Handle client messages (e.g., commands)
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.websocket_clients)}")
    
    async def _handle_client_message(self, websocket, data):
        """Handle messages from websocket clients"""
        message_type = data.get('type')
        
        if message_type == 'get_current_status':
            # Send current predictions for all locations
            current_status = {}
            for location_id, config in self.monitoring_locations.items():
                predictions = self._get_location_predictions(location_id, config)
                current_status[location_id] = predictions
            
            response = {
                'type': 'current_status',
                'data': current_status,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
        
        elif message_type == 'get_alerts':
            # Send recent alerts
            recent_alerts = []
            temp_queue = queue.Queue()
            
            # Get all alerts from queue
            while not self.alert_queue.empty():
                alert = self.alert_queue.get()
                recent_alerts.append(alert)
                temp_queue.put(alert)  # Keep for other consumers
            
            # Put alerts back in queue
            while not temp_queue.empty():
                self.alert_queue.put(temp_queue.get())
            
            response = {
                'type': 'alerts',
                'data': recent_alerts,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
    
    def start_websocket_server(self, host='localhost', port=8765):
        """Start websocket server for real-time updates"""
        async def server():
            async with websockets.serve(self.websocket_handler, host, port):
                print(f"WebSocket server started on ws://{host}:{port}")
                await asyncio.Future()  # Run forever
        
        # Run websocket server in separate thread
        def run_server():
            asyncio.run(server())
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
    
    def get_alert_summary(self, hours=24):
        """Get alert summary for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        # This is a simplified version - in practice, you'd store alerts in a database
        temp_queue = queue.Queue()
        
        while not self.alert_queue.empty():
            alert = self.alert_queue.get()
            alert_time = datetime.fromisoformat(alert['timestamp'])
            
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)
            
            temp_queue.put(alert)
        
        # Put alerts back
        while not temp_queue.empty():
            self.alert_queue.put(temp_queue.get())
        
        # Generate summary
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': {},
            'by_type': {},
            'by_location': {}
        }
        
        for alert in recent_alerts:
            # By severity
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # By type
            alert_type = alert['type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
            
            # By location
            location = alert['location_id']
            summary['by_location'][location] = summary['by_location'].get(location, 0) + 1
        
        return summary

# Usage example
sdk = SmartUrbanSDK()
monitoring_system = RealTimeMonitoringSystem(sdk)

# Add monitoring locations
monitoring_system.add_monitoring_location('downtown', {
    'air_quality': {'country': 'India', 'status': 'Moderate'},
    'traffic': True,
    'housing': {'population_density': 45000, 'average_income': 70000}
})

monitoring_system.add_monitoring_location('residential', {
    'air_quality': {'country': 'India', 'status': 'Good'},
    'traffic': True,
    'housing': {'population_density': 25000, 'average_income': 55000}
})

# Set alert thresholds
monitoring_system.set_alert_thresholds('downtown', {
    'air_quality': {'warning': 75, 'critical': 120},
    'traffic': {'warning': 65, 'critical': 85}
})

monitoring_system.set_alert_thresholds('residential', {
    'air_quality': {'warning': 60, 'critical': 100},
    'traffic': {'warning': 50, 'critical': 75}
})

# Start monitoring and websocket server
monitoring_system.start_monitoring(interval_seconds=60)  # Check every minute
monitoring_system.start_websocket_server(host='localhost', port=8765)

print("Real-time monitoring system started!")
print("WebSocket server: ws://localhost:8765")
print("Monitoring locations:", list(monitoring_system.monitoring_locations.keys()))

# Get alert summary
import time
time.sleep(5)  # Wait a bit for some monitoring cycles
alert_summary = monitoring_system.get_alert_summary(hours=1)
print("\nAlert Summary (last hour):")
print(json.dumps(alert_summary, indent=2))
```

---

This comprehensive usage guide provides practical examples for implementing the Smart Urban Settlement Management System in various scenarios, from basic API usage to advanced real-time monitoring systems. Each example includes complete, runnable code that demonstrates best practices for integrating the system into larger urban planning and management workflows.
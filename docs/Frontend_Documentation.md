# Frontend Documentation

## Overview

The Smart Urban Settlement Management System features a modern, responsive web interface built with HTML5, CSS3, and vanilla JavaScript. The frontend provides an intuitive dashboard for interacting with all three prediction modules.

## Architecture

```
Frontend Structure
├── Dashboard (Main Interface)
├── Air Quality Prediction Form
├── Traffic Congestion Prediction Form
├── Housing Price Prediction Form
└── Results Display & Visualization
```

---

## Main Components

### 1. Dashboard (`templates/dashboard.html`)

The primary interface that consolidates all prediction functionalities into a single, user-friendly page.

#### Structure
- **Header**: Navigation and branding
- **Main Section**: Three prediction modules
- **Footer**: Copyright and branding

#### Features
- Responsive design for desktop and mobile
- Modern card-based layout
- Real-time form validation
- Dynamic result display
- Integrated visualization support

---

## Form Components

### 1. Air Quality Prediction Form

**Purpose**: Collects geographical and temporal data for AQI prediction.

#### Form Fields

| Field | Type | Validation | Description |
|-------|------|------------|-------------|
| Country | text | required | Country name (e.g., "India", "Albania") |
| Status | text | required | Environmental status (e.g., "Good", "Moderate") |
| Year | number | required | Year value (2020-2025 recommended) |
| Month | number | required | Month (1-12) |
| Day | number | required | Day (1-31) |

#### HTML Structure
```html
<form action="/predict/air_quality" method="POST">
    <label for="Country">Country:</label>
    <input type="text" id="Country" name="Country" placeholder="Enter country" required>
    
    <label for="Status">Status:</label>
    <input type="text" id="Status" name="Status" placeholder="Enter status" required>
    
    <label for="year">Year:</label>
    <input type="number" id="year" name="year" placeholder="Enter year" required>
    
    <label for="month">Month:</label>
    <input type="number" id="month" name="month" placeholder="Enter month" required>
    
    <label for="day">Day:</label>
    <input type="number" id="day" name="day" placeholder="Enter day" required>
    
    <button type="submit">Get Prediction</button>
</form>
```

#### Usage Example
```javascript
// Example form submission data
const airQualityData = {
    Country: "India",
    Status: "Good",
    year: 2024,
    month: 7,
    day: 15
};
```

#### Result Display
```html
{% if air_quality_prediction %}
<p><strong>Prediction:</strong> {{ air_quality_prediction }}</p>
<img src="{{ url_for('static', filename='images/actual_vs_predicted.png') }}" alt="Air Quality Plot">
{% endif %}
```

---

### 2. Traffic Congestion Prediction Form

**Purpose**: Collects temporal data for traffic congestion analysis.

#### Form Fields

| Field | Type | Validation | Description |
|-------|------|------------|-------------|
| Hour | number | required, min=0, max=23 | Hour of day (24-hour format) |
| Day of Week | number | required, min=0, max=6 | Day (0=Monday, 6=Sunday) |

#### HTML Structure
```html
<form action="/predict/traffic" method="POST">
    <label for="hour">Hour of the Day (0-23):</label>
    <input type="number" id="hour" name="hour" min="0" max="23" placeholder="Enter hour" required>
    
    <label for="day_of_week">Day of the Week (0=Monday, ..., 6=Sunday):</label>
    <input type="number" id="day_of_week" name="day_of_week" min="0" max="6" placeholder="Enter day" required>
    
    <button type="submit">Get Prediction</button>
</form>
```

#### Usage Example
```javascript
// Example form submission data
const trafficData = {
    hour: 17,        // 5 PM
    day_of_week: 4   // Friday
};
```

#### Helper Information
- **Peak Hours**: Typically 7-9 AM and 5-7 PM (hours 7-9, 17-19)
- **Weekdays**: Monday-Friday (0-4) usually have higher congestion
- **Weekends**: Saturday-Sunday (5-6) typically have lower congestion

#### Result Display
```html
{% if traffic_prediction %}
<p><strong>Prediction:</strong> {{ traffic_prediction }}</p>
<img src="{{ url_for('static', filename='images/actual_vs_predicted_congestion.png') }}" alt="Traffic Plot">
{% endif %}
```

---

### 3. Housing Price Prediction Form

**Purpose**: Collects demographic and economic data for housing price analysis.

#### Form Fields

| Field | Type | Validation | Description |
|-------|------|------------|-------------|
| Population Density | number | required | Population per area unit |
| Average Income | number | required | Average income in the area |

#### HTML Structure
```html
<form action="/predict/housing" method="POST">
    <label for="population_density">Population Density:</label>
    <input type="number" id="population_density" name="population_density" placeholder="Enter population density" required>
    
    <label for="average_income">Average Income:</label>
    <input type="number" id="average_income" name="average_income" placeholder="Enter average income" required>
    
    <button type="submit">Get Prediction</button>
</form>
```

#### Usage Example
```javascript
// Example form submission data
const housingData = {
    population_density: 25000,
    average_income: 75000
};
```

#### Input Guidelines
- **Population Density**: Typical range 10,000-100,000 people per area unit
- **Average Income**: Annual income in local currency (e.g., $30,000-$150,000)

#### Result Display
```html
{% if housing_prediction %}
<p><strong>Prediction:</strong> {{ housing_prediction }}</p>
<img src="{{ url_for('static', filename='images/housing_demand_plot.png') }}" alt="Housing Plot">
{% endif %}
```

---

## Styling & Design

### CSS Architecture

The dashboard uses embedded CSS for simplicity and self-contained deployment.

#### Color Scheme
```css
:root {
    --primary-color: rgb(170, 30, 77);     /* Header background */
    --secondary-color: rgb(36, 77, 0);     /* Button background */
    --accent-color: rgb(121, 0, 34);       /* Headings */
    --background-color: #f4f4f9;           /* Page background */
    --card-background: white;              /* Form backgrounds */
    --text-color: #333;                    /* Primary text */
    --border-color: #ddd;                  /* Form borders */
}
```

#### Typography
```css
body {
    font-family: Arial, sans-serif;
    color: var(--text-color);
}

h1 {
    font-size: 2rem;
    margin: 0;
}

h2 {
    color: var(--accent-color);
    text-align: center;
    margin-bottom: 30px;
}

h3 {
    color: var(--accent-color);
    margin-top: 0;
}
```

#### Layout System
```css
/* Container */
section {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

/* Card Components */
div {
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(121, 0, 0, 0.1);
}
```

#### Form Styling
```css
/* Form Layout */
form {
    display: flex;
    flex-direction: column;
}

/* Input Fields */
form input {
    margin-bottom: 15px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

/* Submit Buttons */
form button {
    background-color: rgb(36, 77, 0);
    color: white;
    padding: 10px;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

form button:hover {
    background-color: rgb(170, 30, 77);
}
```

### Responsive Design

The dashboard is optimized for various screen sizes:

#### Desktop (≥1200px)
- Full-width layout with maximum container width
- Three-column card layout possible
- Full navigation menu

#### Tablet (768px-1199px)
- Responsive padding and margins
- Two-column card layout
- Condensed navigation

#### Mobile (≤767px)
- Single-column layout
- Touch-friendly button sizes
- Optimized form inputs

---

## User Experience Features

### 1. Form Validation

#### Client-side Validation
```html
<!-- Built-in HTML5 validation -->
<input type="number" min="0" max="23" required>
<input type="text" required>
```

#### Custom Validation (JavaScript)
```javascript
// Example custom validation
function validateAirQualityForm() {
    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;
    const day = document.getElementById('day').value;
    
    if (year < 2020 || year > 2025) {
        alert('Please enter a year between 2020 and 2025');
        return false;
    }
    
    if (month < 1 || month > 12) {
        alert('Please enter a valid month (1-12)');
        return false;
    }
    
    if (day < 1 || day > 31) {
        alert('Please enter a valid day (1-31)');
        return false;
    }
    
    return true;
}
```

### 2. Loading States

#### Submission Feedback
```javascript
// Show loading state during form submission
function showLoading(formId) {
    const button = document.querySelector(`#${formId} button[type="submit"]`);
    button.textContent = 'Predicting...';
    button.disabled = true;
}

function hideLoading(formId) {
    const button = document.querySelector(`#${formId} button[type="submit"]`);
    button.textContent = 'Get Prediction';
    button.disabled = false;
}
```

### 3. Result Display

#### Dynamic Content Updates
```javascript
// Update prediction results
function updatePredictionResult(containerId, prediction, imageUrl) {
    const container = document.getElementById(containerId);
    
    // Update text
    const predictionText = container.querySelector('.prediction-text');
    predictionText.textContent = prediction;
    
    // Update image
    const predictionImage = container.querySelector('.prediction-image');
    predictionImage.src = imageUrl;
    predictionImage.style.display = 'block';
}
```

---

## Visualization Integration

### 1. Chart Display

The dashboard displays generated charts from the ML modules:

#### Image Integration
```html
<!-- Air Quality Chart -->
<img src="{{ url_for('static', filename='images/actual_vs_predicted.png') }}" 
     alt="Air Quality Prediction Plot"
     style="max-width: 100%; border-radius: 8px;">

<!-- Traffic Chart -->
<img src="{{ url_for('static', filename='images/actual_vs_predicted_congestion.png') }}" 
     alt="Traffic Congestion Plot"
     style="max-width: 100%; border-radius: 8px;">

<!-- Housing Chart -->
<img src="{{ url_for('static', filename='images/housing_demand_plot.png') }}" 
     alt="Housing Price Plot"
     style="max-width: 100%; border-radius: 8px;">
```

#### Chart Styling
```css
img {
    max-width: 100%;
    margin-top: 15px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
```

### 2. Interactive Elements

#### Chart Zoom (Future Enhancement)
```javascript
// Future: Add chart interaction
function enableChartZoom(chartId) {
    const chart = document.getElementById(chartId);
    chart.addEventListener('click', function() {
        // Open chart in modal or new window
        window.open(chart.src, '_blank');
    });
}
```

---

## Accessibility Features

### 1. Semantic HTML
```html
<header role="banner">
    <h1>Smart Urban Settlement Management</h1>
    <nav role="navigation">
        <ul>
            <li><a href="/">Home</a></li>
        </ul>
    </nav>
</header>

<main role="main">
    <section aria-labelledby="predictions-heading">
        <h2 id="predictions-heading">Real-Time Predictions</h2>
        <!-- Form content -->
    </section>
</main>

<footer role="contentinfo">
    <p>&copy; 2025 Smart Urban Management</p>
</footer>
```

### 2. Form Accessibility
```html
<!-- Properly labeled inputs -->
<label for="Country">Country:</label>
<input type="text" id="Country" name="Country" 
       aria-describedby="country-help" required>
<div id="country-help">Enter the country name for prediction</div>

<!-- ARIA attributes -->
<form aria-labelledby="air-quality-heading">
    <h3 id="air-quality-heading">Air Quality Prediction</h3>
    <!-- Form fields -->
</form>
```

### 3. Keyboard Navigation
```css
/* Focus styles */
input:focus,
button:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

/* Skip navigation */
.skip-nav {
    position: absolute;
    top: -40px;
    left: 6px;
    background: var(--primary-color);
    color: white;
    padding: 8px;
    text-decoration: none;
    transition: top 0.3s;
}

.skip-nav:focus {
    top: 6px;
}
```

---

## Performance Optimization

### 1. CSS Optimization
```css
/* Efficient selectors */
.prediction-form { /* Class-based */ }
#air-quality-form { /* ID-based */ }

/* Avoid expensive selectors */
/* Bad: div > div > div */
/* Good: .card-content */
```

### 2. Image Optimization
```html
<!-- Responsive images -->
<img src="chart.png" 
     alt="Prediction Chart"
     loading="lazy"
     width="800" 
     height="600">
```

### 3. JavaScript Optimization
```javascript
// Debounce form validation
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const debouncedValidation = debounce(validateForm, 300);
```

---

## Browser Support

### Supported Browsers
- **Chrome**: 70+
- **Firefox**: 65+
- **Safari**: 12+
- **Edge**: 18+

### Fallbacks
```css
/* CSS Grid with flexbox fallback */
.card-container {
    display: flex;
    flex-wrap: wrap;
}

@supports (display: grid) {
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
    }
}
```

---

## Future Enhancements

### 1. Interactive Dashboards
- Real-time data updates
- Interactive charts with D3.js or Chart.js
- Dashboard customization options

### 2. Progressive Web App (PWA)
- Offline functionality
- Push notifications
- App-like experience

### 3. Advanced UI Components
- Date/time pickers
- Autocomplete for country/city selection
- Range sliders for numeric inputs

### 4. Data Visualization
- Multiple chart types (line, bar, heatmap)
- Comparative analysis views
- Historical trend displays

---

## Deployment Considerations

### 1. Static File Optimization
```bash
# Minify CSS and JavaScript
npm install -g clean-css-cli uglify-js

# Minify CSS
cleancss -o dashboard.min.css dashboard.css

# Minify JavaScript
uglifyjs script.js -o script.min.js
```

### 2. CDN Integration
```html
<!-- Future: Use CDN for common libraries -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

### 3. Caching Strategy
```html
<!-- Cache busting -->
<link rel="stylesheet" href="style.css?v=1.0.0">
<script src="script.js?v=1.0.0"></script>
```
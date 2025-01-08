from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


air_quality_model = joblib.load('models/air_quality_model.pkl')
traffic_model = joblib.load('models/traffic_congestion_model.pkl')
housing_model = joblib.load('models/housing_model.pkl')


country_encoder = joblib.load('models/label_encoder_country.pkl')  
status_encoder = joblib.load('models/label_encoder_status.pkl')    

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict/air_quality', methods=['POST'])
def predict_air_quality():
    # Get form data
    Country = str(request.form['Country'])
    Status = str(request.form['Status'])
    year = float(request.form['year'])
    month = float(request.form['month'])
    day = float(request.form['day'])

    
    try:
        Country_encoded = country_encoder.transform([Country])[0]  
        Status_encoded = status_encoder.transform([Status])[0]    
    except ValueError as e:
        return render_template('dashboard.html', air_quality_prediction="Invalid category input!")

   
    features = np.array([[Country_encoded, Status_encoded, year, month, day]])

   
    prediction = air_quality_model.predict(features)[0]

   
    if prediction <= 50:
        category = "Good"
    elif 51 <= prediction <= 100:
        category = "Moderate"
    else:
        category = "Bad"

  
    return render_template('dashboard.html', 
                           air_quality_prediction=f"Predicted AQI: {prediction:.2f} ({category})")

@app.route('/predict/traffic', methods=['POST'])
def predict_traffic():
    hour = int(request.form['hour'])
    day_of_week = int(request.form['day_of_week'])

    
    features = np.array([[hour, day_of_week]])

    prediction = traffic_model.predict(features)[0]

    congestion_threshold = 50  
    congestion_level = "High" if prediction >= congestion_threshold else "Low"

    
    return render_template('dashboard.html', 
                           traffic_prediction=f"Traffic Congestion Level: {prediction:.2f} ({congestion_level})")

@app.route('/predict/housing', methods=['POST'])
def predict_housing():
    population_density = int(request.form['population_density'])
    average_income = int(request.form['average_income'])

   
    features = np.array([[population_density, average_income]])

    
    prediction = housing_model.predict(features)[0]

    
    affordability_threshold = 50000  
    affordability = "Affordable" if prediction <= affordability_threshold else "Not Affordable"

    return render_template('dashboard.html', 
                           housing_prediction=f"Predicted Housing Price: {prediction:.2f} ({affordability})")

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os


def preprocess_and_train():
   
    data = pd.read_csv('C:/Users/91860/OneDrive/Desktop/Smart Urban 2/data/air_aquality.csv')

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day

    label_encoder_country = LabelEncoder()
    data['Country'] = label_encoder_country.fit_transform(data['Country'])

    label_encoder_status = LabelEncoder()
    data['Status'] = label_encoder_status.fit_transform(data['Status'])

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(label_encoder_country, 'models/label_encoder_country.pkl')
    joblib.dump(label_encoder_status, 'models/label_encoder_status.pkl')

    X = data[['Country', 'Status', 'year', 'month', 'day']]
    y = data['AQI Value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/air_quality_model.pkl')

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Training Complete. MSE on Test Data: {mse}")

    return model


def predict_aqi(country, status, year, month, day):
   
    model = joblib.load('models/air_quality_model.pkl')
    label_encoder_country = joblib.load('models/label_encoder_country.pkl')
    label_encoder_status = joblib.load('models/label_encoder_status.pkl')


    encoded_country = label_encoder_country.transform([country])[0]
    encoded_status = label_encoder_status.transform([status])[0]

    input_features = pd.DataFrame({
        'Country': [encoded_country],
        'Status': [encoded_status],
        'year': [year],
        'month': [month],
        'day': [day]
    })

 
    prediction = model.predict(input_features)[0]

    if prediction <= 50:
        category = "Good"
    elif 51 <= prediction <= 100:
        category = "Moderate"
    else:
        category = "Bad"

    return prediction, category



train_model = preprocess_and_train()

predicted_aqi, aqi_category = predict_aqi('Albania', 'Good', 2022, 7, 21)
print(f"Predicted AQI: {predicted_aqi}, Air Quality Category: {aqi_category}")

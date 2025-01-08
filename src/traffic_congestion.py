import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess_traffic_data():
    data = pd.read_csv(
        'C:/Users/91860/OneDrive/Desktop/Smart Urban 2/data/traffic_congestion.csv',
        header=None,
        names=["timestamp", "location", "congestion_level", "id"]
    )

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp'])

    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek

    data = data.drop(columns=["timestamp", "id", "location"])

    return data

def train_traffic_model():
    print("Training traffic model...")

  
    if not os.path.exists('models'):
        os.makedirs('models')

    data = preprocess_traffic_data()

    X = data[['hour', 'day_of_week']]
    y = data['congestion_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")



    
    plot_predictions(y_test, y_pred)
     

   
    joblib.dump(model, 'models/traffic_congestion_model.pkl')
    print("Model saved successfully!")

def predict_traffic_congestion(hour, day_of_week):
  
    try:
        model = joblib.load('models/traffic_congestion_model.pkl')
    except FileNotFoundError:
        print("Model not found! Please train the model first.")
        return None

    new_data = pd.DataFrame({'hour': [hour], 'day_of_week': [day_of_week]})

    prediction = model.predict(new_data)
    return prediction[0]



def plot_predictions(y_test, y_pred):
    
    plt.figure(figsize=(24, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Congestion Levels')
    plt.ylabel('Predicted Congestion Levels')
    plt.title('Actual vs Predicted Traffic Congestion Levels')
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig('static/images/actual_vs_predicted_congestion.png')
    plt.show()

if __name__ == '__main__':
   
    train_traffic_model()

   
    hour = 17 
    day_of_week = 4  
    prediction = predict_traffic_congestion(hour, day_of_week)

    if prediction is not None:
        print(f"Predicted Traffic Congestion Level for hour {hour} on day {day_of_week}: {prediction}")


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np

def preprocess_housing_data():
    data = pd.read_csv('C:/Users/91860/OneDrive/Desktop/Smart Urban 2/data/house_prediction.csv')

    data.columns = ['avg_area_income', 'avg_area_house_age', 'avg_area_rooms', 'avg_area_bedrooms', 
                    'area_population', 'price', 'address']
    
    data = data.drop(columns=['address'])

    data['area_population'] = pd.to_numeric(data['area_population'], errors='coerce')
    data['avg_area_income'] = pd.to_numeric(data['avg_area_income'], errors='coerce')

    data = data.dropna()

    data['population_density'] = data['area_population'] / data['avg_area_house_age']

    return data

def apply_kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['population_density', 'avg_area_income']])
    return data, kmeans

def train_housing_model():
    data = preprocess_housing_data()

   
    data, kmeans = apply_kmeans_clustering(data)


    if not os.path.exists('models'):
        os.makedirs('models')

    kmeans_model_path = 'models/kmeans_model.pkl'
    try:
        joblib.dump(kmeans, kmeans_model_path)
        print(f"KMeans model saved successfully to {kmeans_model_path}")
    except Exception as e:
        print(f"Error saving KMeans model: {e}")

    models = {}

   
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        X = cluster_data[['population_density', 'avg_area_income']]
        y = cluster_data['price']

        model = LinearRegression()
        model.fit(X, y)

        models[cluster] = model

    # Save all models
    models_path = 'models/housing_models.pkl'
    try:
        joblib.dump(models, models_path)
        print(f"Housing models saved successfully to {models_path}")
    except Exception as e:
        print(f"Error saving housing models: {e}")

    return models, kmeans

def predict_housing_demand(new_data, affordability_threshold=50000):
    try:
        kmeans = joblib.load('models/kmeans_model.pkl')
        models = joblib.load('models/housing_models.pkl')

        # Predict the cluster for the new data
        cluster = kmeans.predict(new_data[['population_density', 'avg_area_income']])
        cluster = cluster[0] 
       
        model = models.get(cluster)

        if model is None:
            print(f"No model found for cluster {cluster}")
            return None, None

        predictions = model.predict(new_data[['population_density', 'avg_area_income']])

   
        affordability = ["Affordable" if price <= affordability_threshold else "Not Affordable" 
                         for price in predictions]

        return predictions, affordability
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def plot_predictions(actual, predicted):
    plt.figure(figsize=(8, 6))
    plt.plot(actual, label='Actual Prices', marker='o', linestyle='-', color='blue')
    plt.plot(predicted, label='Predicted Prices', marker='x', linestyle='--', color='red')
    plt.title('Housing Price Prediction vs Actual')
    plt.xlabel('Index')
    plt.ylabel('Housing Price')
    plt.legend()
    plt.grid(True)

    plot_path = 'static/images/housing_demand_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved successfully to {plot_path}")

if __name__ == '__main__':
   
    trained_models, kmeans_model = train_housing_model()

   
    new_data = pd.DataFrame({
        'population_density': [24212, 53504],  
        'avg_area_income': [74213, 69423]     
    })

   
    predictions, affordability = predict_housing_demand(new_data)

    if predictions is not None:
        print("Predictions for new data:", predictions)
        print("Affordability:", affordability)

        actual = [23086, 40173]  

        plot_predictions(actual, predictions)

        mae = mean_absolute_error(actual, predictions)
        print(f"Mean Absolute Error: {mae}")

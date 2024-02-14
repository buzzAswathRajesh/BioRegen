# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Define the input data directly
data = pd.DataFrame({
    'Biodegradability': ['Control', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'Ratio 4', 'Ratio 5', 'Ratio 6'],
    'Degradation after 10 days': [8.541667, 10.203333, 10.536667, 12.736667, 13.568333, 14.808333, 15.175],
    'Degradation after 20 days': [20.988333, 24.578333, 25.205, 24.393333, 24.838333, 27.403333, 28.38],
    'Degradation after 30 days': [47.648333, 46.723333, 50.193333, 49.803333, 50.856667, 50.086667, 54.591667],
    'Degradation after 40 days': [57.856667, 59.596667, 58.845, 61.386667, 60.811667, 67.098333, 70.221667],
    'Degradation after 50 days': [69.761667, 73.055, 78.221667, 84.436667, 87.035, 90.036667, 92.98]
})

# Function to estimate days to reach 100% degradation using Holt's linear trend model
def estimate_days_to_100_holt(data):
    estimates_holt = {}
    for index, row in data.iterrows():
        ratio = row['Biodegradability']
        series = np.array(row[1:].astype(float))
        days = 50  # Starting from 50 days, as the last observed interval
        
        # Applying Holt's linear trend model
        model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
        model_fit = model.fit()
        
        # Forecast until reaching 100%
        total_degradation = series[-1]
        while total_degradation < 100:
            forecast = model_fit.forecast(1)[0]
            total_degradation = forecast  # Assuming the forecast is the latest degradation percentage
            days += 10  # Assuming each interval represents an additional 10 days
            # Update the series with the new forecast for re-fitting in the next iteration if necessary
            series = np.append(series, forecast)
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()
        
        estimates_holt[ratio] = days
    
    return estimates_holt

if __name__ == "__main__":
    estimates_holt = estimate_days_to_100_holt(data)
    print("\n-------------------------------------------\n")
    print("Holt's Linear Trend Model Prediction")
    print("\n-------------------------------------------\n")
    for ratio, days in estimates_holt.items():
        print(f"{ratio}: {days} days")
    print("\n-------------------------------------------\n")

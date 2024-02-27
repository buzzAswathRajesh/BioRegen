# Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean

def integrated_arima_forecasting():
    # Define the dataset as a DataFrame
    data = pd.DataFrame({
        'Ratio': ['Control', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'Ratio 4', 'Ratio 5', 'Ratio 6'],
        '0-10 days': [8.54, 10.20, 10.54, 12.74, 13.57, 14.81, 15.18],
        '10-20 days': [12.446667, 14.375, 14.668333, 11.656667, 11.27, 12.595, 13.205],
        '20-30 days': [26.66, 22.145, 24.988333, 25.41, 26.018333, 22.683333, 26.211667],
        '30-40 days': [10.208333, 12.873333, 8.651667, 11.583333, 9.955, 17.011667, 15.63],
        '40-50 days': [11.905, 13.458333, 19.376667, 23.05, 26.223333, 22.938333, 22.758333]
    }).set_index('Ratio')  # Set 'Ratio' column as the DataFrame index
    
    estimates = {}  # Initialize a dictionary to store estimated days to reach 100% degradation for each ratio
    
    # Iterate through each row in the DataFrame to model and forecast for each ratio
    for ratio, series in data.iterrows():
        series = series.astype(float)  # Ensure the series is in float format for ARIMA
        
        # Define and fit the ARIMA model with an order of (1, 1, 1)
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast the next 4 intervals (assuming each interval is 10 days)
        forecast = model_fit.forecast(steps=4)
        
        # Calculate the total cumulative degradation including the forecasted values
        total_cumulative_degradation = series.sum() + forecast.sum()
        
        # Calculate the average degradation in the forecasted period
        average_degradation_forecast = forecast.mean()
        
        # Calculate the remaining degradation needed to reach 100%
        remaining_degradation_needed = 100 - total_cumulative_degradation
        
        # Estimate the number of additional intervals needed to reach 100%, based on average forecast degradation
        additional_intervals_needed = remaining_degradation_needed / average_degradation_forecast if average_degradation_forecast > 0 else 0
        
        # Calculate the total intervals (existing + forecasted + additional) to reach 100% degradation
        total_intervals_to_100 = len(series) + len(forecast) + additional_intervals_needed
        
        # Convert intervals to days (assuming each interval represents 10 days)
        total_days_to_100 = total_intervals_to_100 * 10
        
        # Store the total days to reach 100% degradation for the current ratio
        estimates[ratio] = total_days_to_100
    
    return estimates  # Return the dictionary of estimates

if __name__ == "__main__":
    estimates_integrated = integrated_arima_forecasting()  # Call the forecasting function
    print("\n-------------------------------------------\n")
    print("ARIMA Prediction Model")
    print("\n-------------------------------------------\n")
    # Print the estimated days to reach 100% degradation for each ratio
    for ratio, days in estimates_integrated.items():
        print(f"{ratio}: {days} days")
    print("\n-------------------------------------------\n")

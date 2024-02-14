# Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def integrated_arima_forecasting():
    # Data definition
    data = pd.DataFrame({
        'Ratio': ['Control', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'Ratio 4', 'Ratio 5', 'Ratio 6'],
        '10-20 days': [12.446667, 14.375, 14.668333, 11.656667, 11.27, 12.595, 13.205],
        '20-30 days': [26.66, 22.145, 24.988333, 25.41, 26.018333, 22.683333, 26.211667],
        '30-40 days': [10.208333, 12.873333, 8.651667, 11.583333, 9.955, 17.011667, 15.63],
        '40-50 days': [11.905, 13.458333, 19.376667, 23.05, 26.223333, 22.938333, 22.758333]
    }).set_index('Ratio')
    
    estimates = {}
    
    for ratio, series in data.iterrows():
        series = series.astype(float)
        
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=4)
        
        total_cumulative_degradation = series.sum() + forecast.sum()
        
        average_degradation_forecast = forecast.mean()
        remaining_degradation_needed = 100 - total_cumulative_degradation
        additional_intervals_needed = remaining_degradation_needed / average_degradation_forecast if average_degradation_forecast > 0 else 0
        
        total_intervals_to_100 = len(series) + len(forecast) + additional_intervals_needed
        total_days_to_100 = total_intervals_to_100 * 10
        
        estimates[ratio] = total_days_to_100
    
    return estimates

if __name__ == "__main__":
    estimates_integrated = integrated_arima_forecasting()
    print("\n-------------------------------------------\n")
    print("ARIMA Prediction Model")
    print("\n-------------------------------------------\n")
    for ratio, days in estimates_integrated.items():
        print(f"{ratio}: {days} days")
    print("\n-------------------------------------------\n")





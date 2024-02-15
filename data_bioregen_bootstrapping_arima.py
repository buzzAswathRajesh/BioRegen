import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def estimate_days_for_100_percent_degradation(data, n_iterations=100, threshold=100):
    # Dictionary to hold the final days estimate for each ratio
    final_days_to_100_percent = {}


    for ratio, series in data.iterrows():
        series = series.astype(float)
        initial_sum = series.sum()


        # List to collect the estimated days to reach 100% for each bootstrap iteration
        estimated_days_list = []


        for iteration in range(n_iterations):
            cumulative_degradation = initial_sum
            days_count = len(series) * 10  # Starting point, assuming 10 days per interval already observed


            while cumulative_degradation < threshold:
                # Bootstrap resampling
                resampled_series = series.sample(frac=1, replace=True, random_state=iteration)
                model = ARIMA(resampled_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)


                forecasted_degradation = forecast.iloc[0]
                cumulative_degradation += forecasted_degradation


                # Estimate additional days based on the forecasted degradation
                days_count += 10  # Assuming each forecast step represents an additional 10 days


                if cumulative_degradation >= threshold:
                    # Adjust days based on the exact amount of degradation required to reach 100%
                    excess_degradation = cumulative_degradation - threshold
                    exact_days_adjustment = 10 * (excess_degradation / forecasted_degradation)
                    days_count -= exact_days_adjustment


            estimated_days_list.append(days_count)


        # Calculate the mean of the estimated days from all iterations for this ratio
        mean_estimated_days = np.mean(estimated_days_list)
        final_days_to_100_percent[ratio] = mean_estimated_days


    return final_days_to_100_percent


# Example usage with the defined data
if __name__ == "__main__":
    data = pd.DataFrame({
        'Ratio': ['Control', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'Ratio 4', 'Ratio 5', 'Ratio 6'],
        '0-10 days': [8.54, 10.20, 10.54, 12.74, 13.57, 14.81, 15.18],
        '10-20 days': [12.446667, 14.375, 14.668333, 11.656667, 11.27, 12.595, 13.205],
        '20-30 days': [26.66, 22.145, 24.988333, 25.41, 26.018333, 22.683333, 26.211667],
        '30-40 days': [10.208333, 12.873333, 8.651667, 11.583333, 9.955, 17.011667, 15.63],
        '40-50 days': [11.905, 13.458333, 19.376667, 23.05, 26.223333, 22.938333, 22.758333]
    }).set_index('Ratio')
   
    degradation_days = estimate_days_for_100_percent_degradation(data)
    print("\n-------------------------------------------\n")
    print("Estimated Exact Days to Reach 100% Degradation")
    print("\n-------------------------------------------\n")
    for ratio, days in degradation_days.items():
        print(f"{ratio}: {days:.2f} days")
    print("\n-------------------------------------------\n")

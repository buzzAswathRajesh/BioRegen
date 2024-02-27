import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to ensure cleaner output

# Function to estimate days until 100% degradation for each series in the DataFrame
def estimate_days_for_100_percent_degradation(data, n_iterations=100, threshold=100):
    # Dictionary to hold the final days estimate for each ratio
    final_days_to_100_percent = {}

    # Iterate over each row in the DataFrame, treating each as a separate time series
    for ratio, series in data.iterrows():
        series = series.astype(float)  # Ensure data is in float format
        initial_sum = series.sum()  # Initial sum of degradation

        # List to collect the estimated days to reach 100% for each bootstrap iteration
        estimated_days_list = []

        # Perform bootstrap iterations to simulate various paths to 100% degradation
        for iteration in range(n_iterations):
            cumulative_degradation = initial_sum
            days_count = len(series) * 10  # Start counting from the observed days (assuming 10 days per interval)

            # Continue forecasting until cumulative degradation reaches the threshold (100%)
            while cumulative_degradation < threshold:
                # Resample the series with replacement to simulate uncertainty
                resampled_series = series.sample(frac=1, replace=True, random_state=iteration)
                model = ARIMA(resampled_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)  # Forecast the next step

                forecasted_degradation = forecast.iloc[0]
                cumulative_degradation += forecasted_degradation

                days_count += 10  # Assume each forecast step represents an additional 10 days

                # If cumulative degradation exceeds the threshold, adjust the final day count
                if cumulative_degradation >= threshold:
                    excess_degradation = cumulative_degradation - threshold
                    exact_days_adjustment = 10 * (excess_degradation / forecasted_degradation)
                    days_count -= exact_days_adjustment  # Adjust days based on the excess degradation

            estimated_days_list.append(days_count)  # Collect the estimated days for this iteration

        # Calculate the mean of the estimated days from all iterations for this ratio
        mean_estimated_days = np.mean(estimated_days_list)
        final_days_to_100_percent[ratio] = mean_estimated_days  # Store the mean estimate

    return final_days_to_100_percent  # Return the final estimates for all ratios

# Example usage
if __name__ == "__main__":
    # Sample data definition
    data = pd.DataFrame({
        'Ratio': ['Control', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'Ratio 4', 'Ratio 5', 'Ratio 6'],
        '0-10 days': [8.54, 10.20, 10.54, 12.74, 13.57, 14.81, 15.18],
        '10-20 days': [12.446667, 14.375, 14.668333, 11.656667, 11.27, 12.595, 13.205],
        '20-30 days': [26.66, 22.145, 24.988333, 25.41, 26.018333, 22.683333, 26.211667],
        '30-40 days': [10.208333, 12.873333, 8.651667, 11.583333, 9.955, 17.011667, 15.63],
        '40-50 days': [11.905, 13.458333, 19.376667, 23.05, 26.223333, 22.938333, 22.758333]
    }).set_index('Ratio')  # Set 'Ratio' as the index for the DataFrame

    # Call the estimation function
    degradation_days = estimate_days_for_100_percent_degradation(data)
    # Print the estimated days for each ratio
    print("\n-------------------------------------------\n")
    print("Estimated Exact Days to Reach 100% Degradation")
    print("\n-------------------------------------------\n")
    for ratio, days in degradation_days.items():
        print(f"{ratio}: {days:.2f} days")
    print("\n-------------------------------------------\n")

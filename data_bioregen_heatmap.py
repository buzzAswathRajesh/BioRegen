# Import necessary libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the index (samples) and attributes (measurements) for the DataFrame
index = ['Control', 'Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5', 'Sample6']
attributes = ['Tensile Strength', 'Flexibility', 'Water Resistance', 'Phytotoxicity', 'Biodegradability']

# Input data as a dictionary, with keys as attributes and values as lists of measurements for each sample
data = {
    'Tensile Strength': [53.77, 58.36, 65.96, 72.84, 76.69, 81.27, 87.07],
    'Flexibility': [5.80, 6.60, 6.62, 7.37, 7.72, 8.83, 9.95],
    'Water Resistance': [88.38, 86.83, 85.89, 84.24, 82.39, 80.40, 78.39],
    'Phytotoxicity': [14.18, 14.22, 14.23, 14.47, 14.35, 14.18, 14.42],
    'Biodegradability': [69.76, 73.06, 78.22, 84.44, 87.04, 90.04, 92.98]
}

# Creating a DataFrame from the dictionary, setting the index to the sample names
df = pd.DataFrame(data, index=index)

# Standardizing the data by subtracting the mean and dividing by the standard deviation for each attribute
data_standardized = (df - df.mean()) / df.std()

# Begin plotting with adjusted figure size
plt.figure(figsize=(20, 10))  # Adjusted figure size for better label fit

# Create the heatmap using standardized data, specifying the colormap and aspect ratio
heatmap = plt.imshow(data_standardized, cmap='ocean', aspect='auto')

# Set the background color of the figure to white
plt.gcf().set_facecolor('white')

# Add a color bar to the figure to indicate the scale of standardized values, with customization
cbar = plt.colorbar(heatmap, label='Standardized Values', orientation='horizontal', pad=0.15)

# Set the x-axis and y-axis ticks to the column and row names respectively, making labels bold
plt.xticks(np.arange(data_standardized.shape[1]), df.columns, rotation=0, fontsize=23, fontweight='bold')
plt.yticks(np.arange(data_standardized.shape[0]), df.index, fontsize=24, fontweight='bold')

# Loop through the data to add text annotations in each cell
for i in range(data_standardized.shape[0]):  # Iterate over rows (samples)
    for j in range(data_standardized.shape[1]):  # Iterate over columns (attributes)
        # Choose text color for better visibility based on cell's background color
        text_color = 'white' if abs(data_standardized.iloc[i, j]) > 0.5 else 'black'
        # Add the text annotation, showing the original value with two decimal places
        plt.text(j, i, f'{df.iloc[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=23)

# Add a title to the heatmap, with customization for font size and weight
plt.title('BIOREGEN POLYMER HEATMAP ANALYSIS', fontsize=23, fontweight='bold', pad=30)

# Adjust the layout to ensure all elements are clearly visible without overlap
plt.tight_layout()

# Save the figure before showing it
plt.savefig('bioregen_polymer_heatmap.jpeg', dpi=95, format='jpeg')

# Display the plot
plt.show()

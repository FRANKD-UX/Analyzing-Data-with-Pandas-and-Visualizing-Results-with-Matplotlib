# Data Analysis with Pandas and Matplotlib
# Student Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("muted")

# -----------------------------------------------------
# PART 1: DATA LOADING AND EXPLORATION
# -----------------------------------------------------
print("PART 1: DATA LOADING AND EXPLORATION")
print("-" * 40)

# Load the Iris dataset (a classic dataset for demonstration)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows of the dataset
print("\nFirst 5 rows of the Iris dataset:")
print(df.head())

# Check the shape of the dataset
print(f"\nDataset dimensions: {df.shape[0]} rows and {df.shape[1]} columns")

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic dataset information
print("\nBasic dataset information:")
print(df.info())

# -----------------------------------------------------
# PART 2: BASIC DATA ANALYSIS
# -----------------------------------------------------
print("\nPART 2: BASIC DATA ANALYSIS")
print("-" * 40)

# Compute basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Analyze data by species
print("\nMean measurements by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Find min and max values for each feature
print("\nMinimum values for each feature by species:")
species_min = df.groupby('species').min()
print(species_min)

print("\nMaximum values for each feature by species:")
species_max = df.groupby('species').max()
print(species_max)

# -----------------------------------------------------
# PART 3: DATA VISUALIZATION
# -----------------------------------------------------
print("\nPART 3: DATA VISUALIZATION")
print("-" * 40)

# Set up a figure with multiple subplots
plt.figure(figsize=(12, 10))

# 1. Bar Chart: Average Sepal Length by Species
plt.subplot(2, 2, 1)
species_list = df['species'].unique()
colors = ['skyblue', 'lightgreen', 'salmon']

# Calculate means for sepal length by species
sepal_length_means = [df[df['species'] == species]['sepal length (cm)'].mean() 
                      for species in species_list]

# Create bar chart
plt.bar(species_list, sepal_length_means, color=colors)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Histogram: Distribution of Petal Length
plt.subplot(2, 2, 2)
for i, species in enumerate(species_list):
    plt.hist(df[df['species'] == species]['petal length (cm)'], 
             alpha=0.6, 
             bins=10, 
             label=species,
             color=colors[i])
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

# 3. Scatter Plot: Sepal Length vs Sepal Width
plt.subplot(2, 2, 3)
for i, species in enumerate(species_list):
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], 
                subset['sepal width (cm)'],
                label=species,
                color=colors[i],
                alpha=0.7)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

# 4. Line Chart: Feature comparison across species
plt.subplot(2, 2, 4)
# Transpose the species means for plotting
species_means_plot = species_means.T
species_means_plot.plot(marker='o')
plt.title('Average Measurements by Species')
plt.xlabel('Measurements')
plt.ylabel('Value (cm)')
plt.grid(linestyle='--', alpha=0.7)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('iris_analysis_plots.png')  # Save the plots to a file
plt.show()  # Display the plots

# -----------------------------------------------------
# PART 4: FINDINGS AND OBSERVATIONS
# -----------------------------------------------------
print("\nPART 4: FINDINGS AND OBSERVATIONS")
print("-" * 40)

# Calculate the range of each feature by species
print("\nFeature ranges by species:")
for feature in iris.feature_names:
    print(f"\n{feature.upper()}:")
    for species in species_list:
        species_data = df[df['species'] == species][feature]
        print(f"  {species}: {species_data.min():.2f} to {species_data.max():.2f} cm (range: {species_data.max() - species_data.min():.2f} cm)")

# Key observations
print("\nKEY OBSERVATIONS:")
print("1. Iris setosa has the smallest petal dimensions but largest sepal width on average.")
print("2. Iris virginica has the largest petal and sepal length on average.")
print("3. Petal length and width show clearer separation between species than sepal measurements.")
print("4. There is overlap in sepal measurements between versicolor and virginica species.")
print("5. Setosa is the most easily distinguishable species based on petal size.")

# Calculate correlation between features
print("\nCorrelation between features:")
correlation = df.drop('species', axis=1).corr()
print(correlation)

# Additional analysis: Create a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Iris Features')
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png')
plt.show()

print("\nAnalysis complete! Two image files have been saved: 'iris_analysis_plots.png' and 'iris_correlation_heatmap.png'")

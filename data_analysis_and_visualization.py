import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map the target labels to species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows of the dataset
print(df.head())

# Check for data types and missing values
print(df.info())
print(df.isnull().sum())

# Clean the data (if any missing values)
df = df.dropna()

# Basic Data Analysis
print(df.describe())
print(df.groupby('species').mean())

# Data Visualization

# 1. Line Chart (Example: If time series data is available)
plt.figure(figsize=(10, 6))
df.index = pd.to_datetime(df.index)  # Simulating a time index (replace with real time data)
plt.plot(df.index, df['sepal length (cm)'])
plt.title('Sepal Length Over Time')
plt.xlabel('Time')
plt.ylabel('Sepal Length (cm)')
plt.show()

# 2. Bar Chart: Average Sepal Length by Species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# 3. Histogram: Distribution of Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

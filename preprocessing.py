
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = "D:/forestfires.csv"
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Initial Dataset Overview:")
print(df.info())
print(df.head())

# 1. Handle Missing Values
print("\nChecking for Missing Values:")
print(df.isnull().sum())


# 2. Scale and Normalize Numerical Features
numerical_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nScaled Numerical Features:")
print(df[numerical_features].head())

# 3. Encode Categorical Variables
categorical_features = ['month', 'day']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))


df = df.drop(columns=categorical_features)
df = pd.concat([df, encoded_df], axis=1)

print("\nEncoded Categorical Features:")
print(df.head())

df['log_area'] = np.log1p(df['area'])

# Drop the original 'area' column
df = df.drop(columns=['area'])

print("\nLog-Transformed Target Variable (log_area):")
print(df[['log_area']].head())

# 5. Save Cleaned Dataset
# ------------------------------------------------
# Save the cleaned dataset to a new CSV file for next steps
cleaned_data_path = "cleaned_forestfires.csv"
df.to_csv(cleaned_data_path, index=False)

print("\nCleaned dataset saved to 'cleaned_forestfires.csv'")





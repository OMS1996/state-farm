import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Assuming imputer and scaler are already fitted and available
numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Imputer for numeric columns
non_numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # Imputer for non-numeric columns
std_scaler = StandardScaler()  # Replace with the actual scaler used

def preprocess_data(features_dict):
    # Convert the dictionary to a DataFrame
    data = pd.DataFrame([features_dict])

    # Feature engineering for money and percents
    for col in ['x12', 'x63']:
        if col in data.columns:
            data[col] = data[col].replace({'\$': '', ',': '', '\(': '-', '\)': '','%':''}, regex=True).astype(float)

    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns

    # Handle NaN and infinity values for numeric columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fit the numeric imputer to the data
    numeric_imputer.fit(data[numeric_cols])
    
    # Transform numeric columns
    data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])

    # Handle NaN for non-numeric columns
    data[non_numeric_cols] = non_numeric_imputer.transform(data[non_numeric_cols])

    # Standardization for numeric columns
    data[numeric_cols] = std_scaler.transform(data[numeric_cols])

    # Handling variables for dummy creation
    vars = ['x5', 'x31', 'x81', 'x82']
    for var in vars:
        if var in data.columns:
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
            data = pd.concat([data, dummies], axis=1)
        else:
            print(f"Warning: {var} not found in input data.")

    return data


# Load the CSV file once for all tests
df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')

# Try it on all the data
test_data = df.to_dict(orient='records')
processed_data = preprocess_data(test_data)
print(processed_data.head())

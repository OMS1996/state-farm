import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Function for initial cleanup and conversion
def initial_preprocess(df):
    for col in ['x12', 'x63']:
        if col in df.columns:
            df[col] = df[col].replace({'\$': '', ',': '', '\(': '-', '\)': '','%':''}, regex=True).astype(float)
    return df

# Function to preprocess data for imputation and scaling
def preprocess_data(data, numeric_imputer, non_numeric_imputer, std_scaler):
    # Convert the dictionary to a DataFrame
    #data = pd.DataFrame([features_dict])

    data = initial_preprocess(data)

    # Separate numeric and non-numeric columns, excluding dummy variables
    numeric_cols = data.select_dtypes(include=['number']).columns.difference(['x5', 'x31', 'x81', 'x82'])
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.difference(['x5', 'x31', 'x81', 'x82'])

    # Handle NaN and infinity values for numeric columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Transform numeric columns using already fitted imputer
    data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])

    # Handle NaN for non-numeric columns
    data[non_numeric_cols] = non_numeric_imputer.transform(data[non_numeric_cols])

    # Standardization for numeric columns
    data[numeric_cols] = std_scaler.transform(data[numeric_cols])

    # Create dummies for specified categorical variables
    vars = ['x5', 'x31', 'x81', 'x82']
    for var in vars:
        if var in data.columns:
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
            data = pd.concat([data, dummies], axis=1)
        else:
            print(f"Warning: {var} not found in input data.")

    return data

# Load and preprocess training data for fitting imputer and scaler
train_df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv')
train_df = initial_preprocess(train_df)

# Load the test data
test_df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')
test_df = initial_preprocess(test_df)

# Find common columns in training and test data and convert to a list
common_columns = list(set(train_df.columns).intersection(set(test_df.columns)))

# Align columns in training data with test data
train_df = train_df[common_columns]

# Fit the numeric imputer
numeric_cols = list(set(common_columns).difference(['x5', 'x31', 'x81', 'x82', 'y']))
numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numeric_imputer.fit(train_df[numeric_cols])

# Fit the non-numeric imputer for categorical variables
non_numeric_cols = ['x5', 'x31', 'x81', 'x82']
non_numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
non_numeric_imputer.fit(train_df[non_numeric_cols])

# Fit the scaler
train_imputed = pd.DataFrame(numeric_imputer.transform(train_df[numeric_cols]), columns=numeric_cols)
std_scaler = StandardScaler()
std_scaler.fit(train_imputed)

# Test
# Ensure that the columns in test_df are exactly the same as in train_df
# Preprocess the test data using the fitted imputer and scaler
print("HERE !")
test_df = preprocess_data(test_df, numeric_imputer, non_numeric_imputer, std_scaler)

# Check for NaN or infinity values in processed_data
if test_df.isnull().values.any() or np.isinf(test_df).any():
    raise ValueError("Invalid input data after preprocessing.")

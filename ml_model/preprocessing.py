import numpy as np
import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def initial_preprocess(df):
    for col in ['x12', 'x63']:
        if col in df.columns:
            df[col] = df[col].replace({'\$': '', ',': '', '\(': '-', '\)': '', '%':''}, regex=True).astype(float)
    return df

def create_preprocessors(data_source):
    if isinstance(data_source, str):
        train_df = pd.read_csv(data_source)
    else:
        train_df = data_source

    train_df = initial_preprocess(train_df)

    numeric_cols = train_df.columns.difference(['x5', 'x31', 'x81', 'x82', 'y'])
    non_numeric_cols = ['x5', 'x31', 'x81', 'x82']

    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    numeric_imputer.fit(train_df[numeric_cols])

    non_numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    non_numeric_imputer.fit(train_df[non_numeric_cols])

    train_imputed = pd.DataFrame(numeric_imputer.transform(train_df[numeric_cols]), columns=numeric_cols)
    std_scaler = StandardScaler()
    std_scaler.fit(train_imputed)

    # Save imputers and scaler
    with open('numeric_imputer.pkl', 'wb') as f:
        pickle.dump(numeric_imputer, f)
    with open('non_numeric_imputer.pkl', 'wb') as f:
        pickle.dump(non_numeric_imputer, f)
    with open('std_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    return numeric_imputer, non_numeric_imputer, std_scaler

def preprocess_data(data, numeric_imputer, non_numeric_imputer, std_scaler):
    data = initial_preprocess(data)

    numeric_cols = data.select_dtypes(include=['number']).columns.difference(['x5', 'x31', 'x81', 'x82'])
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.difference(['x5', 'x31', 'x81', 'x82'])

    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])
    data[non_numeric_cols] = non_numeric_imputer.transform(data[non_numeric_cols])
    data[numeric_cols] = std_scaler.transform(data[numeric_cols])

    vars = ['x5', 'x31', 'x81', 'x82']
    for var in vars:
        if var in data.columns:
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
            data = pd.concat([data, dummies], axis=1)
        else:
            print(f"Warning: {var} not found in input data.")

    return data

def run_preprocess(input_data):
    if isinstance(input_data, str):
        data = pd.read_csv(input_data)
    elif isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        raise ValueError("Input data must be a string (URL), dictionary, or DataFrame.")

    # Check if imputers and scaler are available
    if (os.path.exists('numeric_imputer.pkl') and 
        os.path.exists('non_numeric_imputer.pkl') and 
        os.path.exists('std_scaler.pkl')):
        with open('numeric_imputer.pkl', 'rb') as f:
            numeric_imputer = pickle.load(f)
        with open('non_numeric_imputer.pkl', 'rb') as f:
            non_numeric_imputer = pickle.load(f)
        with open('std_scaler.pkl', 'rb') as f:
            std_scaler = pickle.load(f)
    else:
        # If not, create them using the default training data
        default_train_url = 'https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv'
        numeric_imputer, non_numeric_imputer, std_scaler = create_preprocessors(default_train_url)

    return preprocess_data(data, numeric_imputer, non_numeric_imputer, std_scaler)

# Example usage
# processed_data = run_preprocess('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')


# Load and preprocess training data for fitting imputer and scaler
train_df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv')
train_df = initial_preprocess(train_df)
numeric_imputer, non_numeric_imputer, std_scaler = create_preprocessors(train_df)

# Load the test data
test_df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')
test_df = initial_preprocess(test_df)

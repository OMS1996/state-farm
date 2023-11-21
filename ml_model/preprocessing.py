import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union


VARIABLES = ['x5_saturday',
 'x81_July',
 'x81_December',
 'x31_japan',
 'x81_October',
 'x5_sunday',
 'x31_asia',
 'x81_February',
 'x91',
 'x81_May',
 'x5_monday',
 'x81_September',
 'x81_March',
 'x53',
 'x81_November',
 'x44',
 'x81_June',
 'x12',
 'x5_tuesday',
 'x81_August',
 'x81_January',
 'x62',
 'x31_germany',
 'x58',
 'x56']

def initial_preprocess(df: pd.DataFrame):
    for col in ['x12', 'x63']:
        if col in df.columns:
            df[col] = df[col].replace({'\$': '', ',': '', '\(': '-', '\)': '', '%':''}, regex=True).astype(float)
    return df

def create_preprocessors(data_source: Union[str, pd.DataFrame] = 'https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv'):
    if isinstance(data_source, str):
        train_df = pd.read_csv(data_source)
    else:
        train_df = data_source

    train_df = initial_preprocess(train_df)

    numeric_cols = train_df.columns.difference(['x5', 'x31', 'x81', 'x82', 'y'])

    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    numeric_imputer.fit(train_df[numeric_cols])

    train_imputed = pd.DataFrame(numeric_imputer.transform(train_df[numeric_cols]), columns=numeric_cols)
    std_scaler = StandardScaler()
    std_scaler.fit(train_imputed)

    # Save imputers and scaler
    with open('numeric_imputer.pkl', 'wb') as f:
        pickle.dump(numeric_imputer, f)
    with open('std_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    return numeric_imputer, std_scaler

def preprocess_data(data: Union[Dict, pd.DataFrame], numeric_imputer: SimpleImputer, std_scaler: StandardScaler, selected_features: List = VARIABLES) -> pd.DataFrame:
    """
    Preprocesses the input data by performing the following steps:
    1. Initial preprocessing of the data.
    2. Replacing infinity values with NaN.
    3. Imputing missing values with mean for numeric columns and most frequent value for non-numeric columns.
    4. Scaling numeric columns using standardization.
    5. Creating dummy variables for categorical variables.
    
    Parameters:
    - data (Union[Dict, pd.DataFrame]): The input data to be preprocessed.
    - numeric_imputer (SimpleImputer): The imputer object used for imputing missing values in numeric columns.
    - std_scaler (StandardScaler): The scaler object used for scaling numeric columns.
    
    Returns:
    - preprocessed_data (pd.DataFrame): The preprocessed data.
    """
    # Convert dictionary to DataFrame if necessary
    if isinstance(data, Dict):
        data = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a dictionary or DataFrame.")
    
    # Initial preprocessing
    data = initial_preprocess(data) # Initial preprocessing

    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.difference(['x5', 'x31', 'x81', 'x82'])
    
    # Replace infinity values with NaN
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Impute: replace NaN with mean for numeric columns and most frequent value for non-numeric columns
    data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])

    # Scale: standardize numeric columns
    data[numeric_cols] = std_scaler.transform(data[numeric_cols])


    # Create dummy variables for categorical variables
    vars = ['x5', 'x31', 'x81', 'x82']
    for var in vars:
        if var in data.columns:
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(columns=[var], inplace=True)  # Drop the original column
        else:
            print(f"Warning: {var} not found in input data.")

        # Add missing dummy variables if they are not in the data
        for feature in selected_features:
            if feature not in data.columns:
                data[feature] = 0  # Add missing feature with default value of 0

        # Ensure that only the selected features are used
        final_data = data[selected_features]

        return final_data


def run_preprocess(input_data: Union[str, Dict, pd.DataFrame, pd.Series]):
    """Preprocess input data."""
    if isinstance(input_data, str):
        data = pd.read_csv(input_data)
        if 'y' in data.columns:
            data = data.drop(columns=['y']) # Remove target variable if present
    elif isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    elif isinstance(input_data, pd.Series):
        data = pd.DataFrame([input_data])
    else:
        raise ValueError("Input data must be a string (URL), dictionary, or DataFrame.")

    
    # Check if imputers and scaler are available
    if (os.path.exists('numeric_imputer.pkl') and 
        os.path.exists('std_scaler.pkl')):
        with open('numeric_imputer.pkl', 'rb') as f:
            numeric_imputer = pickle.load(f)
        with open('std_scaler.pkl', 'rb') as f:
            std_scaler = pickle.load(f)
    else:
        # If not, create them using the default training data
        default_train_url = 'https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv'
        numeric_imputer, std_scaler = create_preprocessors(default_train_url)

    # Preprocess the data
    preprocessed_data = preprocess_data(data, numeric_imputer, std_scaler)

    return preprocessed_data
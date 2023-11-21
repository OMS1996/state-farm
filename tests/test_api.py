# Tests for the API endpoints
import requests
import numpy as np
import pandas as pd
import pytest
import json
from typing import List, Dict
import traceback
#from app.dependencies import BASE_URL, TEST_DATA_URL
#from ml_model.preprocessing import dataframe_to_dict_list



def dataframe_to_dict_list(df: pd.DataFrame) -> List[Dict[str, float]]:
    """
    Converts a DataFrame into a list of dictionaries.
    """

    # Convert DataFrame to a list of dictionaries
    dict_list = df.to_dict(orient='records')

    return dict_list

# Load the CSV file once for all tests
df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')


def test_batch_prediction():
    
    print("INSIDE TEST BATCH PREDICTION")
    # Ensure the DataFrame is in a JSON-compliant format
    df.replace([np.inf, -np.inf, np.nan], None, inplace=True)
    
    batch_data = dataframe_to_dict_list(df.iloc[:5])

    # Print type of data and nested data to ensure it is JSON-compliant
    print(f"\n\nBatch data: {batch_data}")
    print(f"Type of data: {type(batch_data)}")
    print(f"Type of nested data: {type(batch_data[0])}") 
    
    response = None
    try:
        response = requests.post("http://0.0.0.0:8000/predict", json=batch_data)
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 200
    except Exception:
        # This will print the type, value, and traceback of the current exception
        traceback.print_exc()

    if response is None:
        print("No response received from the API.")



test_batch_prediction()
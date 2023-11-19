# Tests for the API endpoints
import requests
import pandas as pd
import pytest
import json
from typing import List, Dict
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

print(df.iloc[5])
print("ANYTHING")
#single_input = dataframe_to_dict_list(df.iloc[0])
batch_input = dataframe_to_dict_list(df)



def test_batch_prediction():
    """
    Test the API endpoint with multiple data points.
    """
    # Prepare multiple rows of data
    batch_data = dataframe_to_dict_list(df.iloc[:5])
    payload = {"input_data": json.dumps(batch_data)}
    print(type(batch_data))
    # Make the POST request to the predict endpoint
    try:
        response = requests.post("http://0.0.0.0:8000/predict", json=payload)
        print(response.json())
    except Exception as e:
        print(e)

    # Assertions for batch predictions
    assert response.status_code == 200

def test_single_prediction():
    """
    Test the API endpoint with a single data point.
    """
    # Prepare a single row of data
    single_data = dataframe_to_dict_list(df.iloc[[0]])[0]
    payload = {"data": [single_data]}  # Wrap in a list to match the expected input format

    # Make the POST request to the predict endpoint
    response = requests.post("http://0.0.0.0:8000/predict", data=payload)

    # Assertions to validate response
    assert response.status_code == 200


test_batch_prediction()
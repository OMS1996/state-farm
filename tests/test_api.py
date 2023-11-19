# Tests for the API endpoints
import requests
import pandas as pd
import pytest
import json
from typing import List, Dict
from app.dependencies import BASE_URL, TEST_DATA_URL
from ml_model.preprocessing import dataframe_to_dict_list

# Load the CSV file once for all tests
df = pd.read_csv(TEST_DATA_URL)
single_input = dataframe_to_dict_list(df.iloc[0])
batch_input = dataframe_to_dict_list(df)


def test_batch_prediction():
    """
    Test the API endpoint with multiple data points.
    """
    # Prepare multiple rows of data
    batch_data = dataframe_to_dict_list(df.iloc[:5])
    payload = {"data": batch_data}

    # Make the POST request to the predict endpoint
    response = requests.post("http://localhost:8080/predict", json=payload)

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
    response = requests.post("http://localhost:8080/predict", json=payload)

    # Assertions to validate response
    assert response.status_code == 200



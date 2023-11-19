# Tests for the API endpoints
import requests
import pandas as pd
import pytest
import json
from app.dependencies import BASE_URL, TEST_DATA_URL

# Load the CSV file once for all tests
df = pd.read_csv(TEST_DATA_URL)
def test_single_prediction():
    """
    Test the API endpoint with a single data point
    """
    test_data = df.iloc[0].to_dict()
    payload = {"data": [test_data]}  # Note the change here: wrapping the dictionary in a list

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    # Update assertions according to your actual response structure

def test_batch_prediction():
    """
    Test the API endpoint with multiple data points
    """
    test_data = df.iloc[:5].to_dict(orient='records')
    payload = {"data": test_data}

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    # Update assertions according to your actual response structure

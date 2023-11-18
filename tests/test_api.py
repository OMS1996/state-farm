# Tests for the API endpoints
import requests
import pandas as pd
import pytest
from app.dependencies import BASE_URL
import json

# Load the CSV file once for all tests
df = pd.read_csv('https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv')

def test_single_prediction():
    """
    Test the API endpoint with a single data point
    """

    # Prepare a single row of data
    test_data = df.iloc[0].to_dict()
    print(test_data)
    payload = {"data": [{"features": test_data}]}

    # Serialize the payload to JSON and print it for debugging
    json_payload = json.dumps(payload, indent=4)

    # Make the POST request
    response = requests.post(BASE_URL + "/predict", json=payload)

    # Assertions to validate response
    assert response.status_code == 200
    assert "probability" in response.json()[0]
    assert "predicted_class" in response.json()[0]

def test_batch_prediction():
    """
    Test the API endpoint with multiple data points
    """

    # Prepare multiple rows of data
    test_data = df.iloc[:5].to_dict(orient='records')
    payload = {"data": [{"features": data} for data in test_data]}

    # Serialize the payload to JSON and print it for debugging
    json_payload = json.dumps(payload, indent=4)
    print("JSON Payload:")
    print(json_payload)

    # Make the POST request
    response = requests.post(BASE_URL + "/predict", json=payload)

    # Assertions for batch predictions
    assert response.status_code == 200
    assert all("probability" in result for result in response.json())
    assert all("predicted_class" in result for result in response.json())





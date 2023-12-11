# Tests for the API endpoints
import requests
import numpy as np
import pandas as pd
import pytest
import json
from typing import List, Dict
import traceback

# variables
VARIABLES = [
    "x5_saturday",
    "x81_July",
    "x81_December",
    "x31_japan",
    "x81_October",
    "x5_sunday",
    "x31_asia",
    "x81_February",
    "x91",
    "x81_May",
    "x5_monday",
    "x81_September",
    "x81_March",
    "x53",
    "x81_November",
    "x44",
    "x81_June",
    "x12",
    "x5_tuesday",
    "x81_August",
    "x81_January",
    "x62",
    "x31_germany",
    "x58",
    "x56",
]

def test_root_endpoint():
    """
    Test the root endpoint of the API.
    """
    response = requests.get("http://0.0.0.0:1313/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API! By Omar M. Hussein"}


def dataframe_to_dict_list(df: pd.DataFrame) -> List[Dict[str, float]]:
    """
    Converts a DataFrame into a list of dictionaries and replaces NaNs with None.
    """
    dict_list = df.to_dict(orient='records')
    
    # Replace NaNs in each dictionary with None
    for item in dict_list:
        for key, value in item.items():
            if pd.isna(value):
                item[key] = None
    
    return dict_list



# Load the CSV file once for all tests
df = pd.read_csv(
    "https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv"
)


# THIS IS AN INTENSE TEST BECAUSE I AM USING THE CSV FILE TO TEST THE API.
def test_batch_prediction(df):

    # Ensure the DataFrame is in a JSON-compliant format
    df.replace([np.inf, -np.inf, np.nan], None, inplace=True)
    
    # Batch data
    batch_data = dataframe_to_dict_list(df)
    
    # Create the payload
    payload = {
        "input_data": batch_data[0:10],  # The batch data you're already preparing
        "selected_variables": VARIABLES,  # The list of feature names used in the model
    }
    response = None
    try:
        response = requests.post("http://0.0.0.0:1313/predict", json=payload)
        print(response.json())
        assert response.status_code == 200
    except Exception:
        # This will print the type, value, and traceback of the current exception
        traceback.print_exc()

    if response is None:
        print("No response received from the API. Please ensure it is running.")

test_batch_prediction(df)
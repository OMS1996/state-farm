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



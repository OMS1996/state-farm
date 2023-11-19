# Dependencies, configurations, or constants
import os

# Assuming the model file is in the root of the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# http://localhost:8000
API_PREFIX = "/api"
BASE_URL = "http://localhost:8000"

# Data URLs
TRAIN_DATA_URL = "https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv"
TEST_DATA_URL = "https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv"


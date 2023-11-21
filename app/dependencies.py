# Dependencies, configurations, or constants
import os

# Assuming the model file is in the root of the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
BASE_URL = "http://localhost:1313"

# Data URLs
TRAIN_DATA_URL = "https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_train.csv"
TEST_DATA_URL = "https://raw.githubusercontent.com/OMS1996/state-farm/main/data/exercise_26_test.csv"


# Varaibles for model.
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

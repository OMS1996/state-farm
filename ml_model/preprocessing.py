# Data preprocessing functions
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Imputer and scaler initialization (replace with actual objects used)
imputer = SimpleImputer()  # Replace with the actual imputer used
std_scaler = StandardScaler()  # Replace with the actual scaler used

def preprocess_data(data):
    # Imputation
    imputed_data = pd.DataFrame(imputer.transform(data.drop(columns=['x5', 'x31', 'x81' ,'x82'])), columns=data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    
    # Standardization
    standardized_data = pd.DataFrame(std_scaler.transform(imputed_data), columns=imputed_data.columns)
    
    # Creating dummies for categorical variables
    for var in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
        standardized_data = pd.concat([standardized_data, dummies], axis=1, sort=False)
    
    return standardized_data


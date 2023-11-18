import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Assuming imputer and scaler are already fitted and available
imputer = SimpleImputer()  # Replace with the actual imputer used
std_scaler = StandardScaler()  # Replace with the actual scaler used

def preprocess_data(features_dict):
    
    # Convert the dictionary to a DataFrame
    data = pd.DataFrame([features_dict])

    # Imputation and Standardization.
    imputed_data = pd.DataFrame(imputer.transform(data), columns=data.columns)
    standardized_data = pd.DataFrame(std_scaler.transform(imputed_data), columns=imputed_data.columns)

    # Handling variables for dummy creation
    # Update this part based on your actual categorical features
    vars = ['x5', 'x31', 'x81', 'x82']
    for var in vars:
        if var in data.columns:
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var, prefix_sep='_', dummy_na=True)
            standardized_data = pd.concat([standardized_data, dummies], axis=1)
        else:
            # Handling missing categorical variables
            # For now it is a simple print
            print(f"Warning: {var} not found in input data.")

    return standardized_data

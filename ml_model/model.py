# Code for loading and running the ML model
import pandas as pd
import pickle

# Load the model (assuming it's a pickle file)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Prediction function
def predict(model, data):
    # Assuming the model is a logistic regression from statsmodels
    predicted_probs = model.predict(data)
    return predicted_probs

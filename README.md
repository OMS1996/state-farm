# State Farm Machine Learning API Deployment

## Overview

This repository contains the resources for deploying a machine learning model as an API for State Farm. It's designed to provide real-time predictions via a logistic regression model. The deployment is streamlined through Docker, emphasizing ease of use and consistency across different environments. Additionally, a CI/CD pipeline is integrated for automated testing and deployment.

## Reasoning

This section explains the critical decisions made in the project, specifically the choice of FastAPI for the API framework and the use of pickle files for storing components like the imputer, standardizer, and model weights.

### Why FastAPI?

FastAPI was chosen as the web framework for a few compelling reasons:

1. **Performance**: FastAPI is one of the fastest web frameworks for Python, thanks to its Starlette foundation for handling asynchronous tasks and Pydantic for data validation.
2. **Ease of Use**: It offers an intuitive way to define APIs with Python type annotations, leading to less code, fewer bugs, and automatic request data validation.
3. **Automatic API Documentation**: FastAPI automatically generates interactive API documentation (using Swagger UI), which is incredibly helpful for testing and understanding the API.
4. **Scalability**: It's well-suited for building high-performance, scalable APIs, which is crucial for handling real-time data processing and prediction tasks.

5. Swagger : Automatic documentation and versetile endpoint testing from /docs endpoint.

### Storing Imputer, Standardizer, and Model Weights in Pickle Files

The decision to use pickle files for storing the imputer, standardizer, and model weights was made with the following considerations:

1. **Efficiency**: By serializing these components into pickle files, we avoid the computational overhead of recalculating them every time the API is restarted. This approach significantly speeds up the API's response time.
2. **Consistency**: Storing these components ensures that the exact preprocessing and model configuration used during training are applied during prediction, maintaining consistency in the model's performance.
3. **Flexibility and Robustness**: The code is designed to be versatile. If the pickle files are not present (e.g., during the initial setup), the code can still function by creating these components using default or provided training data. This design choice ensures robustness and ease of setup for new users or in different environments.

### Choice of Docker Base Image: `python:3.9-slim-buster`

In the Dockerfile, `python:3.9-slim-buster` was selected as the base image over the standard `python:3.9` image due to the following reasons:

1. **Size Efficiency**: The `slim-buster` variant is significantly smaller in size compared to the standard Python image. This reduced size leads to faster build and deployment times, making the overall development and deployment process more efficient.
2. **Optimized for Production**: Despite its smaller size, `slim-buster` still contains all the essential tools and libraries required to run Python applications. This makes it an ideal choice for production environments where resource efficiency is important.
3. **Stability and Security**: Based on Debian Buster, the `slim-buster` image offers a stable and secure environment. It includes necessary security updates while excluding unnecessary packages and files, reducing the potential attack surface of the application.

By choosing `python:3.9-slim-buster`, we ensure that the Docker image is not only efficient in terms of size but also robust and suitable for production use. This choice reflects our commitment to creating an optimized and secure deployment for the State Farm Machine Learning API.

---
In Short:
By making these strategic decisions, the project achieves a balance between performance, ease of use, and robustness, ensuring that the API is both efficient and user-friendly.


## Prerequisites

Before beginning, ensure you have Docker installed on your system. Docker will handle the creation of the environment and dependency management, offering a seamless setup experience.

## Getting Started

### Clone the Repository

Clone the repository to your local machine:

Docker Setup
The project uses Docker to create a consistent environment:

Build the Docker Image:

```docker build -t state-farm-model .
```

This command builds a Docker image named state-farm-model based on the instructions in the Dockerfile.

Run the Container:

```docker run -p 1313:1313 state-farm-model
```

This command runs the Docker container, mapping the container's port 1313 to the local port 1313.

```
state-farm/
│
├── app/                     # FastAPI application
│   ├── __init__.py
│   ├── main.py              # Main FastAPI app
│   ├── models.py            # Pydantic models for data
│   ├── dependencies.py      # Configurations and constants
│   └── router.py            # API endpoints
│
├── ml_model/                # Machine Learning model and preprocessing
│   ├── __init__.py
│   ├── model.py             # Model loading and prediction
│   └── preprocessing.py     # Data preprocessing
│
├── tests/                   # Unit tests for the API
│   ├── __init__.py
│   └── test_api.py          
│
├── Dockerfile               # Dockerfile for containerization
├── requirements.txt         # Python dependencies
└── run_api.sh               # Script to run API using Docker

```

# Continuous Integration and Deployment (CI/CD)
The project includes a CI/CD pipeline, ensuring automated testing and deployment:

Continuous Integration: Automated tests are run to ensure the codebase's integrity with each commit or pull request.
Continuous Deployment: Upon successful testing and review, changes are automatically deployed to the production environment.
The CI/CD pipeline is configured in the DockerToDockerHub.yml file, automating the process from code changes to deployment.

DockerHub Deployment
Find the Docker image for the project at:

https://hub.docker.com/r/oms96/state-farm-predict/tags

Repository
Access the full codebase and resources:

State Farm GitHub Repository : https://github.com/OMS1996/state-farm

## Detailed Setup Instructions

### Python Version Requirement

This project is designed to work with Python version 3.7 or higher. It is crucial to use a compatible Python version to avoid any compatibility issues with the libraries and frameworks used.

### Main Scripts Explanation

1. **`app/main.py`**:
   - This is the entry point for the FastAPI application.
   - It sets up and runs the FastAPI server, including routing and initialization.

2. **`app/router.py`**:
   - Defines the API endpoints and handles the incoming API requests.
   - Processes input data and calls the prediction functions.

3. **`app/models.py`**:
   - Contains Pydantic models that define the structure of request and response data for the API.
   - Ensures consistent and validated data handling.

4. **`app/dependencies.py`**:
   - Stores configurations, constants, and other dependencies that are used throughout the application.
   - Helps in managing and organizing settings that are shared across different parts of the application.

5. **`ml_model/model.py`**:
   - Responsible for loading the machine learning model and running predictions.
   - Handles the deserialization of the trained model and provides a function for making predictions.
   NOTE: I did not use it because of time constraints.

6. **`ml_model/preprocessing.py`**:
   - Contains functions for preprocessing input data before it is fed into the model.
   - Includes steps like imputation, scaling, and encoding necessary for preparing the data for prediction.

7. **`tests/test_api.py`**:
   - Includes unit tests for testing the API endpoints.
   - Ensures the reliability and correctness of the API functionality.

8. **`Dockerfile`**:
   - Contains instructions for building the Docker image of the application.
   - Specifies the base Python image, sets up the environment, installs dependencies, and defines the command to run the application.

NOTE:
Shell Script (run_api.sh): This script should handle starting the Docker container. Make sure it's executable (chmod +x run_api.sh) and well-commented.

9. **`run_api.sh`**:
   - A shell script for running the API using Docker.
   - Simplifies the process of starting the application in a Docker container.

   First Give Execution Permission to the Script`chmod +x run_api.sh`


Execute the Script: `Run your run_api.sh` script. This will build the Docker image and run the container.

What the Script Does
docker build -t state-farm-predict . : This command builds a Docker image from the Dockerfile in the current directory (.) and tags it with the name state-farm-predict.

docker container run -p 1313:1313 state-farm-predict: This command runs a container from the state-farm-predict image. The -p 1313:1313 flag maps port 1313 of the container to port 1313 on your Mac, allowing you to access the API via localhost:1313.

## Running the Application

Once you have cloned the repository and ensured the correct Python version is installed, you can run the application using Docker as described in the 'Docker Setup' section above.

After Running the Script
Check Container Status: You can check if the container is running by using:

```docker ps```

Accessing the API: Once the container is up, you should be able to make requests to your API at http://localhost:1313.

Debugging: If you encounter any issues, check the Docker container logs for any error messages, ALTHOUGH THERE IS ONLY ONE CONTAINER:
```docker logs [CONTAINER_ID]```

Common Issues on Mac
Firewall or Network Issues: Sometimes, the Mac firewall or network configurations might block ports. Ensure that port 1313 is open and accessible.

Resource Allocation: Docker on Mac sometimes requires adjusting resource allocations (like memory and CPU) for optimal performance, especially for more intensive applications.

Filesystem Permissions: If your Dockerfile or application accesses local files, ensure that Docker has the necessary permissions to access those directories on your Mac.

## Python Example for Using the Predict Endpoint

The following Python script demonstrates how to send a request to the predict endpoint. This example assumes you have a batch of input data that you wish to make predictions on.

### Prerequisites

Ensure you have Python installed, and the `requests` library is available. You can install the `requests` library using pip if it's not already installed:

```bash
pip install requests
```

```
import requests
import traceback

# Prepare your batch data
batch_data = [
    {"x0": "value1", "x1": "value2", ..., "x99": "value99"},
    # Add more data points as needed
]

# Define the list of feature names used in the model
VARIABLES = ["x0", "x1", ..., "x99"]

# Prepare the payload for the request
payload = {
    "input_data": batch_data,
    "selected_variables": VARIABLES
}

# Send the request to the predict endpoint
try:
    response = requests.post("http://0.0.0.0:1313/predict", json=payload)
    if response.status_code == 200:
        # Process the response if it's successful
        predictions = response.json()
        print("Predictions:", predictions)
    else:
        print(f"Failed to get predictions. Status code: {response.status_code}")
except Exception as e:
    # Print the error details
    traceback.print_exc()
    print(f"An error occurred: {e}")

# Check if a response was received
if response is None:
    print("No response received from the API.")

```
Running the Script
Save this script as a .py file.
Run it using Python in your terminal or command prompt.
The script will send the request to your FastAPI application and print the predictions received in response.


## Contact

For any inquiries or contributions, please contact Omar M. Hussein at omarmoh.said@gmail.com or @OMS1996.



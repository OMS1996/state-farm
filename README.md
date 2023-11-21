# State Farm Machine Learning API Deployment

## Overview

This repository contains the resources for deploying a machine learning model as an API for State Farm. It's designed to provide real-time predictions via a logistic regression model. The deployment is streamlined through Docker, emphasizing ease of use and consistency across different environments. Additionally, a CI/CD pipeline is integrated for automated testing and deployment.

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

9. **`run_api.sh`**:
   - A shell script for running the API using Docker.
   - Simplifies the process of starting the application in a Docker container.

## Running the Application

Once you have cloned the repository and ensured the correct Python version is installed, you can run the application using Docker as described in the 'Docker Setup' section above.


## Contact

For any inquiries or contributions, please contact Omar M. Hussein at omarmoh.said@gmail.com or @OMS1996.



# State Farm Model Deployment
Overview
This repository contains the model, EDA, code and instructions for deploying a machine learning model as an API for State Farm. The goal is to make the model accessible through API calls, allowing real-time predictions.

# Setup
Prerequisites
Before getting started, ensure you have the following prerequisites installed on your system:

# Python (version 3.9 or higher)
Git (optional, but recommended)
Clone the Repository
If you have Git installed, you can clone this repository to your local machine using the following command:

bash
Copy code
git clone https://github.com/yourusername/state-farm.git
Replace yourusername with your GitHub username or the repository URL.

Create a Virtual Environment (Optional but Recommended)
To isolate project dependencies, it's a good practice to create a virtual environment. Here's an example using venv:

bash
Copy code
# Navigate to the project directory
```cd state-farm```

# Create a virtual environment (Python 3.9+)
```python3 -m venv venv```

# Activate the virtual environment
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
Install Dependencies
With the virtual environment activated (if you created one), you can install the required dependencies using pip and the requirements.txt file:

bash
Copy code
pip install -r requirements.txt
This command will install all the necessary libraries, including scikit-learn and pandas, based on the versions specified in the requirements.txt file.

Project Structure
The project is structured as follows:

```
# state-farm/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # Main FastAPI app
│   ├── models.py            # Pydantic models for request and response data
│   ├── dependencies.py      # Dependencies, configurations, or constants
│   └── router.py            # Router for the API endpoints
│
├── ml_model/
│   ├── __init__.py
│   ├── model.py             # Code for loading and running the ML model
│   └── preprocessing.py     # Data preprocessing functions
│
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Tests for the API endpoints
│
├── Dockerfile               # Dockerfile for containerizing the application
├── requirements.txt         # Python dependencies
└── run_api.sh               # Shell script to run the API using Docker
```
Usage
To run the FastAPI application using Uvicorn, you can use the following command:

bash
Copy code
```uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload```
Make sure to run this command in the root directory of your FastAPI project, where the app module is located. Adjust the parameters (e.g., --host, --port) to match your specific setup if necessary.

Deployment Steps
Prepare the Model for Production: Update your code to meet production coding standards, including modularization, code quality, unit testing, and documentation.

Wrap the Model in an API: Make the model callable via API call on port 1313. The API should accept data in JSON format and return predictions.

Dockerize the API: Create a Dockerfile to build your API into an image. Write a shell script (run_api.sh) to run your Docker container.

Optimize for Scalability: Identify opportunities to optimize your deployment for scalability, considering a large number of API calls.

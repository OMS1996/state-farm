# Main FastAPI app
from fastapi import FastAPI
from app.router import router  # Adjust the import path based on your project structure

app = FastAPI()

# Include the router
app.include_router(router)

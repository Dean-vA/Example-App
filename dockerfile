# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app/src/number_predictor

# Add metadata to the image
LABEL maintainer="this should be your name and email address"
LABEL version="1.0"
LABEL description="Python CLI MNIST app"

# Copy the current directory contents into the container at /app
COPY . /app

# Install poetry
RUN pip install poetry

# Install only runtime dependencies using poetry
RUN poetry config virtualenvs.create false && poetry install --only main

# Install additional dependencies
# This is needed for the FastAPI app to accept file uploads
RUN pip install python-multipart

# Set the startup command to run your API
# ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

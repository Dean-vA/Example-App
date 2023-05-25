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

# Install dependencies using poetry
RUN poetry config virtualenvs.create false && poetry install

# Set the startup command to run your binary
ENTRYPOINT ["/bin/bash"]
# Display help by default
#CMD ["python", "predict.py", "--help"]

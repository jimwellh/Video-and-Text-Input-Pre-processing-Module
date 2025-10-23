# Dockerfile

# Start from a Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies (like ffmpeg)
# We update apt, install ffmpeg, and then clean up the apt cache
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Your Application Code
# Copy the 'app' directory from your project into the container's '/app' directory
COPY ./app /app/app

# 4. Set the command to run your application
# This tells Docker to run Uvicorn, pointing it to the 'app' object in 'app.main'
# It will be accessible on port 8000 inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
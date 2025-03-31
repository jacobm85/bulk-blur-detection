# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from the GitHub repository
# Install necessary dependencies for OpenCV and Git
RUN apt-get update && \
    apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/jacobm85/bulk-blur-detection.git .

# Copy all files into the container
# COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask flask-socketio eventlet

# Copy the index.html file
COPY templates/index.html /app/templates/index.html

# Copy the web app code
COPY app.py /app/app.py

# Copy the detector code
COPY blur_detector.py /app/blur_detector.py

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

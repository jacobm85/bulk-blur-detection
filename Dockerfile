# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from the GitHub repository
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/jacobm85/bulk-blur-detection.git .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the index.html file (you need to create this file in your repo)
COPY index.html /app/index.html

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]  # Assume you have an app.py to run your web interface

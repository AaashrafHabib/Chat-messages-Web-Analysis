# Use Python 3.10 image as base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Define the command to run the backend Flask app
CMD ["python", "app.py"]

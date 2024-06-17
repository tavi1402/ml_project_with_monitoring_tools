# Use the official Python image from the Docker Hub
FROM python:3.9-slim
#
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself (assuming setup.py is in the root directory)
RUN pip install -e .

# Set environment variables to ensure Python outputs are visible in the logs
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Expose the port the app runs on
EXPOSE 5001

# Run the Flask App
CMD ["python", "app.py"]

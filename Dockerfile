FROM python:3.9-slim-buster
WORKDIR /flaskApp
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv \
    && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir -r requirements.txt

# Copy the application files
ADD yolo_tiny_configs /flaskApp/yolo_tiny_configs
ADD object_detection.py /flaskApp/object_detection.py
ADD Cloudiod_client.py /flaskApp/Cloudiod_client.py

# Expose the port that the application will run on
EXPOSE 5000

# Set the command to run the application
CMD ["python3", "object_detection.py"]

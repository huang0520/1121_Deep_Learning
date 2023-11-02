FROM tensorflow/tensorflow:latest-gpu-jupyter

# Copy requirement
COPY requirements.txt /tmp/requirements.txt

# Install requirement
RUN pip install -r /tmp/requirements.txt

# Delete temp file
RUN rm -rf /tmp/

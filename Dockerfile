FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

# Copy dependencies
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -r /tmp/requirements.txt



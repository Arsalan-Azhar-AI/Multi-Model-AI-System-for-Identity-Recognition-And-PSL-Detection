# Base image with GPU support for TensorFlow and YOLOv8
FROM nvidia/cuda:12.2.0-base-ubuntu22.04


# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python and basic tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    tensorflow \
    pandas \
    dvc \
    notebook \
    numpy \
    matplotlib \
    seaborn \
    python-box==6.0.2 \
    pyYAML \
    tqdm \
    ensure==1.0.2 \
    joblib \
    types-PyYAML \
    scipy \
    Flask \
    Flask-Cors \
    kaggle \
    opencv-python \
    Pillow \
    mlflow==2.2.2 \
    ultralytics \
    mediapipe

# Copy rest of the code
COPY . .

# Default command
CMD ["python", "main.py"]

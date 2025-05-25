FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install remaining Python dependencies (YOLOv8 is already included)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . /app

# Run the app
CMD ["python3", "app.py"]

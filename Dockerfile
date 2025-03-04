# Use a standard Python image instead of the pytorch-specific one for better compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies first (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (these change less frequently than code)
COPY requirements.txt .

# Install dependencies in a single layer to optimize caching
# Install PyTorch first to avoid issues with other dependencies
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (changes frequently)
COPY . .

# Set default port for the application
EXPOSE 23123

# Command to run the application
CMD ["python", "app.py"]

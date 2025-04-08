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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements files first to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a single layer to optimize caching
# Combine pip install with prisma install to reduce layers
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install prisma

# Copy application code (changes frequently)
COPY . .

# Generate Prisma client - run after code is copied
RUN prisma generate

# Set default port for the application
EXPOSE 23123

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "23123"]

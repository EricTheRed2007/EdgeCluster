# Base image
FROM python:3.11-slim

# Prevent python buffering
ENV PYTHONUNBUFFERED=1

# Prevent HuggingFace from attempting downloads
ENV TRANSFORMERS_OFFLINE=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY node/requirements.txt ./requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY node/ ./node
COPY shared/ ./shared

# Set working directory to node folder
WORKDIR /app/node

# Default command
CMD ["python", "node.py"]
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Fetch the actual LFS files now that the repo is copied
RUN git lfs install && git lfs pull

# Expose port
EXPOSE 5000

# Run with gunicorn (reads PORT from environment, defaulting to 5000)
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 8 --timeout 0 app:app

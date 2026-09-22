FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install transformers rasterio

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the unified backend + frontend server
CMD ["python", "backend/main.py"]

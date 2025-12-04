FROM python:3.10-slim

WORKDIR /app

# Install system dependencies jika diperlukan
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu untuk caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

# Expose port
EXPOSE 5000

# Run dengan gunicorn (untuk production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_ml:app"]
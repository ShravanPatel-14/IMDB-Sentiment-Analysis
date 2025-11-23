# Dockerfile - lightweight image for Render (no TensorFlow)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (basic build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.render.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose Flask port
EXPOSE 5000

# Environment defaults
ENV PORT=5000
ENV HOST=0.0.0.0

# Use gunicorn to serve app_render:app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app_render:app"]
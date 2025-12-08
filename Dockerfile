# FROM python:3.12-slim

# # avoid interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV TRANSFORMERS_CACHE=/tmp/hf-cache

# WORKDIR /app

# # system deps (enough for transformers/torch)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*



# # install Python deps (from slim runtime requirements)
# COPY requirements.txt /app/requirements.txt
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # copy the rest of the app
# COPY . /app

# # Expose port used by Flask app
# EXPOSE 5000

# # Run the lightweight HF pipeline app, NOT the old TF app
# # app_render.py contains: app = Flask(__name__)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app_render:app"]




FROM python:3.12-slim

# avoid interactive prompts and unnecessary .pyc files
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/hf-cache \
    HF_HOME=/tmp/hf-home

WORKDIR /app

# Minimal system deps (enough for transformers + torch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from slim requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 # CPU-only PyTorch (no CUDA → much smaller, fits Render free tier)
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

# Copy the rest of the project
COPY . /app

# Port used by Flask/Gunicorn
EXPOSE 5000

# Run the lightweight HF pipeline app
# app_render.py contains: app = Flask(__name__)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app_render:app"]

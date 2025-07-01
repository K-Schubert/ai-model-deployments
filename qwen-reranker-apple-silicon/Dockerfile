# build for ARM64 so it runs on Apple Silicon
FROM --platform=linux/arm64 python:3.11-slim

# install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        ninja-build \
        libopenblas-dev \
        libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# install Python deps
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# copy app code
WORKDIR /app
COPY app/ /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10

# Update ubuntu's package installer and download python and other important packages.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y curl unzip sudo libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install shapely --no-binary shapely && \
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip

# Create app directory.
WORKDIR /app

# Copy all files to environment.
COPY . .

# Set environment variables for data access
ENV GRAPHCAST_DATA_BUCKET "dataset" 
ENV GRAPHCAST_PARAMS_BUCKET "params" 
ENV GRAPHCAST_STATS_BUCKET "stats"
ENV GRAPHCAST_BUCKET_NAME "elet-dm-graphcast"
ENV GRAPHCAST_MODEL_PATH "GraphCast_operational.npz" 

# Set environment variables for TPU
ENV JAX_PLATFORM_NAME tpu
ENV JAX_TPU_DRIVER_MODE async

# Set the entry point for your container to run predictions.py
ENTRYPOINT ["python3", "predictions.py"]
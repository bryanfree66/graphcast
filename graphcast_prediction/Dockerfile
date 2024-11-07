# Get ubuntu image from https://hub.docker.com/_/ubuntu.
FROM ubuntu:22.04

# Update ubuntu's package installer and download python and other important packages.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3 python3-pip curl unzip sudo

# Download components required by Graphcast.
RUN pip3 install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
# RUN pip3 install git+https://github.com/deepmind/dm-haiku

# Create app directory.
RUN mkdir -p /app
WORKDIR /app

# Copy all files to environment.
COPY . .

# Download necessary packages.
RUN sudo apt-get install -y libgeos-dev
RUN pip3 uninstall -y shapely
RUN pip3 install -r requirements.txt
RUN pip3 install shapely --no-binary shapely
RUN pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Set environment variables for data access
ENV GRAPHCAST_DATA_BUCKET gs://elet-dm-graphcast/dataset
ENV GRAPHCAST_PARAMS_BUCKET gs://elet-dm-graphcast/params
ENV GRAPHCAST_STATS_BUCKET gs://elet-dm-graphcast/stats
ENV GRAPHCAST_MODEL_PATH "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"

# Set environment variables for TPU
ENV JAX_PLATFORM_NAME tpu
ENV JAX_TPU_DRIVER_MODE async

# Set the entry point for your container to run predictions.py
ENTRYPOINT ["python3", "predictions.py"]
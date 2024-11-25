# GraphCast Weather Forecasting

This repository contains the code and artifacts for deploying and evaluating Google DeepMind's GraphCast model for medium-range weather forecasting, specifically tailored for Electrobras's operational needs in Brazil.

## Project Overview

This project aims to demonstrate the value of GraphCast in providing accurate and timely weather forecasts to support decision-making processes and enhance the resilience of Brazil's energy infrastructure.

**Key Objectives:**

* Deploy a pretrained GraphCast model on Google Cloud Platform (GCP).
* Acquire and process relevant weather data for Brazil.
* Generate medium-range weather forecasts (up to 10 days).
* Evaluate the model's performance and visualize the forecasts.
* Assess the potential impact of GraphCast on Electrobras's operations.

## Repository Structure

* **data**: Scripts and utilities for acquiring and preprocessing weather data from ECMWF and other sources.
* **model**: Code for deploying and interacting with the GraphCast model on GCP (TPU, Cloud Functions, etc.).
* **evaluation**: Scripts and notebooks for evaluating the model's performance using various metrics and visualizations.
* **reporting**: Code and templates for generating reports and presentations summarizing the project findings.
* **pipelines**: Workflow orchestration using Cloud Functions, Cloud Run, or Kubeflow Pipelines.

## Getting Started

This project leverages Google Cloud Platform (GCP) to deploy and run GraphCast. Follow these steps to get started:

**1. Set up a GCP Project**

* Create a GCP project and enable the following APIs:
    * Compute Engine
    * Cloud Storage
    * Cloud Functions
    * Cloud Build
    * Artifact Registry

**2.  Prepare Cloud Storage**

* Create a Cloud Storage bucket to store model checkpoints, normalization weights, and the ERA5 dataset: `gs://cs_elet_meteorologia_dev/graphcast-v1`
* Download pre-trained model parameters and normalization weights:
    ```bash
    gsutil -m rsync -r gs://dm_graphcast/params gs://cs_elet_meteorologia_dev/graphcast-v1/params
    gsutil -m rsync -r gs://dm_graphcast/stats gs://cs_elet_meteorologia_dev/graphcast-v1/stats
    ```
* Use the provided Python script in the `data` directory to download the ERA5 dataset to `gs://cs_elet_meteorologia_dev/graphcast-v1/dataset`.

**3. Set up BigQuery**

* Create a BigQuery dataset to store weather data using the provided `bigquery_schema.sql` file:
    ```bash
    bq query --location=US --use_legacy_sql=false < bigquery_schema.sql
    ```

**4.  Create a TPU VM**

* Create a TPU VM instance for running the GraphCast model:
    ```bash
    gcloud compute tpus tpu-vm create elet-graphcast-tpuv5-32-vm \
        --zone=us-east1-d \
        --accelerator-type=v5p-32 \
        --version=tpu-vm-tf-2.11.0 \
        --network=default \
        --subnetwork=default \
        --preemptible 
    ```

**5. Build and Push the Docker Image**

* Create a container repository in Artifact Registry:
    ```bash
    gcloud artifacts repositories create elet-graphcast-repo \
      --repository-format=docker \
      --location=us-central1 \
      --description="Docker repository for GraphCast" 
    ```
* Build and push the Docker image using the provided `cloudbuild.yaml` file:
    ```bash
    gcloud builds submit --config cloudbuild.yaml
    ```

**6. Run GraphCast**

* SSH to the TPU instance:
    ```bash
    gcloud compute tpus tpu-vm ssh elet-graphcast-tpuv5-32-vm \
        --zone=us-east1-d -- -o ProxyCommand='corp-ssh-helper %h %p'
    ```
* Set up application default login:
    ```bash
    gcloud auth application-default login
    ```
* Pull the Docker image from Artifact Registry:
    ```bash
    docker pull us-central1-docker.pkg.dev/bryan-dev-396914/elet-meteorologia-graphcast-dev/graphcast:latest
    ```
* Run the container:
    ```bash
    docker run -it us-central1-docker.pkg.dev/bryan-dev-396914/elet-meteorologia-graphcast-dev/graphcast:latest python3 predictions.py 2022-01-01 10
    ```

## Key Technologies

* **GraphCast**: Google DeepMind's machine learning model for weather prediction.
* **ECMWF**: European Centre for Medium-Range Weather Forecasts for weather data.
* **Google Cloud Platform (GCP)**: 
    * Compute Engine, TPU v5
    * Cloud Storage
    * Cloud Functions
    * Cloud Build
    * Artifact Registry
    * BigQuery

# GraphCast Weather Forecasting for Electrobras

This repository contains the code and artifacts for deploying and evaluating Google DeepMind's GraphCast model for medium-range weather forecasting, specifically tailored for Electrobras's operational needs in Brazil.

## Project Overview

This project aims to demonstrate the value of GraphCast in providing accurate and timely weather forecasts to support Electrobras's decision-making processes and enhance the resilience of Brazil's energy infrastructure.

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
* **pipelines**:  Workflow orchestration using Cloud Functions, Cloud Run, or Kubeflow Pipelines.

## Getting Started

1. **Set up a GCP Project**: Create a GCP project and enable the necessary APIs (Compute Engine, Cloud Storage, Cloud Functions, etc.).
2. **Clone this Repository**: `git clone https://github.com/your-username/graphcast-electrobras.git`
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Configure Environment**: Set up environment variables for API keys, data paths, etc.
5. **Run Pipelines**: Execute the data acquisition, preprocessing, forecast generation, and evaluation pipelines.

## Key Technologies

* **GraphCast**: Google DeepMind's machine learning model for weather prediction.
* **ECMWF**: European Centre for Medium-Range Weather Forecasts for weather data.
* **Google Cloud Platform (GCP)**: 
    * Compute Engine, TPU v4
    * Cloud Storage
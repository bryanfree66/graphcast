#!/bin/bash
# Install docker
sudo apt-get update && sudo apt-get install -y docker.io
# Start docker service
sudo systemctl enable docker.service
sudo systemctl start docker.service
# Pull your docker image
docker pull us-central1-docker.pkg.dev/$PROJECT_ID/elet-meteorologia-graphcast-dev/graphcast:latest

# Grant permissions to the user to execute docker commands without sudo
sudo usermod -aG docker bryanfreeman@google.com

# This line is necessary to ensure the changes take effect immediately
newgrp docker
import subprocess

def build_and_deploy():
  """Builds the application, connects to the TPU VM, and deploys the Docker image."""

  try:
    # Step 1: Build the application using gcloud
    subprocess.run(["gcloud", "builds", "submit", "--config", "cloudbuild.yaml", "."], check=True)
    print("Application build successful!")

    # Step 2: SSH into the TPU VM
    ssh_command = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", 
        "elet-graphcast-tpu-vm", "--", 
        "-o", "ProxyCommand='corp-ssh-helper %h %p'"
    ]
    subprocess.run(ssh_command, check=True)
    print("Successfully connected to TPU VM.")

    # Steps 3 & 4:  Pull and run the Docker image (combined)
    docker_commands = [
        "docker pull us-central1-docker.pkg.dev/bryan-dev-396914/elet-meteorologia-graphcast-dev/graphcast:latest",
        "us-central1-docker.pkg.dev/bryan-dev-396914/elet-graphcast-repo/graphcast-image eletrobras/graphcast"
    ]
    for command in docker_commands:
      subprocess.run(command, shell=True, check=True) 

    print("Docker image pulled and running!")

  except subprocess.CalledProcessError as e:
    print(f"Error during execution: {e}")

if __name__ == "__main__":
  build_and_deploy()
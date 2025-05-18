#!/usr/bin/env python3
import subprocess

def build_docker_image():
    """Build the Docker image with the name 'metadrive'."""
    docker_image_name = "metadrive"
    subprocess.run(
        ["docker", "build", "-t", docker_image_name, "."],
        check=True,
        text=True
    )
    print(f"Docker image '{docker_image_name}' built successfully")

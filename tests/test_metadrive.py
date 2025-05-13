import subprocess

def test_metadrive_docker_build_and_run():
    """Smoke test: Dockerfile builds and runs MetaDrive profile example."""

    # Step 1: Build Docker image from project Dockerfile
    
    build = subprocess.run(
        ["docker", "build", "-t", "metadrive", "."],
        capture_output=True,
        text=True
    )
    assert build.returncode == 0, f"Docker build failed:\n{build.stderr}"

    # Step 2: Verify Docker image exists and is properly configured
    inspect = subprocess.run(
        ["docker", "inspect", "metadrive"],
        capture_output=True,
        text=True
    )
    assert inspect.returncode == 0, f"Docker inspect failed:\n{inspect.stderr}"

    # Step 3: Run MetaDrive profile example inside container
    run = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", "metadrive",
         "-m", "metadrive.examples.profile_metadrive"],
        capture_output=True,
        text=True
    )
    assert run.returncode == 0, f"Docker run failed:\n{run.stderr}"

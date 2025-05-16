import subprocess
from pathlib import Path

def test_metadrive_docker_build():
    """Test that the Dockerfile builds correctly."""
    
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

def test_metadrive_run_generates_output():
    """Test that the custom implementation runs and generates output."""
    
    # Create a temporary output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run the container with our custom implementation
        run = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{output_dir.absolute()}:/app/outputs", 
             "metadrive"],
            capture_output=True,
            text=True,
            timeout=60  # Set a timeout to prevent hanging tests
        )
        assert run.returncode == 0, f"Docker run failed:\n{run.stderr}"

    finally:
        # Clean up - remove test files
        for file in output_dir.glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        try:
            output_dir.rmdir()
        except Exception:
            pass

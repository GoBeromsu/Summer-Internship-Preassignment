FROM python:3.8-slim

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt /app/
COPY src /app/src/
COPY pyproject.toml /app/

# Install project requirements
RUN pip install --upgrade pip && pip install -e .

# Clone MetaDrive repository directly from GitHub (recommended method)
RUN git clone https://github.com/metadriverse/metadrive.git --single-branch

# Install MetaDrive from source
RUN cd metadrive && pip install -e .

# Define the entrypoint to run MetaDrive simulation
ENTRYPOINT ["python", "src/metadrive/run.py"]
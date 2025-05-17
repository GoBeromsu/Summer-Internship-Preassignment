FROM python:3.8-slim

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy metadrive from local
# Recommended way :  https://metadrive-simulator.readthedocs.io/en/latest/install.html
# COPY metadrive /app/metadrive
# RUN cd /app/metadrive && pip install -e . 
# RUN python -m metadrive.pull_asset

# Clone and install metadrive
RUN git clone https://github.com/metadriverse/metadrive.git --single-branch && \
    pip install -e ./metadrive && \
    python -m metadrive.pull_asset

COPY src /app/src/

RUN mkdir -p /app/outputs && chmod 777 /app/outputs

ENTRYPOINT ["python", "src/main.py"]
CMD ["--map", "10", "--output-type", "png", "--output-dir", "/app/outputs"]

VOLUME ["/app/outputs"]
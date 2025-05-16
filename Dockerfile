FROM python:3.8-slim

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY metadrive /app/metadrive/
RUN pip install -e ./metadrive

COPY src /app/src/

RUN mkdir -p /app/outputs && chmod 777 /app/outputs

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--map", "10", "--num-scenarios", "3", "--output-type", "png", "--output-dir", "/app/outputs"]

VOLUME ["/app/outputs"]
# Gemma-4 Local Servers on Jetson Orin

This repository contains Docker Compose configurations to deploy and run Gemma-4 models locally on an NVIDIA Jetson Orin AGX.

## Prerequisites

- NVIDIA Jetson Orin AGX (configured with JetPack and NVIDIA Container Toolkit)
- Docker & Docker Compose
- SSD mounted at `/mnt/ssd/` with a `huggingface` directory for model caching

## Configuration

The services run using the `nvidia` runtime and share the host network.

- **Gemma-4 31B (Dense QAT GGUF)**: Port `8080`
- **Gemma-4 26B-A4B (MoE QAT GGUF)**: Port `8081`

## Usage

Start the Gemma-4 31B server:
```bash
docker compose up -d gemma4-31b
```

Start the Gemma-4 26B-A4B server:
```bash
docker compose up -d gemma4-26b-a4b
```

Stop the services:
```bash
docker compose down
```

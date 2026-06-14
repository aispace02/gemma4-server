# Gemma-4 Local Servers on Jetson Orin

This repository contains Docker Compose configurations to deploy and run Gemma-4 models locally on an NVIDIA Jetson Orin AGX.

## Prerequisites

- NVIDIA Jetson Orin AGX (configured with JetPack and NVIDIA Container Toolkit)
- Docker & Docker Compose
- SSD mounted at `/mnt/ssd/` with a `huggingface` directory for model caching

---

## Model Downloads (Mainland China / High Stability)

Since the built-in downloader in `llama-server` is slow and unstable under container runtime, it is highly recommended to pre-download the GGUF models to your local SSD directory (`/mnt/ssd/huggingface`) using `huggingface-cli` before starting the containers.

### 1. Install Hugging Face Hub CLI
On the host terminal, install the tools and set the mirror endpoint:
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. Download the Models
Choose and download the model GGUF file(s) you need:

- **Gemma-4 31B (Dense QAT GGUF)**:
  ```bash
  huggingface-cli download unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local-dir /mnt/ssd/huggingface
  ```

- **Gemma-4 26B-A4B (MoE QAT GGUF)**:
  ```bash
  huggingface-cli download unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --local-dir /mnt/ssd/huggingface
  ```

---

## Configuration & Usage

The services run using the `nvidia` runtime and share the host network.

- **Gemma-4 31B**: Port `8080`
- **Gemma-4 26B-A4B**: Port `8081`

### Running the Services

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

### Viewing Real-time Logs
To monitor model loading progress and API requests, view logs in real-time:
```bash
# View all logs
docker compose logs -f

# View logs for a specific model service
docker compose logs -f gemma4-31b
docker compose logs -f gemma4-26b-a4b
```

---

## Multiline Input in Client Terminal (agy / MobaXterm)

If you are accessing this server via SSH client (like MobaXterm on Windows) and using `agy` or another terminal-based client, the standard `Enter` key will send your message immediately.

To insert a newline (multiline text) in the terminal:
1. **Shortcut Combinations**: Try `Shift + Enter` or `Alt + Enter`.
2. **Raw Carriage Return (Universal)**: Press `Ctrl + V` followed by `Enter` to input a literal newline character (`\n`) into the prompt.
3. **Copy-Paste Method**: Draft your multiline text in a local text editor (like Notepad/VS Code), copy it, and paste it into MobaXterm using the right-click menu.

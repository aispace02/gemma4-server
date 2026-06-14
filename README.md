# Gemma-4 Local Servers on Jetson Orin

This repository contains Docker Compose configurations to deploy and run Gemma-4 models locally on an NVIDIA Jetson Orin AGX.

## Prerequisites

- NVIDIA Jetson Orin AGX (configured with JetPack and NVIDIA Container Toolkit)
- Docker & Docker Compose
- SSD mounted at `/mnt/ssd/` with a `huggingface` directory for model caching

---

## 从零开始部署所需下载的资源 (Deploying From Scratch)

如果在一台干净的设备上从零开始部署，您需要下载并配置以下三个核心部分：

1. **容器及推理引擎**：拉取预编译好的 `llama.cpp` Docker 镜像（包含 CUDA 编译的 `llama-server` 引擎）。
2. **大模型权重文件**：下载 Gemma-4 的 `.gguf` 格式权重文件（存放于 SSD）。
3. **系统依赖**：设备需要安装 Docker、Docker Compose 和 NVIDIA Container Toolkit（用于在容器中调用 GPU）。

---

## 极速模型下载指南 (Model Downloads via ModelScope)

在大陆网络环境下，直接从 Hugging Face 下载模型可能会非常缓慢或中断。**推荐使用阿里 ModelScope (魔搭社区) 下载**，它可以提供跑满带宽的极速下载体验，且无需配置代理。

### 1. 安装 ModelScope CLI 命令行工具
在宿主机终端中执行：
```bash
pip install modelscope
```

### 2. 从 ModelScope 下载 GGUF 权重到 SSD
利用 `modelscope` 命令行工具，只下载需要的单个 GGUF 文件并保存到缓存路径：

- **Gemma-4 31B (Dense QAT GGUF)**：
  ```bash
  modelscope download --model unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local_dir /mnt/ssd/huggingface
  ```

- **Gemma-4 26B-A4B (MoE QAT GGUF)**：
  ```bash
  modelscope download --model unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --local_dir /mnt/ssd/huggingface
  ```

*(备份方案：如果通过 ModelScope 遇到问题，也可以使用 Hugging Face 的新命令行工具下载：`pip install -U huggingface_hub && export HF_ENDPOINT=https://hf-mirror.com && hf download unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local-dir /mnt/ssd/huggingface`)*

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

# Gemma-4 Local Servers on Jetson Orin

This repository contains Docker Compose configurations to deploy and run Gemma-4 models locally on an NVIDIA Jetson Orin AGX.

## Prerequisites

- NVIDIA Jetson Orin AGX (configured with JetPack and NVIDIA Container Toolkit)
- Docker & Docker Compose
- SSD mounted at `/mnt/ssd/` with a `huggingface` directory for model caching

---

## 从零开始部署所需下载的资源 (Deploying From Scratch)

如果在一台干净的设备上从零开始部署，您需要准备和下载以下三个部分。针对国内网络环境，我们推荐使用**南京大学 (NJU) 镜像源**加速容器镜像拉取，以及**ModelScope**加速模型权重下载。

### 1. 推理引擎及容器镜像（使用南京大学镜像源加速）
由于国内直接访问 `ghcr.io` (GitHub Container Registry) 极慢或无法连接，推荐使用南京大学的 GHCR 镜像站拉取并使用“双标签”方式配置：

```bash
# A. 使用南大镜像源极速拉取 llama_cpp 镜像
docker pull ghcr.nju.edu.cn/nvidia-ai-iot/llama_cpp:latest-jetson-orin

# B. 为镜像打上官方标签（双标签指向同一 Image ID，不占用额外磁盘空间，确保 Compose 文件可无缝引用）
docker tag ghcr.nju.edu.cn/nvidia-ai-iot/llama_cpp:latest-jetson-orin ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin
```

### 2. 大模型权重文件（使用 ModelScope 加速）
使用 ModelScope 魔搭社区的国内带宽极速下载 `.gguf` 权重文件，详见下方的**模型下载指南**。

### 3. 系统依赖（宿主机一次性配置）
* 安装 Docker 和 Docker Compose。
* 安装 NVIDIA Container Toolkit 并配置 `/etc/docker/daemon.json`（已在您之前的系统文档中记录）。

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

# Gemma-4 Local Servers on Jetson Orin

This repository contains Docker Compose configurations to deploy and run Gemma-4 models locally on an NVIDIA Jetson Orin AGX.

## Prerequisites

- NVIDIA Jetson Orin AGX (configured with JetPack and NVIDIA Container Toolkit)
- Docker & Docker Compose
- SSD mounted at `/mnt/ssd/` with a `huggingface` directory for model caching

---

## 什么是“量化感知训练”版本 (Quantization-Aware Training - QAT) ？

我们默认推荐并配置了 Gemma-4 31B 和 26B-A4B 的 **QAT (Quantization-Aware Training) 量化感知训练**版本（即文件名中的 `-qat`），其优势如下：

- **区别于传统量化 (PTQ)**：传统的 Post-Training Quantization (如常见的 `Q4_K_M`) 是直接对训练好的高精度权重进行截断压缩，这会导致模型（特别是数学、代码等强逻辑能力）出现明显的智力下降。
- **高精度、零损失**：QAT 是由 Google DeepMind 和 Unsloth 联合开发的技术。在模型微调/训练阶段就提前引入了量化误差的模拟，使模型在训练中自我适应 4-bit 环境。最终转换为 `UD-Q4_K_XL` 格式后，其**推理准确度几乎与原始未量化的 bfloat16 权重完全一致**。
- **显存极大节省**：31B QAT 版本在保持原版精度的前提下，显存开销仅需约 **18GB - 20GB**；26B-A4B 版本（MoE 混合专家架构，每次仅激活 4B 参数）更是只需要 **15GB - 18GB** 显存，是 Jetson Orin 等嵌入式边缘设备运行大模型的最优解。

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

使用 ModelScope 魔搭社区的国内静态镜像带宽极速下载 `.gguf` 权重文件，详见下方的**模型下载指南**。

### 3. 系统依赖（宿主机一次性配置）

- 安装 Docker 和 Docker Compose。
- 安装 NVIDIA Container Toolkit 并配置 `/etc/docker/daemon.json`（已在您之前的系统文档中记录）。

---

## 测试

在局域网内（另外一台电脑上），使用curl测试命令如下：

```bash
# linux or macOS terminal for Gemma 4 31B (端口 8080)
curl -sN http://192.168.137.251:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "stream": true
  }'

# windows PowerShell for Gemma 4 31B (端口 8080)
curl -sN http://192.168.137.251:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "stream": true
  }'
```

---

## 极速模型下载指南 (Model Downloads via ModelScope)

在大陆网络环境下，直接从 Hugging Face 下载模型可能会非常缓慢或中断。**推荐使用阿里 ModelScope (魔搭社区) 下载**，它可以提供跑满带宽的极速下载体验，且无需配置代理。

### 1. 使用 uv 虚拟环境安装 ModelScope (推荐，保持系统干净)

如果宿主机上没有全局安装 `modelscope`，推荐使用现代 Python 包管理器 **`uv`** 创建虚拟环境并安装，避免污染系统的 Python 环境。

在大区终端输入以下命令：

```bash
# 1. 在当前目录下创建虚拟环境 (.venv)
uv venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 使用 uv 安装 modelscope (速度极快)
uv pip install modelscope
```

> **极速免激活方案**：您也可以完全不激活虚拟环境，直接使用 `uv run` 临时带入运行，效果是一样的：
> `uv run --with modelscope modelscope download ...`

### 2. 从 ModelScope 下载 GGUF 权重到 SSD

激活虚拟环境后（或使用上面的 `uv run` 方案），利用 `modelscope` 命令行工具，只下载需要的单个 GGUF 文件并保存到缓存路径：

- **Gemma-4 31B (Dense QAT GGUF)**：

  ```bash
  modelscope download --model unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local_dir /mnt/ssd/huggingface
  ```

- **Gemma-4 26B-A4B (MoE QAT GGUF)**：

  ```bash
  modelscope download --model unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --local_dir /mnt/ssd/huggingface
  ```

- **Gemma-4 12B-Agentic (蒸馏精调编程版本 - yuxinlu1 - Q6_K无损量化)**：

  ```bash
  modelscope download --model hf/yuxinlu1-gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF gemma4-v2-Q6_K.gguf --local_dir /mnt/ssd/huggingface
  ```

下载完成后，如果您不再需要该虚拟环境，可以直接输入 `deactivate` 退出虚拟环境，并删除生成的 `.venv` 文件夹（权重已安全地存在了 `/mnt/ssd/huggingface` 下）。

备份方案：如果通过 ModelScope 遇到问题，也可以使用 Hugging Face 的新命令行工具下载：

```bash
pip install -U huggingface_hub && export HF_ENDPOINT=https://hf-mirror.com && hf download unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local-dir /mnt/ssd/huggingface
```

- [ModelScope -- Gemma4-12B v2 — 编程 + 智能体版](https://www.modelscope.cn/models/hf/yuxinlu1-gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF) **12B Agentic 版本**：这是由用户 `yuxinlu1` 基于 Gemma-4 12B 进行蒸馏精调的版本，专门针对 AI Agent 和代码分析场景进行了优化微调。虽然是 Q6_K 无损量化，但得益于蒸馏和针对性微调，在实际使用中表现出色，且推理速度更快，非常适合 Jetson Orin 等边缘设备部署。
  - [Yuxin Lu](https://huggingface.co/yuxinlu1)：Yuxin Lu 在 huggingface 上的主页

---

## Configuration & Usage

The services run using the `nvidia` runtime and share the host network.

- **Gemma-4 31B**: Port `8080`
- **Gemma-4 26B-A4B**: Port `8081`
- **Gemma-4 12B-Agentic**: Port `8082` (针对 AI Agent、代码分析深度微调的 12B 无损量化版)

### Running the Services

```bash
# Start the Gemma-4 31B server
docker compose up -d gemma4-31b

# Start the Gemma-4 26B-A4B server
docker compose up -d gemma4-26b-a4b

# Start the Gemma-4 12B-Agentic server
docker compose up -d gemma4-12b-agentic

#Stop the services
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
docker compose logs -f gemma4-12b-agentic
```

更新推理引擎`llama.cpp`：

```bash
# 1. 拉取带有 "latest-jetson-orin" 标签的最新镜像
docker compose pull

# 2. 重新启动服务（Docker 会自动检测镜像变更，并销毁旧容器创建新容器）
docker compose up -d
```

---

## 接入 Pi Agent 配置 (Connecting to Pi Agent)

如果需要将本地部署的 Gemma-4 接入开源终端编程助手 **Pi Agent (`pi-coding-agent`)**，请按照以下步骤进行配置：

### 1. 配置自定义模型接口 (`~/.pi/agent/models.json`)

创建或修改 `~/.pi/agent/models.json` 文件，将本地的 31B（端口 8080）和 26B-A4B（端口 8081）分别注册为自定义 Provider：

```json
{
  "providers": {
    "local-gemma-31b": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "name": "Local Gemma-4 31B",
          "id": "gemma-4-31B",
          "contextWindow": 65536,
          "maxOutputTokens": 16384,
          "input": ["text"]
        }
      ]
    },
    "local-gemma-26b": {
      "baseUrl": "http://localhost:8081/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "gemma-4-26B",
          "name": "Local Gemma-4 26B",
          "contextWindow": 8192,
          "maxOutputTokens": 2048,
          "input": ["text"]
        }
      ]
    },
    "local-gemma-12b-agentic": {
      "baseUrl": "http://localhost:8082/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "gemma-4-12b-agentic",
          "name": "Local Gemma-4 12B Agentic",
          "contextWindow": 65536,
          "maxOutputTokens": 16384,
          "input": ["text"]
        }
      ]
    }
  }
}
```

### 2. 配置默认模型 (`~/.pi/agent/settings.json`)

若希望默认调用本地 12B-Agentic 模型，请在 `~/.pi/agent/settings.json` 中配置：

```json
{
  "defaultProvider": "local-gemma-12b-agentic",
  "defaultModel": "gemma4-v2-Q6_K",
  "defaultThinkingLevel": "off"
}
```

_注：建议将 `defaultThinkingLevel` 设为 `"off"` 以确保本地运行流畅。_

### 3. 会话中动态切换模型

在 Pi Agent 的交互命令行中，可直接输入 `/model` 指令进行切换：

- **切换到 12B Agentic 模型（推荐，速度最快、专门面向多轮分析微调）**：

  ```text
  /model local-gemma-12b-agentic/gemma4-v2-Q6_K
  ```

- **切换到 31B 模型**：

  ```text
  /model local-gemma-31b/unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL
  ```

- **切换到 26B A4B 模型**：

  ```text
  /model local-gemma-26b/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL
  ```

- 也可以按下快捷键 **`Ctrl + L`** 在弹出的模型菜单中选择。

---

## Multiline Input in Client Terminal (agy / MobaXterm)

If you are accessing this server via SSH client (like MobaXterm on Windows) and using `agy` or another terminal-based client, the standard `Enter` key will send your message immediately.

To insert a newline (multiline text) in the terminal:

1. **Shortcut Combinations**: Try `Shift + Enter` or `Alt + Enter`.
2. **Raw Carriage Return (Universal)**: Press `Ctrl + V` followed by `Enter` to input a literal newline character (`\n`) into the prompt.
3. **Copy-Paste Method**: Draft your multiline text in a local text editor (like Notepad/VS Code), copy it, and paste it into MobaXterm using the right-click menu.

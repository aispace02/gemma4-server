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

- **Qwopus3.6-35B-A3B-v1-MTP (基于 Claude Opus 蒸馏的 MoE 推理模型)**：

  ```bash
  modelscope download --model Jackrong/Qwopus3.6-35B-A3B-v1-MTP-GGUF Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf --local_dir /mnt/ssd/huggingface
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
- **Qwopus-3.6 35B MoE**: Port `8084` (基于 Claude Opus 蒸馏的 35B MoE 推理模型，激活仅 3B，速度极快)

### Running the Services

```bash
# Start the Gemma-4 31B server
docker compose up -d gemma4-31b

# Start the Gemma-4 26B-A4B server
docker compose up -d gemma4-26b-a4b

# Start the Gemma-4 12B-Agentic server
docker compose up -d gemma4-12b-agentic

# Start the Qwopus-3.6 35B MoE server
docker compose up -d qwen36-35b-moe

# Stop the services
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
docker compose logs -f qwen36-35b-moe
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
    },
    "local-qwopus36-35b-moe": {
      "baseUrl": "http://localhost:8084/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "qwopus3.6-35b-moe",
          "name": "Local Qwopus-3.6 35B MoE",
          "contextWindow": 131072,
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

- **切换到 Qwopus3.6 35B MoE 模型 (推荐，速度极快，智能体推荐)**：

  ```text
  /model local-qwopus36-35b-moe/qwopus3.6-35b-moe
  ```

- 也可以按下快捷键 **`Ctrl + L`** 在弹出的模型菜单中选择。

---

## Jetson 性能优化与故障排查 (Jetson Performance & Troubleshooting)

### 1. 扩大 Swap 交换空间 (防止长上下文 OOM 闪退)
在运行长上下文或多个大模型容器时，若超出 64GB 物理内存可能导致容器因 OOM（Out of Memory）被系统直接强制终止。建议在 SSD 上扩充 **16GB** 的 Swap 空间：

```bash
# A. 关闭当前系统 swap
sudo swapoff -a

# B. 在 SSD 挂载路径下创建 16GB 大小的交换文件 (以 /mnt/ssd/ 目录为例)
sudo dd if=/dev/zero of=/mnt/ssd/swapfile bs=1G count=16

# C. 设置安全权限并初始化 swap 分区
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile

# D. 启动新 swap
sudo swapon /mnt/ssd/swapfile
```

若需每次开机自动加载，请在宿主机编辑 `/etc/fstab` 文件并在末尾追加一行：
```text
/mnt/ssd/swapfile swap swap defaults 0 0
```

### 2. 内存锁定失败警告 (failed to mlock ... Cannot allocate memory)
* **警告原因**：当运行带 `--mlock` 参数的容器服务时，Docker 默认限制了容器能锁定的最大物理内存，从而导致 `failed to mlock`。
* **解决方案**：已在 [docker-compose.yml](file:///home/hxf0223/tmp/gemma-server/docker-compose.yml) 基础模板 `ulimits.memlock` 中将限制设为 `-1`（不限额）。请重新拉起服务使配置生效（警告信息会在重新启动后消失）。

---

## Docker & 本地 LLM 常用运维命令速查 (Docker & Local LLM Cheat Sheet)

在本地测试与部署大模型时，您在宿主机终端可能会高频使用以下命令：

### 1. 启动和停止服务
* **后台启动所有模型服务**（首次运行时会下载镜像）：
  ```bash
  docker compose up -d
  ```
* **启动特定的模型服务**（例如仅运行 Qwen3.6 MoE）：
  ```bash
  docker compose up -d qwen36-35b-moe
  ```
* **重启某个已运行的模型**（修改配置文件或遇到异常时常用）：
  ```bash
  docker compose restart qwen36-35b-moe
  ```
* **停止并完全销毁所有容器**（不会删除已下载的模型权重）：
  ```bash
  docker compose down
  ```
* **仅停止（暂停）容器而不删除它**：
  ```bash
  docker compose stop
  ```

### 2. 查看状态与运行日志
* **查看当前正在运行的容器状态**（包括其暴露的端口）：
  ```bash
  docker compose ps
  ```
* **实时追踪所有服务的组合日志**（便于观察 API 调用）：
  ```bash
  docker compose logs -f
  ```
* **实时查看单个服务日志**（最常用的 debug 手段，可确认模型加载进度）：
  ```bash
  docker compose logs -f qwen36-35b-moe
  ```
* **查看最新的 100 行日志并附带时间戳**：
  ```bash
  docker compose logs --tail=100 -f -t qwen36-35b-moe
  ```

### 3. 硬件与资源监控
* **使用 jtop 监控 Jetson 整机状态**（CPU、GPU 占用、温度、功耗）：
  ```bash
  jtop
  ```
* **查看 Docker 容器的内存和 CPU 实时占用率**（快速判断哪个容器占满了内存）：
  ```bash
  docker stats
  ```

### 4. 引擎升级与高级调试
* **升级 llama.cpp 推理引擎容器到官方最新版**：
  ```bash
  docker compose pull && docker compose up -d
  ```
* **进入正在运行的容器 shell 内部进行调试**（以 root 身份）：
  ```bash
  docker exec -it qwen36-35b-moe bash
  ```

---

## Multiline Input in Client Terminal (agy / MobaXterm)

If you are accessing this server via SSH client (like MobaXterm on Windows) and using `agy` or another terminal-based client, the standard `Enter` key will send your message immediately.

To insert a newline (multiline text) in the terminal:

1. **Shortcut Combinations**: Try `Shift + Enter` or `Alt + Enter`.
2. **Raw Carriage Return (Universal)**: Press `Ctrl + V` followed by `Enter` to input a literal newline character (`\n`) into the prompt.
3. **Copy-Paste Method**: Draft your multiline text in a local text editor (like Notepad/VS Code), copy it, and paste it into MobaXterm using the right-click menu.

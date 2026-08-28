# Gemma-4 Local Servers on Jetson Orin

本项目使用 Docker Compose 在 NVIDIA Jetson Orin AGX 上运行多个本地 GGUF 推理服务。当前主要面向 JetPack 7.2（CUDA 13.2.1 / Ubuntu 24.04），通过宿主机编译 `llama.cpp`，再由容器挂载运行。

## 0. 架构总览

### 0.1 构建与运行流程

```text
构建阶段
Dockerfile + docker compose build
        ↓
Image（CUDA 运行层 + libcurl4/libgomp1）

运行阶段
Image + docker-compose.yml + Host bind mounts + NVIDIA Runtime
        ↓
Container
        ↓
llama-server 加载 Host 挂载的 GGUF 模型
```

`docker compose build` 只构建镜像，不启动模型容器，也不会编译 `llama.cpp` 或下载模型。`docker compose up` 才会基于镜像创建容器，并按 Compose 配置挂载宿主机文件、GPU 和启动命令。

### 0.2 Host、Image 与 Container 的关系

```text
Host（Jetson Orin）
├── Jetson Linux、NVIDIA Driver、GPU
├── NVIDIA Container Toolkit / Runtime
├── /usr/local/bin 和 /usr/local/lib
│   └── 宿主机编译安装的 llama-server 和共享库
├── /mnt/ssd/huggingface
│   └── GGUF 模型和 MTP 文件
└── repository
    ├── Dockerfile：定义镜像
    └── docker-compose.yml：定义容器、挂载、GPU 和启动命令

Image（docker compose build）
├── NVIDIA CUDA runtime / Ubuntu 24.04 基础层
├── libcurl4
└── libgomp1

Container（docker compose up）
├── 来自 Image：CUDA 运行库和基础依赖
├── bind mount：/usr/local/bin → /opt/llama/bin
├── bind mount：/usr/local/lib → /opt/llama/lib
├── bind mount：/mnt/ssd/huggingface → /root/.cache/huggingface
└── NVIDIA Runtime 注入的 GPU 设备和驱动接口
```

| 内容 | 是否进入 Image | Host 来源 | Container 内路径 | 更新方式 |
| --- | --- | --- | --- | --- |
| CUDA 运行库和系统依赖 | 是 | NVIDIA NGC 基础镜像、Dockerfile | 镜像文件系统 | 修改 Dockerfile 后 `build` |
| `llama-server` 二进制和共享库 | 否 | `/usr/local/bin`、`/usr/local/lib` | `/opt/llama/bin`、`/opt/llama/lib` | 宿主机编译安装后启动或重启服务 |
| GGUF 模型和 MTP 文件 | 否 | `/mnt/ssd/huggingface` | `/root/.cache/huggingface` | 使用模型更新脚本后启动或重启服务 |
| GPU 和 NVIDIA Driver | 否 | Jetson 宿主机 | NVIDIA Runtime 注入 | 更新 JetPack/驱动和宿主机配置 |
| Compose 启动参数 | 否 | `docker-compose.yml` | 不复制到容器文件系统 | `docker compose up -d <service>` 重建容器 |

## 1. 快速开始

### 1.1 系统要求

- NVIDIA Jetson Orin AGX，已安装 JetPack 7.2 和 NVIDIA Container Toolkit
- Docker 与 Docker Compose v2
- SSD 挂载到 `/mnt/ssd/`，且存在 `/mnt/ssd/huggingface/` 模型缓存目录
- 宿主机已编译并安装 `llama.cpp` 到 `/usr/local`；若尚未准备，见第 4 节

### 1.2 构建容器镜像

如果本机还没有构建过运行时镜像，先执行下面的命令。镜像内容的详细组成见第 0 节。

```bash
# 构建全部服务的运行时镜像；只创建镜像，不会启动模型容器
docker compose build

# 也可以只构建一个服务
docker compose build gemma4-31b
```

### 1.3 下载模型并启动服务

优先使用仓库脚本从 ModelScope 下载模型，避免手工维护下载命令：

```bash
# 查看可用的 Compose 服务名
python3 scripts/update-models.py --list

# 下载一个模型；默认写入 /mnt/ssd/huggingface
python3 scripts/update-models.py gemma4-31b

# 启动单个模型服务
docker compose up -d gemma4-31b
```

如果系统没有安装 `modelscope`，脚本会自动通过 `uv run --no-project --with modelscope` 临时运行，不会修改项目虚拟环境。启动前脚本会清除代理变量，模型下载始终直连中国大陆 ModelScope。

### 1.4 验证服务

在局域网内另一台设备上执行：

```bash
curl -sN http://<JETSON_IP>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "stream": true
  }'
```

`curl` 在 Linux、macOS 和 Windows PowerShell 中均可用。若在 Jetson 本机测试，可将 `<JETSON_IP>` 改为 `localhost`。

## 2. 服务与端口

所有服务使用 `network_mode: host`，没有 Docker 端口映射。

| Compose 服务 | 模型 | 后端端口 | 上下文 | 适用场景 |
| --- | --- | ---: | ---: | --- |
| `gemma4-31b` | Gemma-4 31B QAT | 8080 | 64K | 高质量通用对话 |
| `gemma4-26b-a4b` | Gemma-4 26B-A4B | 8081 | 64K | 平衡速度与质量 |
| `gemma4-12b-agentic` | Gemma-4 12B Agentic | 8082 | 64K | 多轮 Agent 和代码分析 |
| `qwen36-35b-moe` | Qwen3.6 35B-A3B MoE | 8084 | 128K | 长上下文、Agent、高吞吐 |
| `qwen38-27b` | Qwen3.8 27B Dense | 8085 | 256K | 单点推理质量和编程任务 |

### 2.1 服务管理命令

```bash
# 查看状态
docker compose ps

# 启动单个服务；不要无参数启动全部大模型
docker compose up -d qwen36-35b-moe

# 查看实时日志
docker compose logs -f
docker compose logs -f --tail=100 -t qwen36-35b-moe

# 修改 docker-compose.yml 后重新创建单个服务
docker compose up -d --force-recreate qwen36-35b-moe

# 重启已运行服务
docker compose restart qwen36-35b-moe

# 仅停止服务，保留容器
docker compose stop

# 停止并删除容器；模型文件和镜像不会被删除
docker compose down
```

> `docker compose restart` 不会重新解析并应用所有配置变更。修改镜像、环境变量、卷、设备或 `ulimits` 后，应使用 `docker compose up -d --force-recreate <service>`。

### 2.2 资源监控

```bash
# Jetson 整机状态
jtop

# 容器 CPU、内存和 I/O
docker stats
```

## 3. 模型更新

仓库脚本 [`scripts/update-models.py`](scripts/update-models.py) 统一维护每个服务对应的 ModelScope 仓库、GGUF 文件、MTP 附属文件和本地路径。

```bash
# 先检查将要执行的 ModelScope 命令，不产生网络请求
python3 scripts/update-models.py --dry-run gemma4-31b

# 更新一个或多个模型
python3 scripts/update-models.py gemma4-31b gemma4-12b-agentic

# 更新全部模型；请先确认磁盘空间充足
python3 scripts/update-models.py --all

# 使用其他模型目录；必须同步修改 docker-compose.yml 中的模型路径
python3 scripts/update-models.py --model-dir /mnt/ssd/huggingface-test gemma4-31b
```

下载成功后重启对应服务：

```bash
docker compose restart gemma4-31b
```

如需手动下载，可安装 ModelScope 后调用 `modelscope download`。推荐使用 `uv` 隔离安装，避免污染系统 Python：

```bash
uv tool install --python 3.12 modelscope
modelscope download unsloth/gemma-4-31B-it-qat-GGUF \
    gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    --local-dir /mnt/ssd/huggingface --max-workers 1
```

## 4. 更新 llama.cpp 引擎

容器不编译 `llama.cpp`。宿主机编译并安装到 `/usr/local` 后，Compose 会将 `/usr/local/bin` 和 `/usr/local/lib` 只读挂载到容器内的 `/opt/llama`。因此引擎更新后只需要重启容器，不需要重新构建镜像。

```bash
sudo apt install ninja-build
git submodule update --init --remote --checkout llama.cpp

cmake -S llama.cpp -B llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NO_VMM=ON

cmake --build llama.cpp/build --config Release --parallel
sudo cmake --install llama.cpp/build --prefix /usr/local
sudo ldconfig
```

更新引擎后不需要立即执行重启命令。`llama-server` 来自宿主机挂载，服务关闭时，下次启动该服务就会自动使用新二进制：

```bash
docker compose up -d gemma4-31b
```

如果对应服务正在运行，再逐个重启它：

```bash
docker compose restart gemma4-31b
```

> 若 `/usr/local/lib` 不在动态链接器搜索路径中，执行 `echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/usr_local_lib.conf && sudo ldconfig`。
>
> `.gitmodules` 已将 `llama.cpp` 跟踪分支设置为 `master`，并忽略子模块指针变化，因此更新后主仓库不会要求提交子模块指针。

## 5. Docker 运行时说明

### 5.1 架构

- `Dockerfile` 只安装 `libcurl4` 和 `libgomp1` 等运行时依赖，不编译推理引擎。
- 模型缓存通过 `/mnt/ssd/huggingface:/root/.cache/huggingface` 共享。
- `llama-server` 二进制和共享库从宿主机 `/usr/local` 挂载到 `/opt/llama`，避开 NVIDIA CDI hook 对标准库路径的扫描。
- `runtime: nvidia`、`NVIDIA_VISIBLE_DEVICES=all` 和 `devices: nvidia.com/gpu=all` 提供 GPU 访问。
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 启用 Jetson UMA 统一内存调度。

### 5.2 更新 Compose 配置

```bash
# 先验证 YAML、锚点和变量插值
docker compose config --quiet

# 更新一个服务
docker compose up -d qwen36-35b-moe

# 如果某个服务没有自动应用新配置，再强制重建该服务
docker compose up -d --force-recreate qwen36-35b-moe
```

首次执行 `docker compose up -d <service>` 时，如果本地镜像不存在，Compose 会自动构建。

更新 `llama.cpp` 时，不需要重新构建镜像，也不需要清理 Docker 构建缓存；`llama-server` 二进制来自宿主机挂载，新容器启动时会直接使用新版引擎。

更新 `Dockerfile` 时，才需要重新构建镜像；模型卷和宿主机 `/usr/local` 挂载不受影响：

```bash
docker compose build <service>
docker compose up -d --force-recreate <service>
```

更新 `docker-compose.yml` 时，不需要重新构建镜像。`up -d` 会检测 Compose 配置变化并重建受影响的服务；如果某个服务没有变化，就不会被重建：

```bash
docker compose up -d <service>
```

### 5.3 NVIDIA Container Toolkit 兼容性修复

JetPack 7.2 上，`nvidia-container-toolkit 1.19.1` 的默认 `mode = "auto"` 可能触发：

```text
panic: runtime error: slice bounds out of range [:73] with capacity 71
```

按以下方式调整宿主机配置：

1. 编辑 `/etc/nvidia-container-runtime/config.toml`：

   ```toml
   [nvidia-container-runtime]
   mode = "cdi"
   ```

2. 在 `/etc/nvidia-container-toolkit/nvidia-cdi-refresh.env` 中禁用会出错的 hook：

   ```bash
   NVIDIA_CTK_CDI_GENERATE_DISABLED_HOOKS=enable-cuda-compat,update-ldcache
   ```

3. 重新生成 CDI 配置并重启容器服务：

   ```bash
   sudo systemctl restart nvidia-cdi-refresh.service
   sudo systemctl restart containerd
   sudo systemctl restart docker
   ```

由于禁用了 `update-ldcache`，Compose 已手动设置：

```bash
LD_LIBRARY_PATH=/opt/llama/lib:/usr/lib/aarch64-linux-gnu/nvidia:/opt/nvidia/l4t-gpu-libs/nvgpu
```

不要添加 `openrm` 路径，否则 CUDA 设备可能无法识别。

## 6. 性能优化与故障排查

### 6.1 扩大 Swap

长上下文或多个大模型可能超过 64GB 物理内存，导致容器被 OOM killer 终止。可在 SSD 上创建 16GB swap：

```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo swapoff -a
sudo dd if=/dev/zero of=/mnt/ssd/swapfile bs=1G count=16
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile
sudo swapon /mnt/ssd/swapfile
```

若需开机自动挂载，向 `/etc/fstab` 追加：

```text
/mnt/ssd/swapfile swap swap defaults 0 0
```

### 6.2 常见问题

**`failed to mlock ... Cannot allocate memory`**

容器默认的 memlock 限制不足。Compose 已设置：

```yaml
ulimits:
  memlock:
    soft: -1
    hard: -1
```

修改该配置后执行 `docker compose up -d --force-recreate <service>` 重新创建容器。

**容器无法看到 GPU**

依次检查：

```bash
docker info | grep -A2 -i runtime
sudo systemctl status nvidia-cdi-refresh.service
docker compose config | grep -A8 -E 'runtime:|devices:'
```

确认 `nvidia` runtime 已注册，CDI 配置已刷新，且 Compose 服务包含 `runtime: nvidia` 与 GPU 设备。

## 7. Pi Agent 接入

完整示例配置见 `backup/pi-agent` 和 `backup/prime-agent`。以下是最小接入步骤。

### 7.1 注册本地模型

创建或修改 `~/.pi/agent/models.json`。IP 需替换为 Jetson 实际地址：

```json
{
  "providers": {
    "local-gemma-12b-agentic": {
      "baseUrl": "http://192.168.137.13:8082/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "gemma4-v2-Q6_K",
          "name": "Local Gemma-4 12B Agentic",
          "contextWindow": 65536,
          "maxOutputTokens": 8192,
          "input": ["text"]
        }
      ]
    },
    "local-qwen36-35b-moe": {
      "baseUrl": "http://192.168.137.13:8084/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "qwen3.6-35b-moe",
          "name": "Local Qwen3.6 35B MoE",
          "contextWindow": 131072,
          "maxOutputTokens": 8192,
          "input": ["text"]
        }
      ]
    }
  }
}
```

其他服务可按同一格式注册，端口见第 2 节。完整历史示例保留在 `backup/pi-agent`。

### 7.2 设置默认模型

在 `~/.pi/agent/settings.json` 中配置：

```json
{
  "defaultProvider": "local-gemma-12b-agentic",
  "defaultModel": "gemma4-v2-Q6_K",
  "defaultThinkingLevel": "off"
}
```

会话中可输入 `/model` 切换模型，例如：

```text
/model local-gemma-12b-agentic/gemma4-v2-Q6_K
/model local-qwen36-35b-moe/qwen3.6-35b-moe
```

## 8. Forge Proxy 接入

[Forge](https://github.com/antoinezambelli/forge) 是自托管 LLM tool-calling 可靠性代理，用于在客户端和 `llama-server` 之间增强工具调用解析、校验和重试。

### 8.1 安装与启动

```bash
uv tool install --python 3.12 forge-guardrails
forge-proxy --help
```

推荐让 Forge 端口等于后端端口 + 1000。示例：

```bash
forge-proxy --backend-url http://localhost:8084 \
  --host 0.0.0.0 --port 9084 --serialize -v
```

`--serialize` 强制串行化请求，防止单 GPU 后端并发导致 OOM。后台运行可用 `tmux`：

```bash
tmux new-session -d -s forge \
  'forge-proxy --backend-url http://localhost:8084 --host 0.0.0.0 --port 9084 --serialize -v'
tmux attach -t forge
```

分离会话按 `Ctrl+B`，然后按 `D`。

### 8.2 客户端配置

将 OpenAI 兼容客户端的 `base_url` 指向 Forge 端口：

| 服务 | 后端 URL | Forge URL |
| --- | --- | --- |
| `gemma4-31b` | `http://<IP>:8080/v1` | `http://<IP>:9080/v1` |
| `gemma4-26b-a4b` | `http://<IP>:8081/v1` | `http://<IP>:9081/v1` |
| `gemma4-12b-agentic` | `http://<IP>:8082/v1` | `http://<IP>:9082/v1` |
| `qwen36-35b-moe` | `http://<IP>:8084/v1` | `http://<IP>:9084/v1` |
| `qwen38-27b` | `http://<IP>:8085/v1` | `http://<IP>:9085/v1` |

例如 `aider`：

```bash
aider --openai-api-base http://<JETSON_IP>:9084/v1 --openai-api-key not-needed
```

## 9. 参考资源

- [NVIDIA AI IOT](https://github.com/NVIDIA-AI-IOT)
- [Jetson Containers Quickstart](https://forums.developer.nvidia.com/t/jetson-containers-quickstart-on-nvidia-jetson-agx-orin-64GB/365700)
- [Forge 官方仓库](https://github.com/antoinezambelli/forge)

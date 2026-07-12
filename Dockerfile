# =============================================================================
# 针对 JetPack 7.2 (L4T 39.2 / Ubuntu 24.04 / CUDA 13.2.1) 的 llama.cpp 镜像
# 目标平台: NVIDIA Jetson Orin AGX (Ampere, CUDA Compute Capability sm_87)
#
# 背景说明:
#   JetPack 7.2 (L4T 39.2) 相比 JetPack 6 (L4T 36.x) 是重大版本升级:
#     - CUDA: 12.6  →  13.2.1
#     - OS:   Ubuntu 22.04 → Ubuntu 24.04
#   旧的 ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin 镜像基于
#   JetPack 6 / CUDA 12.x 构建，在 JetPack 7.2 主机上会因 CUDA ABI 不匹配
#   导致 "operation not supported" 错误或直接崩溃。
#
#   因此，本 Dockerfile 直接使用 NVIDIA NGC 官方 CUDA 13.2 devel 镜像
#   作为基础，在容器内编译最新版 llama.cpp，以完全匹配 JetPack 7.2 环境。
#
# 构建方式:
#   docker compose build           # 首次构建 (约 20~40 分钟)
#   docker compose up -d           # 启动服务 (和以前完全一样)
#
# 更新到最新版 llama.cpp:
#   docker compose build --no-cache && docker compose up -d
# =============================================================================

# JetPack 7.2 = L4T 39.2 = CUDA 13.2.1 = Ubuntu 24.04
# 使用 NVIDIA NGC 官方 CUDA devel 镜像 (含 nvcc 编译器和 cuDNN 开发头文件)
# 该镜像支持 ARM64 / SBSA，与 JetPack 7.2 的 NVIDIA Container Toolkit 完全兼容
#
# 注意: 在 Jetson 上运行时，NVIDIA Container Toolkit (nvidia-container-runtime)
# 会自动将宿主机的 GPU 驱动库挂载进容器，因此无需在镜像内安装驱动。
FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04

# -----------------------------------------------------------------------------
# 安装编译依赖
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libcurl4-openssl-dev \
        ccache \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ccache 加速 (下次重复构建时有效)
ENV CCACHE_DIR=/tmp/ccache
ENV PATH="/usr/lib/ccache:${PATH}"

# =============================================================================
# 编译 llama.cpp (最新 master)
# =============================================================================
WORKDIR /opt

RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp.git llama.cpp

WORKDIR /opt/llama.cpp

# -----------------------------------------------------------------------
# CMake 参数说明 (针对 JetPack 7.2 / Jetson Orin AGX 优化):
#
# [平台 & 基础]
#   -DGGML_CUDA=ON                    启用 CUDA 后端 (CUDA 13.2.1)
#   -DCMAKE_CUDA_ARCHITECTURES=87     仅编译 sm_87 (Orin AGX)，
#                                     跳过无关架构，大幅缩短编译时间
#   -DCMAKE_BUILD_TYPE=Release        启用 -O3 编译优化
#
# [Flash Attention & FP16 Tensor Core]
#   -DGGML_CUDA_F16=ON                Flash Attention 使用 FP16，
#                                     利用 Ampere Tensor Core 加速，
#                                     长上下文 (32K+) 推理速度显著提升
#   -DGGML_CUDA_FA_ALL_QUANTS=ON      对 Q4/Q8 等所有量化格式均启用
#                                     Flash Attention，防止 fallback 到慢路径
#
# [矩阵-向量乘法内核调优]
#   -DGGML_CUDA_DMMV_X=64            X 维并行度，Orin AGX 有 2048 CUDA Cores，
#                                     64 可充分利用 SM 资源，两者必须为 2 的幂
#   -DGGML_CUDA_MMV_Y=2              Y 维并行度，与 DMMV_X 配合，
#                                     社区实测可获得 10~20% 推理加速
#
# [Jetson UMA 统一内存架构专项优化]
#   -DGGML_CUDA_NO_VMM=ON            禁用 CUDA VMM 虚拟内存大块预分配。
#                                     Jetson CPU/GPU 共享物理内存 (UMA)，
#                                     VMM 大块预分配会因内存碎片触发 OOM，
#                                     禁用后改为按需小块分配，更适合 UMA 架构。
#                                     在 JetPack 7 / L4T 39.x 上尤为重要。
#
# [Server 功能]
#   -DLLAMA_CURL=ON                   llama-server 支持从 URL 加载模型
#
# [构建工具]
#   -G Ninja                          Ninja 构建比 Make 更快 (并行度更高)
#
# [暂未启用 - 可按需测试]
#   -DGGML_CUDA_GRAPHS=ON            CUDA Graphs 可降低调度延迟提升吞吐，
#                                     但在部分 JetPack/L4T 版本上有兼容问题，
#                                     JetPack 7.2 社区尚无充分测试数据，
#                                     待社区验证稳定后可取消注释。
#   -DGGML_SCHED_MAX_COPIES=1        仅在单 GPU 且显存极度紧张时启用
#                                     (会降低批处理吞吐量)
# -----------------------------------------------------------------------
RUN cmake -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_F16=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_DMMV_X=64 \
        -DGGML_CUDA_MMV_Y=2 \
        -DGGML_CUDA_NO_VMM=ON \
        -DLLAMA_CURL=ON \
    && cmake --build build --parallel \
    && cmake --install build --prefix /usr/local \
    && rm -rf /opt/llama.cpp

# 验证安装
RUN llama-server --version

# 运行时环境变量说明 (已在 docker-compose.yml 中统一设置):
#   GGML_CUDA_ENABLE_UNIFIED_MEMORY=1  运行时启用 UMA 统一内存调度，
#                                      社区实测可提升推理性能约 10~15%

WORKDIR /

# CMD 由 docker-compose.yml 的 command 字段控制，此处不写死

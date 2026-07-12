# =============================================================================
# 针对 JetPack 7.2 (L4T 39.2 / Ubuntu 24.04 / CUDA 13.2.1) 的 llama-server 运行环境
# 目标平台: NVIDIA Jetson Orin AGX (Ampere, CUDA Compute Capability sm_87)
#
# 设计思路:
#   llama.cpp 在宿主机上编译并安装到 /usr/local，通过 docker-compose.yml 的
#   volume 挂载注入容器，容器本身只需提供运行时依赖库（libcurl 等）。
#
#   优势:
#     - Docker build 几乎瞬间完成（无编译步骤）
#     - 宿主机更新 llama.cpp 后只需 docker compose restart，无需重新 build
#     - llama.cpp 源码由 git submodule (./llama.cpp) 统一管理
#
#   宿主机更新 llama.cpp 并重新编译:
#     cd llama.cpp && git pull && cd ..
#     cmake -B llama.cpp/build -G Ninja \
#         -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
#         -DGGML_CUDA_DMMV_X=64 -DGGML_CUDA_MMV_Y=2 \
#         -DGGML_CUDA_NO_VMM=ON -DLLAMA_CURL=ON
#     cmake --build llama.cpp/build --parallel
#     sudo cmake --install llama.cpp/build --prefix /usr/local
#     sudo ldconfig
#     docker compose restart   # 无需 build，直接重启即可
#
# 注意: 在 Jetson 上 runtime: nvidia 会自动将宿主机 CUDA 13.2.1 驱动库
#       挂载进容器，宿主机编译的 ARM aarch64 二进制在容器内可直接运行。
# =============================================================================

# 使用与宿主机 OS 一致的 Ubuntu 24.04 基础镜像（仅运行时，无需 devel）
# runtime 镜像体积远小于 devel 镜像
FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04

# 安装运行时依赖
# libcurl4: llama-server 的 URL 模型下载支持
# libgomp1: OpenMP 并行计算支持（llama.cpp 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /usr/local/bin 和 /usr/local/lib 由 docker-compose.yml 从宿主机挂载
# 此处无需任何编译或安装步骤

WORKDIR /

# CMD 由 docker-compose.yml 的 command 字段控制

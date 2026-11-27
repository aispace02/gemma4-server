# Graph Report - /home/hxf0223/tmp/gemma-server  (2026-11-27)

## Corpus Check
- Corpus is ~3,888 words - fits in a single context window. You may not need a graph.

## Summary
- 51 nodes · 85 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Runtime Image & Service Containers|Runtime Image & Service Containers]]
- [[_COMMUNITY_Compose Anchors, Mounts & CDI|Compose Anchors, Mounts & CDI]]
- [[_COMMUNITY_File Fingerprinting & CLI Args|File Fingerprinting & CLI Args]]
- [[_COMMUNITY_ModelScope Env & CLI Runner|ModelScope Env & CLI Runner]]
- [[_COMMUNITY_Model Download & Date Check|Model Download & Date Check]]
- [[_COMMUNITY_Model Spec Registry|Model Spec Registry]]
- [[_COMMUNITY_Update Result Reporting|Update Result Reporting]]
- [[_COMMUNITY_Timestamp Formatting Helper|Timestamp Formatting Helper]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 12 edges
2. `download_model()` - 11 edges
3. `x-llama-common Compose Anchor` - 8 edges
4. `x-sd-common Compose Anchor` - 8 edges
5. `ModelSpec` - 6 edges
6. `file_snapshot()` - 6 edges
7. `ModelResult` - 5 edges
8. `qwen38-27b LLM Service Container (port 8085)` - 5 edges
9. `FileSnapshot` - 4 edges
10. `file_was_changed()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `ModelSpec` --references--> `qwen38-27b LLM Service Container (port 8085)`  [EXTRACTED]
  scripts/update-models.py → docker-compose.yml
- `ModelSpec` --references--> `qwen-image Service Container (port 8083)`  [EXTRACTED]
  scripts/update-models.py → docker-compose.yml
- `download_model()` --shares_data_with--> `Shared Model Volume (/mnt/ssd/huggingface -> /root/.cache/huggingface)`  [EXTRACTED]
  scripts/update-models.py → docker-compose.yml
- `Pi & Prime Agent Client Config Docs` --references--> `qwen38-27b LLM Service Container (port 8085)`  [INFERRED]
  docs/backup/README.md → docker-compose.yml
- `README Section 0 & Section 5 Docker Architecture Docs` --references--> `qwen38-27b LLM Service Container (port 8085)`  [EXTRACTED]
  README.md → docker-compose.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Shared Jetson Docker Runtime Stack** — dockerfile_runtime_image, docker_compose_host_model_volume, docker_compose_nvidia_cdi_gpu, docker_compose_uma_env, docker_compose_host_network, docker_compose_qwen38_27b, docker_compose_qwen_image [EXTRACTED 1.00]
- **JetPack 7.2 CDI Panic Workaround Chain** — readme_nvidia_cdi_fix, docker_compose_llama_mount, docker_compose_ld_library_path [EXTRACTED 1.00]

## Communities (8 total, 1 thin omitted)

### Community 0 - "Runtime Image & Service Containers"
Cohesion: 0.18
Nodes (11): Unlimited Memlock ulimit (soft/hard: -1), qwen38-27b LLM Service Container (port 8085), qwen-image Service Container (port 8083), Host /opt/sdcpp/bin -> /opt/sdcpp/bin:ro Mount, NGC CUDA 13.2.1 Runtime Base Image, Runtime System Libraries (libcurl4, libgomp1), Shared Docker Runtime Image, Pi & Prime Agent Client Config Docs (+3 more)

### Community 1 - "Compose Anchors, Mounts & CDI"
Cohesion: 0.31
Nodes (10): Shared Model Volume (/mnt/ssd/huggingface -> /root/.cache/huggingface), Host Network Mode (network_mode: host), Manual LD_LIBRARY_PATH (nvgpu, no openrm), x-llama-common Compose Anchor, Host /usr/local -> /opt/llama:ro Mount, NVIDIA Runtime & CDI GPU Injection (nvidia.com/gpu=all), x-sd-common Compose Anchor, Jetson UMA Unified Memory (GGML_CUDA_ENABLE_UNIFIED_MEMORY=1) (+2 more)

### Community 2 - "File Fingerprinting & CLI Args"
Cohesion: 0.22
Nodes (9): ArgumentParser, Path, build_parser(), file_snapshot(), file_was_changed(), FileSnapshot, Return metadata used to detect whether a download replaced the file., Detect a changed/new file without hashing a 20–40 GiB model again. (+1 more)

### Community 3 - "ModelScope Env & CLI Runner"
Cohesion: 0.33
Nodes (6): direct_download_environment(), downloader_command(), main(), print_models(), Return a child environment that cannot inherit a shell proxy., Find ModelScope, or provide an isolated uv fallback.

### Community 4 - "Model Download & Date Check"
Cohesion: 0.40
Nodes (5): download_model(), fetch_remote_file_date(), print_date_check(), Explain the coarse date comparison before the downloader runs., Fetch the target file's commit date from ModelScope's file API.      The reposit

### Community 5 - "Model Spec Registry"
Cohesion: 0.67
Nodes (3): ModelSpec, A model file expected by one docker-compose service., select_models()

### Community 6 - "Update Result Reporting"
Cohesion: 0.50
Nodes (4): ModelResult, print_summary(), Print one final, easy-to-scan status for every selected model., Outcome of one model update attempt.

## Knowledge Gaps
- **3 isolated node(s):** `Host-Compiled llama.cpp Engine (/usr/local)`, `qwen-image APIs (/v1, /sdapi/v1, /sdcpp/v1)`, `Pi & Prime Agent Client Config Docs`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `x-llama-common Compose Anchor` connect `Compose Anchors, Mounts & CDI` to `Runtime Image & Service Containers`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Why does `x-sd-common Compose Anchor` connect `Compose Anchors, Mounts & CDI` to `Runtime Image & Service Containers`?**
  _High betweenness centrality (0.004) - this node is a cross-community bridge._
- **Why does `download_model()` connect `Model Download & Date Check` to `Compose Anchors, Mounts & CDI`, `File Fingerprinting & CLI Args`, `Model Spec Registry`, `Update Result Reporting`, `Timestamp Formatting Helper`?**
  _High betweenness centrality (0.004) - this node is a cross-community bridge._
- **What connects `A model file expected by one docker-compose service.`, `Small, cheap fingerprint for a potentially multi-gigabyte model file.`, `Outcome of one model update attempt.` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
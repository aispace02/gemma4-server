#!/usr/bin/env python3
"""Update the GGUF models used by this repository from ModelScope.

The downloader is deliberately kept outside Docker: the model directory is a
host volume and the containers only need to be restarted after a successful
download.  All standard proxy environment variables are removed before
starting ModelScope (or uv), so a proxy configured in the calling shell cannot
silently be used for the model download.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass


DEFAULT_MODEL_DIR = "/mnt/ssd/huggingface"


@dataclass(frozen=True)
class ModelSpec:
    """A model file expected by one docker-compose service."""

    service: str
    repository: str
    filename: str
    description: str


@dataclass(frozen=True)
class FileSnapshot:
    """Small, cheap fingerprint for a potentially multi-gigabyte model file."""

    size: int
    mtime_ns: int
    inode: int


@dataclass(frozen=True)
class ModelResult:
    """Outcome of one model update attempt."""

    spec: ModelSpec
    status: str
    target: Path
    size: int | None = None
    local_date: str | None = None
    remote_date: str | None = None
    detail: str = ""


STATUS_UPDATED = "已更新"
STATUS_UNCHANGED = "已是最新"
STATUS_DRY_RUN = "仅演练"
STATUS_FAILED = "失败"


MODEL_SPECS = (
    ModelSpec(
        "gemma4-31b",
        "unsloth/gemma-4-31B-it-qat-GGUF",
        "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "Gemma-4 31B Dense QAT",
    ),
    ModelSpec(
        "gemma4-26b-a4b",
        "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "Gemma-4 26B-A4B MoE QAT",
    ),
    ModelSpec(
        "gemma4-12b-agentic",
        "hf/yuxinlu1-gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF",
        "gemma4-v2-Q6_K.gguf",
        "Gemma-4 12B Agentic",
    ),
    ModelSpec(
        "qwen36-35b-moe",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "Qwen3.6 35B-A3B MoE",
    ),
    ModelSpec(
        "qwen38-27b",
        "unsloth/Qwen3.8-27B-GGUF",
        "Qwen3.8-27B-Q4_K_M.gguf",
        "Qwen3.8 27B Dense",
    ),
)

MODEL_BY_SERVICE = {spec.service: spec for spec in MODEL_SPECS}

# requests, urllib3, curl and git recognize these names.  Keep both cases:
# environment variable names are case-insensitive on Windows and commonly
# duplicated with different casing on Linux.
PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "no_proxy",
    "GIT_PROXY_COMMAND",
)


def direct_download_environment() -> dict[str, str]:
    """Return a child environment that cannot inherit a shell proxy."""

    environment = os.environ.copy()
    for variable in PROXY_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)

    # Keep localhost and every other host out of a proxy even if a dependency
    # supplies its own default proxy handling.
    environment["NO_PROXY"] = "*"
    environment["no_proxy"] = "*"

    # If uv has to bootstrap ModelScope, use a mainland package mirror too.
    # Removing these overrides avoids accidentally sending package resolution
    # through an inherited private/proxy index.
    for variable in (
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    ):
        environment.pop(variable, None)
    environment["UV_DEFAULT_INDEX"] = "https://pypi.tuna.tsinghua.edu.cn/simple"

    # Pin ModelScope to the mainland endpoint even when the parent shell has
    # configured the newer international endpoint-selection variables.
    environment["MODELSCOPE_ENDPOINT"] = "https://modelscope.cn"
    environment["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"
    environment["MODELSCOPE_PREFER_AI_SITE"] = "false"

    return environment


def downloader_command() -> list[str]:
    """Find ModelScope, or provide an isolated uv fallback."""

    installed_modelscope = shutil.which("modelscope")
    if installed_modelscope:
        return [installed_modelscope]

    installed_uv = shutil.which("uv")
    if installed_uv:
        return [
            installed_uv,
            "run",
            "--no-project",
            "--no-env-file",
            "--with",
            "modelscope",
            "modelscope",
        ]

    raise RuntimeError(
        "未找到 modelscope 或 uv。请安装 uv（https://docs.astral.sh/uv/），"
        "或先安装 ModelScope：python3 -m pip install modelscope"
    )


def print_models() -> None:
    for spec in MODEL_SPECS:
        print(f"{spec.service:20} {spec.description:28} {spec.repository}/{spec.filename}")


def select_models(requested: list[str], download_all: bool) -> list[ModelSpec]:
    if download_all and requested:
        raise ValueError("--all 不能和模型名称同时使用")
    if not download_all and not requested:
        raise ValueError("请指定模型名称，或使用 --all 下载全部模型")

    names = [spec.service for spec in MODEL_SPECS] if download_all else requested
    unknown = [name for name in names if name not in MODEL_BY_SERVICE]
    if unknown:
        available = ", ".join(MODEL_BY_SERVICE)
        raise ValueError(f"未知模型: {', '.join(unknown)}；可选值: {available}")

    # Preserve the order supplied by the user while avoiding duplicate work.
    selected: list[ModelSpec] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            selected.append(MODEL_BY_SERVICE[name])
            seen.add(name)
    return selected


def file_snapshot(path: Path) -> FileSnapshot | None:
    """Return metadata used to detect whether a download replaced the file."""

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None

    if not path.is_file():
        return None

    return FileSnapshot(
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        inode=stat_result.st_ino,
    )


def file_was_changed(before: FileSnapshot | None, after: FileSnapshot) -> bool:
    """Detect a changed/new file without hashing a 20–40 GiB model again."""

    if before is None:
        return True
    return (
        before.size != after.size
        or before.mtime_ns != after.mtime_ns
        or before.inode != after.inode
    )


def format_mtime(mtime_ns: int) -> str:
    """Format nanosecond timestamp to YYYY-MM-DD string in local time."""

    dt = datetime.datetime.fromtimestamp(
        mtime_ns / 1e9, tz=datetime.timezone.utc
    ).astimezone()
    return dt.strftime("%Y-%m-%d")


def print_date_check(local_date: str | None, remote_date: str | None) -> None:
    """Explain the coarse date comparison before the downloader runs."""

    if not remote_date:
        return
    if not local_date:
        print(f"      日期判断: 本地文件不存在，远端文件: {remote_date}")
    elif remote_date > local_date:
        print(
            f"      日期判断: 远端文件较新（本地: {local_date} | 远端文件: {remote_date}），"
            "将检查并下载"
        )
    else:
        print(
            f"      日期判断: 本地日期不早于远端文件（本地: {local_date} | "
            f"远端文件: {remote_date}）"
        )


def fetch_remote_file_date(repository: str, filename: str) -> str | None:
    """Fetch the target file's commit date from ModelScope's file API.

    The repository's ``LastUpdatedTime`` can be changed by README or metadata
    edits, which is not sufficient to tell whether the GGUF being downloaded
    changed.  The file listing contains a ``CommittedDate`` for each file.
    """

    query = urllib.parse.urlencode({"Revision": "master", "Recursive": "true"})
    url = f"https://www.modelscope.cn/api/v1/models/{repository}/repo/files?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ModelScope-UpdateScript/1.0"},
    )
    handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)
    try:
        with opener.open(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("Data", {})
            files = data.get("Files", [])
            remote_file = next(
                (
                    item
                    for item in files
                    if item.get("Path") == filename
                    or item.get("Name") == filename
                ),
                None,
            )
            committed = remote_file.get("CommittedDate") if remote_file else None
            if isinstance(committed, (int, float)):
                dt = datetime.datetime.fromtimestamp(
                    committed, tz=datetime.timezone.utc
                ).astimezone()
                return dt.strftime("%Y-%m-%d")
            elif isinstance(committed, str) and len(committed) >= 10:
                return committed[:10]
    except Exception:
        pass
    return None


def download_model(
    spec: ModelSpec,
    command_prefix: list[str],
    model_dir: Path,
    revision: str | None,
    environment: dict[str, str],
    dry_run: bool,
    ordinal: str,
) -> ModelResult:
    command = command_prefix + [
        "download",
        "--model",
        spec.repository,
        spec.filename,
        "--max-workers",
        "1",
    ]
    if revision:
        command.extend(["--revision", revision])
    command.extend(["--local_dir", str(model_dir)])

    target = model_dir / spec.filename
    print(f"[{ordinal}] {spec.description} -> {target}")
    print(f"      ModelScope: {spec.repository}/{spec.filename}")

    remote_date = fetch_remote_file_date(spec.repository, spec.filename)
    before_check = file_snapshot(target)
    local_date = format_mtime(before_check.mtime_ns) if before_check else None
    print_date_check(local_date, remote_date)

    if dry_run:
        print(f"      dry-run: {shlex.join(command)}")
        print(f"      状态: {STATUS_DRY_RUN}")
        return ModelResult(
            spec,
            STATUS_DRY_RUN,
            target,
            local_date=local_date,
            remote_date=remote_date,
        )

    before = before_check

    result = subprocess.run(command, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ModelScope 下载失败（退出码 {result.returncode}）: {spec.service}"
        )

    after = file_snapshot(target)
    if after is None:
        raise RuntimeError(
            f"下载命令已返回成功，但没有找到目标文件: {target}\n"
            "请检查 ModelScope 仓库中的文件名是否发生变化。"
        )
    if after.size <= 0:
        raise RuntimeError(f"目标文件为空，已停止: {target}")

    status = STATUS_UPDATED if file_was_changed(before, after) else STATUS_UNCHANGED
    local_date = format_mtime(after.mtime_ns)

    date_parts = [f"本地: {local_date}"]
    if remote_date:
        date_parts.append(f"远端文件: {remote_date}")
    date_str = " | ".join(date_parts)

    print(f"      状态: {status}（{after.size / (1024**3):.2f} GiB | {date_str}）")
    return ModelResult(
        spec,
        status,
        target,
        size=after.size,
        local_date=local_date,
        remote_date=remote_date,
    )


def print_summary(results: list[ModelResult]) -> None:
    """Print one final, easy-to-scan status for every selected model."""

    symbols = {
        STATUS_UPDATED: "✓",
        STATUS_UNCHANGED: "=",
        STATUS_DRY_RUN: "·",
        STATUS_FAILED: "✗",
    }

    print("\n模型更新结果:")
    for result in results:
        symbol = symbols[result.status]

        date_parts = []
        if result.local_date:
            date_parts.append(f"本地: {result.local_date}")
        else:
            date_parts.append("本地: 未下载")

        if result.remote_date:
            date_parts.append(f"远端文件: {result.remote_date}")

        date_str = f"[{' | '.join(date_parts)}]"

        print(
            f"  {symbol} {result.spec.service:20} "
            f"{result.status:6} {date_str:27} {result.spec.description}"
        )
        if result.detail:
            print(f"      {result.detail}")

    counts = Counter(result.status for result in results)
    order = (STATUS_UPDATED, STATUS_UNCHANGED, STATUS_DRY_RUN, STATUS_FAILED)
    summary = "，".join(
        f"{status} {counts[status]} 个" for status in order if counts[status]
    )
    print(f"统计：{summary}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从中国大陆 ModelScope 更新本项目使用的 GGUF 模型"
            "（始终禁用网络代理）"
        )
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="SERVICE",
        help="Compose 服务名；可通过 --list 查看。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="更新全部模型（模型文件很大，请确认磁盘空间）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可更新的模型并退出",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser(),
        help=f"模型目录（默认: $MODEL_DIR 或 {DEFAULT_MODEL_DIR}）",
    )
    parser.add_argument(
        "--revision",
        help="可选的 ModelScope 分支、标签或提交；默认使用仓库最新版本。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的命令，不下载模型。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_models()
        return 0

    try:
        selected = select_models(args.models, args.all)
        command_prefix = downloader_command()
        environment = direct_download_environment()

        if not args.dry_run:
            args.model_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"model-dir: {args.model_dir}")
        print("download-source: https://modelscope.cn (mainland endpoint forced)")
        print("proxy: disabled (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and lowercase variants removed)")

        total = len(selected)
        results: list[ModelResult] = []
        for index, spec in enumerate(selected, start=1):
            try:
                results.append(
                    download_model(
                        spec,
                        command_prefix,
                        args.model_dir,
                        args.revision,
                        environment,
                        args.dry_run,
                        f"{index}/{total}",
                    )
                )
            except (OSError, RuntimeError) as error:
                target = args.model_dir / spec.filename
                snap = file_snapshot(target)
                local_date = format_mtime(snap.mtime_ns) if snap else None
                remote_date = fetch_remote_file_date(spec.repository, spec.filename)
                print(f"      状态: {STATUS_FAILED} — {error}", file=sys.stderr)
                results.append(
                    ModelResult(
                        spec,
                        STATUS_FAILED,
                        target,
                        local_date=local_date,
                        remote_date=remote_date,
                        detail=str(error),
                    )
                )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    print_summary(results)

    failed = [result for result in results if result.status == STATUS_FAILED]
    if failed:
        print(f"\n有 {len(failed)} 个模型更新失败，请根据上面的错误信息处理。", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n演练完成，未下载任何模型。")
    else:
        print(
            "\n模型更新检查完成。状态为“已更新”的模型已经下载成功；"
            "若对应服务正在运行，请执行 docker compose restart <服务名>。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

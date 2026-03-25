"""Model serving and deployment."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DeployConfig:
    """Deployment configuration."""

    model_path: str
    runtime: str = "mlx-lm"
    host: str = "127.0.0.1"
    port: int = 8080
    context_length: int = 4096


def save_config(
    strategy_dict: dict,
    model_path: Path,
    config_path: Path,
) -> Path:
    """Save optimization result as a YAML config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_path": str(model_path),
        **strategy_dict,
    }
    config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    return config_path


def serve_mlx(
    model_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> subprocess.Popen:
    """Start an mlx-lm server (OpenAI-compatible API).

    Returns the Popen object so the caller can manage the process.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def serve_ollama(
    model_name: str,
    modelfile_path: Path | None = None,
) -> bool:
    """Create and start an Ollama model.

    Returns True on success.
    """
    if modelfile_path:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False

    result = subprocess.run(
        ["ollama", "run", model_name, "--verbose"],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def format_deploy_info(config: DeployConfig) -> str:
    """Format deployment information for display."""
    lines = [
        "Deployment",
        "=" * 50,
        f"  Model:    {config.model_path}",
        f"  Runtime:  {config.runtime}",
        f"  Address:  http://{config.host}:{config.port}",
        f"  Context:  {config.context_length:,}",
        "",
        "  API Endpoints (OpenAI-compatible):",
        f"    POST http://{config.host}:{config.port}/v1/chat/completions",
        f"    POST http://{config.host}:{config.port}/v1/completions",
    ]
    return "\n".join(lines)

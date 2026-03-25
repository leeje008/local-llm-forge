"""Model serving and deployment."""

from __future__ import annotations

import shutil
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
    config = {"model_path": str(model_path), **strategy_dict}
    config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    return config_path


def serve_mlx(
    model_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> subprocess.Popen:
    """Start an mlx-lm server (OpenAI-compatible API)."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


# ---------------------------------------------------------------------------
# Ollama integration
# ---------------------------------------------------------------------------

def create_ollama_model(
    model_id: str,
    output_name: str | None = None,
    context_length: int = 4096,
    system_prompt: str | None = None,
    num_gpu: int = 99,
) -> tuple[bool, str]:
    """Register a HuggingFace model with Ollama via hf.co shorthand or Modelfile.

    Returns (success, message).
    """
    if not shutil.which("ollama"):
        return False, "Ollama is not installed"

    name = output_name or model_id.split("/")[-1].lower()

    # Try direct hf.co pull first (simplest path for GGUF models)
    result = subprocess.run(
        ["ollama", "show", f"hf.co/{model_id}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # Model already available via hf.co shorthand
        return True, f"Model available as: ollama run hf.co/{model_id}"

    # Create via Modelfile
    modelfile_content = _build_modelfile(
        model_ref=f"hf.co/{model_id}",
        context_length=context_length,
        system_prompt=system_prompt,
        num_gpu=num_gpu,
    )

    # Write temp Modelfile
    modelfile_path = Path(f"/tmp/forge-modelfile-{name}")
    modelfile_path.write_text(modelfile_content)

    result = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
        timeout=600,
    )
    modelfile_path.unlink(missing_ok=True)

    if result.returncode != 0:
        err = result.stderr or result.stdout
        return False, f"ollama create failed: {err[:300]}"

    return True, f"Model registered as: ollama run {name}"


def create_ollama_from_gguf(
    gguf_path: Path,
    model_name: str,
    context_length: int = 4096,
    system_prompt: str | None = None,
    num_gpu: int = 99,
) -> tuple[bool, str]:
    """Create an Ollama model from a local GGUF file."""
    if not shutil.which("ollama"):
        return False, "Ollama is not installed"

    if not gguf_path.exists():
        return False, f"GGUF file not found: {gguf_path}"

    modelfile_content = _build_modelfile(
        model_ref=str(gguf_path),
        context_length=context_length,
        system_prompt=system_prompt,
        num_gpu=num_gpu,
    )

    modelfile_path = Path(f"/tmp/forge-modelfile-{model_name}")
    modelfile_path.write_text(modelfile_content)

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
        timeout=600,
    )
    modelfile_path.unlink(missing_ok=True)

    if result.returncode != 0:
        err = result.stderr or result.stdout
        return False, f"ollama create failed: {err[:300]}"

    return True, f"Model registered as: ollama run {model_name}"


def _build_modelfile(
    model_ref: str,
    context_length: int = 4096,
    system_prompt: str | None = None,
    num_gpu: int = 99,
) -> str:
    """Build Ollama Modelfile content."""
    lines = [f"FROM {model_ref}"]
    lines.append(f"PARAMETER num_ctx {context_length}")
    lines.append(f"PARAMETER num_gpu {num_gpu}")
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    return "\n".join(lines) + "\n"


def list_ollama_models() -> list[dict]:
    """List models registered in Ollama."""
    if not shutil.which("ollama"):
        return []
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []

    models = []
    for line in result.stdout.strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append({"name": parts[0], "size": parts[2] if len(parts) > 2 else "?"})
    return models


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

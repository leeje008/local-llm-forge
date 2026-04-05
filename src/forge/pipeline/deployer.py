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
    # Phase 10 advanced scheduling options
    chunked_prefill: bool = False
    chunk_size: int = 512
    interruptible: bool = False
    multi_model: bool = False
    pmpd: bool = False
    use_vllm_mlx: bool = False
    memory_budget_gb: float = 32.0

    def advanced_enabled(self) -> bool:
        return any([
            self.chunked_prefill,
            self.interruptible,
            self.multi_model,
            self.pmpd,
            self.use_vllm_mlx,
        ])


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


def serve_advanced(config: DeployConfig) -> subprocess.Popen:
    """Start a server with Phase 10 advanced scheduling hooks enabled.

    Builds a :class:`forge.engine.scheduler.AdvancedServingContext` with
    the chunked-prefill / interruptible / multi-model / PMPD / vllm-mlx
    components requested on ``config``, then hands off to
    :func:`serve_mlx` as the actual transport. The advanced context is
    attached to the returned process object as ``.forge_context`` so
    that callers (tests, future HTTP glue) can inspect or drive it.

    This keeps the control plane decoupled from the mlx-lm server
    binary: when upstream exposes request-level hooks, the runner can be
    swapped for one that actually consumes the scheduler's work units.
    """
    from forge.engine import scheduler as _sched
    from forge.engine.mlx_engine import EngineConfig

    base_cfg = EngineConfig(
        model_path=config.model_path,
        pmpd_mode=config.pmpd,
    )

    ctx = _sched.build_context(
        model_path=config.model_path,
        chunked_prefill=config.chunked_prefill,
        chunk_size=config.chunk_size,
        interruptible=config.interruptible,
        multi_model=config.multi_model,
        pmpd=config.pmpd,
        use_vllm_mlx=config.use_vllm_mlx,
        memory_budget_gb=config.memory_budget_gb,
        base_config=base_cfg,
    )

    # For now we still launch mlx_lm.server as the actual transport.
    # Once upstream exposes per-request hooks (or we ship our own HTTP
    # layer) this is where the runner would consume ``ctx.scheduler``.
    process = serve_mlx(config.model_path, host=config.host, port=config.port)
    try:
        process.forge_context = ctx  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return process


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

"""Model format conversion pipeline."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConvertResult:
    """Result of a model conversion."""

    output_path: Path
    format: str
    size_gb: float
    success: bool
    error: str | None = None


def download_model(model_id: str, cache_dir: Path | None = None) -> Path:
    """Download a model from HuggingFace Hub. Returns the local cache path."""
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    path = snapshot_download(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def convert_to_mlx(
    model_id: str,
    output_dir: Path,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 128,
) -> ConvertResult:
    """Convert a HuggingFace model to MLX format with optional quantization.

    Uses mlx-lm's convert command which handles download + conversion + quantization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_id,
        "--mlx-path", str(output_dir),
    ]
    if quantize:
        cmd.append("-q")
        cmd.extend(["--q-bits", str(bits)])
        cmd.extend(["--q-group-size", str(group_size)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            return ConvertResult(
                output_path=output_dir,
                format="mlx",
                size_gb=0,
                success=False,
                error=result.stderr[:500] if result.stderr else "Conversion failed",
            )

        size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        return ConvertResult(
            output_path=output_dir,
            format="mlx",
            size_gb=size / (1024**3),
            success=True,
        )
    except subprocess.TimeoutExpired:
        return ConvertResult(
            output_path=output_dir, format="mlx", size_gb=0,
            success=False, error="Conversion timed out (2 hours)",
        )
    except Exception as e:
        return ConvertResult(
            output_path=output_dir, format="mlx", size_gb=0,
            success=False, error=str(e),
        )


def create_ollama_modelfile(
    model_path: Path,
    output_path: Path,
    context_length: int = 4096,
    system_prompt: str | None = None,
) -> Path:
    """Generate an Ollama Modelfile for a GGUF model."""
    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {model_path}")

    modelfile_path = output_path / "Modelfile"
    lines = [f'FROM {gguf_files[0]}']
    lines.append(f"PARAMETER num_ctx {context_length}")
    lines.append("PARAMETER num_gpu 99")
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')

    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text("\n".join(lines) + "\n")
    return modelfile_path

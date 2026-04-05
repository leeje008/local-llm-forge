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


def convert_bitnet(
    model_id: str,
    output_dir: Path,
    bitnet_cpp_repo: Path | None = None,
) -> ConvertResult:
    """Convert a BitNet b1.58 model to a ``bitnet.cpp``-compatible GGUF (Phase 11.4).

    BitNet inference on Apple Silicon currently runs through Microsoft's
    ``bitnet.cpp`` project, which ships its own conversion script
    (``setup_env.py`` / ``convert-hf-to-gguf-bitnet.py``). This function
    is a thin control-plane wrapper: it downloads the HF model, invokes
    that script when the ``bitnet.cpp`` checkout is available, and
    writes the resulting artifact under ``output_dir``.

    If ``bitnet_cpp_repo`` is not supplied and no such checkout is on
    ``$PATH``, it returns a ``ConvertResult`` with ``success=False`` and
    a clear installation hint in ``error``.
    """
    from forge.engine.bitnet_engine import BitNetConfig  # structural link  # noqa: F401

    output_dir.mkdir(parents=True, exist_ok=True)
    repo = bitnet_cpp_repo
    if repo is None:
        candidates = [
            Path.home() / "src" / "BitNet",
            Path.home() / "projects" / "BitNet",
            Path("/opt/BitNet"),
        ]
        repo = next((c for c in candidates if c.exists()), None)
    if repo is None or not repo.exists():
        return ConvertResult(
            output_path=output_dir,
            format="bitnet-gguf",
            size_gb=0,
            success=False,
            error=(
                "bitnet.cpp repository not found. Clone https://github.com/microsoft/BitNet, "
                "run its setup_env.py, and pass the path via bitnet_cpp_repo."
            ),
        )

    script = repo / "utils" / "convert-hf-to-gguf-bitnet.py"
    if not script.exists():
        return ConvertResult(
            output_path=output_dir,
            format="bitnet-gguf",
            size_gb=0,
            success=False,
            error=f"Expected conversion script missing: {script}",
        )

    try:
        hf_path = download_model(model_id)
    except Exception as e:
        return ConvertResult(
            output_path=output_dir, format="bitnet-gguf", size_gb=0,
            success=False, error=f"download failed: {e}",
        )

    cmd = [
        sys.executable, str(script),
        str(hf_path),
        "--outdir", str(output_dir),
        "--outtype", "i2_s",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            return ConvertResult(
                output_path=output_dir, format="bitnet-gguf", size_gb=0,
                success=False,
                error=result.stderr[:500] if result.stderr else "bitnet conversion failed",
            )
        size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        return ConvertResult(
            output_path=output_dir,
            format="bitnet-gguf",
            size_gb=size / (1024**3),
            success=True,
        )
    except subprocess.TimeoutExpired:
        return ConvertResult(
            output_path=output_dir, format="bitnet-gguf", size_gb=0,
            success=False, error="bitnet conversion timed out",
        )
    except Exception as e:
        return ConvertResult(
            output_path=output_dir, format="bitnet-gguf", size_gb=0,
            success=False, error=str(e),
        )


def convert_hybrid(
    model_id: str,
    output_dir: Path,
    architecture_family: str = "hybrid-mamba",
) -> ConvertResult:
    """Prepare a hybrid Mamba / RWKV model for serving (Phase 11.1, 11.2).

    MLX 0.31.0 cannot natively execute Mamba or RWKV blocks, so this
    "conversion" simply materializes a local HuggingFace snapshot
    (which :class:`forge.engine.mamba_engine.HybridMambaEngine` /
    :class:`RWKV7Engine` will then load via ``transformers``) and tags
    the directory with a small ``forge_family.json`` marker so the
    deployer knows which engine to instantiate.
    """
    # Structural link to the engine module — also verifies it imports.
    from forge.engine.mamba_engine import (  # noqa: F401
        HybridMambaEngine,
        RWKV7Engine,
    )
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        hf_path = download_model(model_id, cache_dir=output_dir)
    except Exception as e:
        return ConvertResult(
            output_path=output_dir, format=architecture_family, size_gb=0,
            success=False, error=f"download failed: {e}",
        )

    marker = output_dir / "forge_family.json"
    marker.write_text(
        json.dumps(
            {
                "architecture_family": architecture_family,
                "source_model_id": model_id,
                "hf_snapshot": str(hf_path),
                "engine": "HybridMambaEngine" if architecture_family != "rwkv7" else "RWKV7Engine",
            },
            indent=2,
        )
    )
    size = sum(f.stat().st_size for f in Path(hf_path).rglob("*") if f.is_file())
    return ConvertResult(
        output_path=Path(hf_path),
        format=architecture_family,
        size_gb=size / (1024**3),
        success=True,
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

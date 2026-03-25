"""Quantization pipeline — wraps mlx-lm, HQQ, and AQLM."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuantResult:
    """Result of a quantization operation."""

    output_path: Path
    quant: str
    method: str
    size_gb: float
    success: bool
    error: str | None = None


def quantize_mlx(
    model_id: str,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 128,
    recipe: str | None = None,
) -> QuantResult:
    """Quantize a model using mlx-lm convert.

    This downloads the model from HuggingFace and converts + quantizes in one step.
    """
    # mlx_lm refuses to write to existing dirs — remove first
    if output_dir.exists():
        shutil.rmtree(output_dir)
    # Only create parent dirs; mlx_lm creates the output dir itself
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_id,
        "--mlx-path", str(output_dir),
    ]

    # Only add quantization flags for non-fp16
    if bits < 16:
        cmd.extend(["-q", "--q-bits", str(bits)])
        if group_size != 64:
            cmd.extend(["--q-group-size", str(group_size)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode != 0:
            return QuantResult(
                output_path=output_dir,
                quant=f"int{bits}",
                method="mlx_native",
                size_gb=0,
                success=False,
                error=result.stderr[:500] if result.stderr else "Unknown error",
            )

        # Calculate output size
        size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        size_gb = size / (1024**3)

        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="mlx_native",
            size_gb=size_gb,
            success=True,
        )
    except subprocess.TimeoutExpired:
        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="mlx_native",
            size_gb=0,
            success=False,
            error="Quantization timed out (1 hour limit)",
        )
    except Exception as e:
        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="mlx_native",
            size_gb=0,
            success=False,
            error=str(e),
        )


def quantize_hqq(
    model_id: str,
    output_dir: Path,
    bits: int = 3,
    group_size: int = 128,
) -> QuantResult:
    """Quantize a model using HQQ (Half-Quadratic Quantization).

    Requires the `hqq` package to be installed.
    """
    try:
        from hqq.core.quantize import BaseQuantizeConfig  # type: ignore[import-untyped]
        from hqq.models.hf.base import AutoHQQHFModel  # type: ignore[import-untyped]
    except ImportError:
        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="hqq",
            size_gb=0,
            success=False,
            error="HQQ not installed. Run: pip install 'local-llm-forge[quantization]'",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        quant_config = BaseQuantizeConfig(nbits=bits, group_size=group_size)
        model = AutoHQQHFModel.from_pretrained(model_id, quant_config=quant_config)
        model.save_quantized(str(output_dir))

        size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        size_gb = size / (1024**3)

        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="hqq",
            size_gb=size_gb,
            success=True,
        )
    except Exception as e:
        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method="hqq",
            size_gb=0,
            success=False,
            error=str(e),
        )


def quantize(
    model_id: str,
    output_dir: Path,
    method: str = "mlx_native",
    bits: int = 4,
    group_size: int = 128,
    recipe: str | None = None,
) -> QuantResult:
    """Unified quantization entry point."""
    if method == "mlx_native":
        return quantize_mlx(model_id, output_dir, bits, group_size, recipe)
    elif method == "hqq":
        return quantize_hqq(model_id, output_dir, bits, group_size)
    else:
        return QuantResult(
            output_path=output_dir,
            quant=f"int{bits}",
            method=method,
            size_gb=0,
            success=False,
            error=f"Unknown quantization method: {method}",
        )

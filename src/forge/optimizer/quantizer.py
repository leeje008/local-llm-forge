"""Quantization pipeline — wraps mlx-lm, HQQ, and AQLM."""

from __future__ import annotations

import json
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


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)


def quantize_mlx(
    model_id: str,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 128,
    recipe: str | None = None,
) -> QuantResult:
    """Quantize using mlx-lm convert (download + convert + quantize)."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_id,
        "--mlx-path", str(output_dir),
    ]
    if bits < 16:
        cmd.extend(["-q", "--q-bits", str(bits)])
        if group_size != 64:
            cmd.extend(["--q-group-size", str(group_size)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            return QuantResult(
                output_path=output_dir, quant=f"int{bits}", method="mlx_native",
                size_gb=0, success=False,
                error=(result.stderr or result.stdout)[:500],
            )
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="mlx_native",
            size_gb=_dir_size_gb(output_dir), success=True,
        )
    except subprocess.TimeoutExpired:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="mlx_native",
            size_gb=0, success=False, error="Timed out (1h limit)",
        )
    except Exception as e:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="mlx_native",
            size_gb=0, success=False, error=str(e),
        )


def quantize_hqq(
    model_id: str,
    output_dir: Path,
    bits: int = 3,
    group_size: int = 128,
    convert_to_mlx: bool = True,
) -> QuantResult:
    """Quantize using HQQ then optionally convert to MLX format.

    Pipeline: HuggingFace model → HQQ quantize (PyTorch) → save →
              optionally convert to MLX for fast inference.
    """
    try:
        import torch
        from hqq.core.quantize import BaseQuantizeConfig  # type: ignore[import-untyped]
        from hqq.models.hf.base import AutoHQQHFModel  # type: ignore[import-untyped]
    except ImportError:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq",
            size_gb=0, success=False,
            error="HQQ not installed. Run: pip install 'local-llm-forge[quantization]'",
        )

    hqq_dir = output_dir.parent / f"{output_dir.name}-hqq-tmp"
    if hqq_dir.exists():
        shutil.rmtree(hqq_dir)
    hqq_dir.mkdir(parents=True)

    try:
        # Step 1: Load and quantize with HQQ
        quant_config = BaseQuantizeConfig(nbits=bits, group_size=group_size)
        model = AutoHQQHFModel.from_pretrained(
            model_id,
            quant_config=quant_config,
            dtype=torch.float16,
            device_map="cpu",  # Apple Silicon — keep on CPU, no CUDA
        )

        # Step 2: Save HQQ quantized model
        model.save_quantized(str(hqq_dir))

        # Also save tokenizer
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(str(hqq_dir))

        if convert_to_mlx:
            # Step 3: Convert HQQ output to MLX format
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            mlx_cmd = [
                sys.executable, "-m", "mlx_lm", "convert",
                "--hf-path", str(hqq_dir),
                "--mlx-path", str(output_dir),
            ]
            result = subprocess.run(mlx_cmd, capture_output=True, text=True, timeout=3600)
            # Clean up temp dir
            shutil.rmtree(hqq_dir, ignore_errors=True)

            if result.returncode != 0:
                return QuantResult(
                    output_path=output_dir, quant=f"int{bits}", method="hqq",
                    size_gb=0, success=False,
                    error=f"MLX conversion failed: {(result.stderr or '')[:300]}",
                )
        else:
            # Keep as HQQ format
            if output_dir.exists():
                shutil.rmtree(output_dir)
            hqq_dir.rename(output_dir)

        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq",
            size_gb=_dir_size_gb(output_dir), success=True,
        )

    except Exception as e:
        shutil.rmtree(hqq_dir, ignore_errors=True)
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq",
            size_gb=0, success=False, error=str(e),
        )


def quantize_transformers_hqq(
    model_id: str,
    output_dir: Path,
    bits: int = 3,
    group_size: int = 128,
) -> QuantResult:
    """Quantize using HuggingFace Transformers HqqConfig integration.

    This is an alternative HQQ path using the native Transformers API.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig  # type: ignore
    except ImportError as e:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq_transformers",
            size_gb=0, success=False, error=f"Missing dependency: {e}",
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    try:
        quant_config = HqqConfig(nbits=bits, group_size=group_size)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq_transformers",
            size_gb=_dir_size_gb(output_dir), success=True,
        )
    except Exception as e:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method="hqq_transformers",
            size_gb=0, success=False, error=str(e),
        )


def quantize(
    model_id: str,
    output_dir: Path,
    method: str = "mlx_native",
    bits: int = 4,
    group_size: int = 128,
    recipe: str | None = None,
) -> QuantResult:
    """Unified quantization entry point with fallback chain."""
    if method == "mlx_native":
        return quantize_mlx(model_id, output_dir, bits, group_size, recipe)
    elif method == "hqq":
        result = quantize_hqq(model_id, output_dir, bits, group_size)
        if not result.success:
            # Fallback: try transformers HqqConfig path
            result2 = quantize_transformers_hqq(model_id, output_dir, bits, group_size)
            if result2.success:
                return result2
            # Fallback: try mlx_native at same bits
            result3 = quantize_mlx(model_id, output_dir, bits, group_size)
            if result3.success:
                result3.method = "mlx_native (hqq fallback)"
                return result3
        return result
    else:
        return QuantResult(
            output_path=output_dir, quant=f"int{bits}", method=method,
            size_gb=0, success=False, error=f"Unknown method: {method}",
        )

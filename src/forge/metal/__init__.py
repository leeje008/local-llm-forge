"""Custom Metal kernel utilities for Apple Silicon optimization.

These kernels provide fused operations that reduce kernel launch overhead
and memory round-trips compared to running operations separately.

Available kernels:
  - fused_rmsnorm_residual: RMSNorm(x + residual) in single dispatch
  - fused_silu_gate: SiLU(gate) * up (SwiGLU activation)
  - dequant_matvec_4bit: 4-bit dequantize + matrix-vector multiply

Usage with MLX:
  MLX uses its own Metal kernels internally. These custom kernels are
  provided as reference implementations and for use in custom inference
  pipelines that bypass mlx_lm's built-in generation.
"""

from pathlib import Path

KERNEL_SOURCE_PATH = Path(__file__).parent / "kernels.metal"


def get_kernel_source() -> str:
    """Read the Metal shader source code."""
    return KERNEL_SOURCE_PATH.read_text()


def list_kernels() -> list[dict]:
    """List available custom Metal kernels with descriptions."""
    return [
        {
            "name": "fused_rmsnorm_residual",
            "description": "Fused RMSNorm + residual addition (float32)",
            "saves": "1 kernel launch + 1 memory round-trip per layer",
        },
        {
            "name": "fused_rmsnorm_residual_half",
            "description": "Fused RMSNorm + residual addition (float16)",
            "saves": "1 kernel launch + 1 memory round-trip, half precision",
        },
        {
            "name": "fused_silu_gate",
            "description": "SwiGLU activation: silu(gate) * up (float32)",
            "saves": "1 kernel launch for FFN activation",
        },
        {
            "name": "fused_silu_gate_half",
            "description": "SwiGLU activation: silu(gate) * up (float16)",
            "saves": "1 kernel launch for FFN activation, half precision",
        },
        {
            "name": "dequant_matvec_4bit",
            "description": "4-bit dequantization + matrix-vector multiply",
            "saves": "Eliminates separate dequant pass, uses shared memory + LUT",
        },
    ]

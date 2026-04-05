"""Attention backend detection and reporting utilities (Phase 12 Tier S).

This module provides helpers to detect optional fused attention backends —
currently ``mlx-mfa`` (a pip wrapper around philipturner/metal-flash-attention)
— and produce human-readable reports for CLI display.

The heavy lifting (actually swapping the SDPA kernel inside mlx-lm's decode
loop) is performed by :class:`forge.engine.mlx_engine.MLXEngine`. This module
stays dependency-free and import-safe even when ``mlx_mfa`` is not installed.
"""

from __future__ import annotations

import importlib.util


def detect_mlx_mfa() -> tuple[bool, str]:
    """Check whether the ``mlx_mfa`` package is importable.

    Returns:
        A tuple ``(available, version_or_reason)``. On success the second
        element is the installed version (or ``"unknown"`` if the package
        does not expose ``__version__``). On failure it is a short human
        readable reason.
    """
    spec = importlib.util.find_spec("mlx_mfa")
    if spec is None:
        return False, "mlx-mfa not installed (pip install mlx-mfa)"
    try:
        import mlx_mfa  # type: ignore[import-not-found]
        version = getattr(mlx_mfa, "__version__", "unknown")
        return True, str(version)
    except ImportError:
        return False, "mlx-mfa not installed (pip install mlx-mfa)"
    except Exception as e:  # pragma: no cover — defensive
        return False, str(e)


def format_attention_backend_report(
    backend: str,
    detected: bool,
    reason: str,
    seq_len_threshold: int,
) -> str:
    """Format a multi-line human-readable report for CLI display.

    Args:
        backend: Requested backend name (``"default"``, ``"mfa"``, ``"auto"``).
        detected: Whether ``mlx_mfa`` was found on import.
        reason: Version string on success, error reason on failure.
        seq_len_threshold: Threshold used in ``auto`` mode.
    """
    lines: list[str] = []
    lines.append("Attention Backend Report")
    lines.append("========================")
    lines.append(f"  requested      : {backend}")
    lines.append(f"  mlx-mfa found  : {'yes' if detected else 'no'}")
    lines.append(f"  detail         : {reason}")

    if backend == "default":
        lines.append("  active         : default (mlx_lm SDPA)")
    elif backend == "mfa":
        if detected:
            lines.append("  active         : mfa (always-on)")
        else:
            lines.append("  active         : default (fallback — mlx-mfa unavailable)")
    elif backend == "auto":
        if detected:
            lines.append(f"  active         : auto (mfa when seq_len >= {seq_len_threshold})")
        else:
            lines.append("  active         : default (auto fallback — mlx-mfa unavailable)")
    else:
        lines.append(f"  active         : unknown backend '{backend}'")

    if not detected:
        lines.append("")
        lines.append("  Hint: install with `pip install mlx-mfa` to enable the")
        lines.append("        metal-flash-attention kernel (+8-15% decode on long seqs).")
    return "\n".join(lines)


def benchmark_attention_backend(
    backend: str,
    seq_lens: list[int] | None = None,
) -> dict:
    """Benchmark an attention backend across sequence lengths (stub).

    This is a placeholder for future work. Actual benchmarking requires a
    loaded model, a calibration prompt, and synchronized GPU timing — which
    is out of scope for the pure detection helper here. The stub exists so
    downstream CLI plumbing and tests can target a stable signature.

    Args:
        backend: Backend name to benchmark.
        seq_lens: Optional list of sequence lengths to sweep.

    Returns:
        A dict describing the (not yet implemented) benchmark request.
    """
    return {
        "backend": backend,
        "seq_lens": seq_lens or [],
        "note": "actual benchmarking requires a loaded model — not yet implemented",
    }

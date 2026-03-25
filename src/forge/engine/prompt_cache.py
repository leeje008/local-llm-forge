"""Prompt caching — pre-compute KV cache for repeated system prompts."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheResult:
    cache_path: Path
    success: bool
    size_mb: float = 0.0
    error: str | None = None


def cache_prompt(
    model_path: str | Path,
    prompt: str,
    cache_path: Path | None = None,
) -> CacheResult:
    """Pre-compute and save KV cache for a prompt using mlx-lm.

    The saved cache can be loaded later for zero-latency prompt reuse.
    """
    model_path = Path(model_path)
    if cache_path is None:
        cache_dir = model_path / "prompt_caches"
        cache_dir.mkdir(exist_ok=True)
        # Use hash of prompt for filename
        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"cache_{prompt_hash}.safetensors"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "cache_prompt",
        "--model", str(model_path),
        "--prompt", prompt,
        "--prompt-cache-file", str(cache_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return CacheResult(
                cache_path=cache_path, success=False,
                error=(result.stderr or result.stdout)[:300],
            )

        size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_path.exists() else 0
        return CacheResult(cache_path=cache_path, success=True, size_mb=size_mb)

    except subprocess.TimeoutExpired:
        return CacheResult(cache_path=cache_path, success=False, error="Timed out (5min)")
    except Exception as e:
        return CacheResult(cache_path=cache_path, success=False, error=str(e))


def generate_with_cache(
    model_path: str | Path,
    prompt: str,
    cache_path: Path,
    max_tokens: int = 256,
) -> str:
    """Generate text using a pre-computed prompt cache."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", str(model_path),
        "--prompt", prompt,
        "--prompt-cache-file", str(cache_path),
        "--max-tokens", str(max_tokens),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout


def list_caches(model_path: str | Path) -> list[dict]:
    """List available prompt caches for a model."""
    cache_dir = Path(model_path) / "prompt_caches"
    if not cache_dir.exists():
        return []

    caches = []
    for f in cache_dir.glob("*.safetensors"):
        size_mb = f.stat().st_size / (1024 * 1024)
        caches.append({"path": str(f), "name": f.stem, "size_mb": round(size_mb, 1)})
    return caches

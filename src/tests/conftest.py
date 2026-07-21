"""Shared fixtures for pure-logic characterization tests.

All fixtures build dataclasses directly — no hardware detection, no HuggingFace
lookups, no mlx. This keeps the suite fully offline and deterministic.
"""

from __future__ import annotations

import pytest

from forge.analyzer.hardware_profiler import HardwareProfile
from forge.analyzer.model_inspector import ModelProfile


@pytest.fixture
def m4_pro() -> HardwareProfile:
    """M4 Pro 48GB — usable_memory_gb == 38.0."""
    return HardwareProfile(
        chip="Apple M4 Pro",
        cpu_cores_physical=14,
        cpu_cores_logical=14,
        gpu_cores=20,
        ane_tops=38.0,
        total_memory_gb=48.0,
        memory_bandwidth_gbs=273.0,
        disk_available_gb=500.0,
        metal_version=4,
        os_version="26.3.1",
        has_mlx=True,
        mlx_version="0.31.0",
        has_ollama=True,
        python_version="3.14.0",
    )


@pytest.fixture
def small_hw() -> HardwareProfile:
    """16GB machine — usable_memory_gb == 6.0, forces tight-memory paths."""
    return HardwareProfile(
        chip="Apple M1",
        cpu_cores_physical=8,
        cpu_cores_logical=8,
        gpu_cores=8,
        ane_tops=11.0,
        total_memory_gb=16.0,
        memory_bandwidth_gbs=68.0,
        disk_available_gb=100.0,
        metal_version=3,
        os_version="26.0.0",
        has_mlx=True,
        mlx_version="0.31.0",
        has_ollama=False,
        python_version="3.14.0",
    )


@pytest.fixture
def dense_7b() -> ModelProfile:
    """Qwen2.5-7B-like dense GQA model."""
    return ModelProfile(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        architecture="qwen2",
        model_type="dense",
        architecture_family="transformer",
        total_params_b=7.6,
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_kv_heads=4,
        attention_type="GQA",
        head_dim=128,
        vocab_size=152064,
        max_context=32768,
        torch_dtype="bfloat16",
    )


@pytest.fixture
def dense_72b() -> ModelProfile:
    """Qwen2.5-72B-like dense model — too large for 48GB in FP16."""
    return ModelProfile(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        architecture="qwen2",
        model_type="dense",
        architecture_family="transformer",
        total_params_b=72.7,
        num_layers=80,
        hidden_size=8192,
        intermediate_size=29568,
        num_attention_heads=64,
        num_kv_heads=8,
        attention_type="GQA",
        head_dim=128,
        vocab_size=152064,
        max_context=32768,
        torch_dtype="bfloat16",
    )


@pytest.fixture
def moe_8x7b() -> ModelProfile:
    """Mixtral-8x7B-like MoE model."""
    return ModelProfile(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        architecture="mixtral",
        model_type="moe",
        architecture_family="transformer",
        total_params_b=46.7,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        attention_type="GQA",
        head_dim=128,
        vocab_size=32000,
        max_context=32768,
        num_experts=8,
        num_active_experts=2,
        torch_dtype="bfloat16",
    )


@pytest.fixture
def tiny_model() -> ModelProfile:
    """Degenerate profile — all-zero dimensions, for boundary cases."""
    return ModelProfile(model_id="test/empty")

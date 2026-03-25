"""Local hardware detection for Apple Silicon Macs."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """Detected local hardware specification."""

    chip: str = "Unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    gpu_cores: int = 0
    ane_tops: float = 0.0
    total_memory_gb: float = 0.0
    memory_bandwidth_gbs: float = 0.0
    disk_available_gb: float = 0.0
    metal_version: int = 0
    os_version: str = ""
    has_mlx: bool = False
    mlx_version: str | None = None
    has_ollama: bool = False
    python_version: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_memory_gb(self) -> float:
        """Memory available for model loading (total - OS - framework)."""
        os_overhead = 8.0
        framework_buffer = 2.0
        return max(self.total_memory_gb - os_overhead - framework_buffer, 0.0)


# Known Apple Silicon specs: (gpu_cores → bandwidth GB/s, ANE TOPS)
_CHIP_SPECS: dict[str, dict[str, dict[str, float]]] = {
    "M1": {
        "base": {"bw": 68.25, "ane": 11.0},
        "Pro": {"bw": 200.0, "ane": 11.0},
        "Max": {"bw": 400.0, "ane": 11.0},
        "Ultra": {"bw": 800.0, "ane": 22.0},
    },
    "M2": {
        "base": {"bw": 100.0, "ane": 15.8},
        "Pro": {"bw": 200.0, "ane": 15.8},
        "Max": {"bw": 400.0, "ane": 15.8},
        "Ultra": {"bw": 800.0, "ane": 31.6},
    },
    "M3": {
        "base": {"bw": 100.0, "ane": 18.0},
        "Pro": {"bw": 150.0, "ane": 18.0},
        "Max": {"bw": 400.0, "ane": 18.0},
        "Ultra": {"bw": 800.0, "ane": 36.0},
    },
    "M4": {
        "base": {"bw": 120.0, "ane": 38.0},
        "Pro": {"bw": 273.0, "ane": 38.0},
        "Max": {"bw": 546.0, "ane": 38.0},
        "Ultra": {"bw": 819.2, "ane": 76.0},
    },
}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key])


def _parse_chip(brand: str) -> tuple[str, str]:
    """Parse chip family and variant from brand string like 'Apple M4 Pro'."""
    parts = brand.replace("Apple ", "").split()
    if not parts:
        return "Unknown", "base"
    family = parts[0]  # "M4"
    variant = parts[1] if len(parts) > 1 else "base"
    return family, variant


def _detect_gpu_cores() -> int:
    out = _run(["system_profiler", "SPDisplaysDataType"])
    for line in out.splitlines():
        stripped = line.strip()
        if "Total Number of Cores" in stripped or "Cores" in stripped:
            # e.g. "Total Number of Cores: 20"
            parts = stripped.split(":")
            if len(parts) == 2:
                try:
                    return int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return 0


def _detect_metal_version() -> int:
    out = _run(["system_profiler", "SPDisplaysDataType"])
    for line in out.splitlines():
        if "Metal" in line:
            # e.g. "Metal Support: Metal 4" or "Metal Family: ..."
            for word in line.split():
                try:
                    v = int(word)
                    if 1 <= v <= 10:
                        return v
                except ValueError:
                    continue
    return 0


def _detect_disk_available() -> float:
    out = _run(["df", "-g", "/"])
    lines = out.splitlines()
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 4:
            try:
                return float(parts[3])
            except ValueError:
                pass
    return 0.0


def _detect_mlx() -> tuple[bool, str | None]:
    try:
        import mlx  # type: ignore[import-untyped]

        return True, getattr(mlx, "__version__", "unknown")
    except ImportError:
        return False, None


def _detect_python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def detect() -> HardwareProfile:
    """Detect local hardware and return a HardwareProfile."""
    profile = HardwareProfile()
    warnings: list[str] = []

    # CPU
    brand = _sysctl("machdep.cpu.brand_string")
    profile.chip = brand or "Unknown"
    family, variant = _parse_chip(brand)

    try:
        profile.cpu_cores_physical = int(_sysctl("hw.physicalcpu") or "0")
    except ValueError:
        pass
    try:
        profile.cpu_cores_logical = int(_sysctl("hw.ncpu") or "0")
    except ValueError:
        pass

    # GPU
    profile.gpu_cores = _detect_gpu_cores()

    # Memory
    try:
        mem_bytes = int(_sysctl("hw.memsize") or "0")
        profile.total_memory_gb = round(mem_bytes / (1024**3), 1)
    except ValueError:
        pass

    # Bandwidth & ANE from known specs
    specs = _CHIP_SPECS.get(family, {}).get(variant, {})
    profile.memory_bandwidth_gbs = specs.get("bw", 0.0)
    profile.ane_tops = specs.get("ane", 0.0)
    if not specs:
        warnings.append(f"Unknown chip '{brand}' — bandwidth/ANE estimates unavailable")

    # Metal
    profile.metal_version = _detect_metal_version()

    # OS
    profile.os_version = _run(["sw_vers", "-productVersion"])

    # Disk
    profile.disk_available_gb = _detect_disk_available()

    # MLX
    profile.has_mlx, profile.mlx_version = _detect_mlx()

    # Ollama
    profile.has_ollama = shutil.which("ollama") is not None

    # Python
    profile.python_version = _detect_python_version()

    profile.warnings = warnings
    return profile


def format_report(hw: HardwareProfile) -> str:
    """Format a human-readable hardware report."""
    lines = [
        "Hardware Profile",
        "=" * 50,
        f"  Chip:            {hw.chip}",
        f"  CPU Cores:       {hw.cpu_cores_physical} physical / {hw.cpu_cores_logical} logical",
        f"  GPU Cores:       {hw.gpu_cores}",
        f"  ANE:             {hw.ane_tops} TOPS",
        f"  Memory:          {hw.total_memory_gb} GB (usable: {hw.usable_memory_gb:.1f} GB)",
        f"  Bandwidth:       {hw.memory_bandwidth_gbs} GB/s",
        f"  Metal:           {hw.metal_version}",
        f"  Disk Available:  {hw.disk_available_gb:.0f} GB",
        f"  OS:              macOS {hw.os_version}",
        f"  Python:          {hw.python_version}",
        f"  MLX:             {'v' + hw.mlx_version if hw.has_mlx else 'Not installed'}",
        f"  Ollama:          {'Installed' if hw.has_ollama else 'Not installed'}",
    ]
    if hw.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in hw.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)

"""Hardware detection and auto-configuration."""

from dataclasses import dataclass, field


@dataclass
class HardwareInfo:
    """Fully resolved hardware profile."""
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cuda_version: str = ""
    cpu_cores: int = 0
    ram_total_mb: int = 0

    @property
    def recommended_tier(self) -> str:
        """Suggest a model tier based on VRAM."""
        if not self.gpu_available:
            return "mobile"
        if self.vram_total_mb >= 16000:
            return "server"
        if self.vram_total_mb >= 8000:
            return "medium"
        return "mobile"

    @property
    def supports_fp16(self) -> bool:
        return self.gpu_available and self.vram_total_mb >= 4000


def detect() -> HardwareInfo:
    info = HardwareInfo()
    _detect_cpu(info)
    _detect_gpu(info)
    return info


def _detect_cpu(info: HardwareInfo) -> None:
    import os
    info.cpu_cores = os.cpu_count() or 0

    try:
        import psutil
        info.ram_total_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass


def _detect_gpu(info: HardwareInfo) -> None:
    try:
        import paddle
        if not paddle.is_compiled_with_cuda():
            return
        count = paddle.device.cuda.device_count()
        if count == 0:
            return
        info.gpu_available = True
        info.gpu_count = count
        info.gpu_name = paddle.device.cuda.get_device_name(0)
        caps = paddle.device.cuda.get_device_capability(0)
        info.cuda_version = f"{caps[0]}.{caps[1]}" if caps else ""
        props = paddle.device.cuda.get_device_properties(0)
        info.vram_total_mb = int(props.total_memory / (1024 * 1024))
    except Exception:
        pass

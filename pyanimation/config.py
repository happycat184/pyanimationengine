from dataclasses import dataclass
import os


@dataclass
class RenderConfig:
    """Output and world-coordinate settings for a render.

    ``antialias`` is the supersampling factor. Increasing it improves edge
    coverage at the cost of roughly antialias**2 raster work per frame.
    ``device`` defaults to automatic GPU selection with CPU fallback.
    """

    width: int = 1280
    height: int = 720
    fps: int = 30
    antialias: int = 2
    frame_width: float = 14.2222222222
    background_color: str = "#10131c"
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    font_path: str | None = None
    workers: int = 16
    device: str = "auto"

    def __post_init__(self) -> None:
        for field_name in ("width", "height", "fps", "antialias", "workers"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for H.264 yuv420p output")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.antialias < 1:
            raise ValueError("antialias must be at least 1")
        if self.workers < 0:
            raise ValueError("workers must be zero (auto) or a positive integer")
        if self.device not in {"auto", "gpu", "cpu"}:
            raise ValueError("device must be 'auto', 'gpu', or 'cpu'")
        if self.frame_width <= 0:
            raise ValueError("frame_width must be positive")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

    @property
    def frame_height(self) -> float:
        return self.frame_width * self.height / self.width

    @property
    def raster_width(self) -> int:
        return self.width * self.antialias

    @property
    def raster_height(self) -> int:
        return self.height * self.antialias

    @property
    def pixels_per_unit(self) -> float:
        return self.raster_width / self.frame_width

    @property
    def effective_workers(self) -> int:
        if self.workers:
            return self.workers
        return max(1, min(4, os.cpu_count() or 1))

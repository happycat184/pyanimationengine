from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import time
from threading import local
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import RenderConfig
from .fonts import load_pillow_font, resolve_font_file
from .mobject import Mobject, color_rgb
from .geometry3d import Camera3D, Mesh3D, RenderMesh


@lru_cache(maxsize=128)
def _decode_text_bitmap(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("L").copy()


class ProgressReporter:
    """Print throttled, frame-based progress for each render pipeline phase."""

    def __init__(self, total_frames: int, stream=None, interval_percent: int = 10):
        self.total_frames = max(1, int(total_frames))
        self.stream = stream or sys.stdout
        self.interval_percent = max(1, min(100, int(interval_percent)))
        self._last_bucket: dict[str, int] = {}
        self.has_3d = False

    def update(self, stage: str, processed: int, total: int, info: str = "") -> None:
        if stage.startswith("3D "):
            self.has_3d = True
        total = max(1, int(total))
        processed = max(0, min(int(processed), total))
        percent = processed * 100 // total
        bucket = percent // self.interval_percent
        last_bucket = self._last_bucket.get(stage)
        should_report = last_bucket is None or bucket > last_bucket
        if not should_report:
            return
        self._last_bucket[stage] = bucket
        suffix = f" {info}" if info else ""
        print(f"[INFO {stage}] {processed}/{total}{suffix}", file=self.stream, flush=True)


@dataclass
class RenderObject:
    paths: list[np.ndarray]
    closed: list[bool]
    fill_color: tuple[int, int, int]
    stroke_color: tuple[int, int, int]
    fill_opacity: float
    stroke_opacity: float
    stroke_width: float
    z_index: int
    insertion_order: int
    reveal: float = 1.0
    text: str | None = None
    text_font: str | None = None
    text_size: int | None = None
    position: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    text_bitmap: bytes | None = None
    mesh: RenderMesh | None = None


class Converter:
    """Converts public Mobjects to backend-neutral pixel-space objects."""

    def __init__(self, config: RenderConfig, camera: Camera3D | None = None):
        self.config = config
        self.camera = camera or Camera3D()
        self._thread_state = local()

    def _rasterize_text(self, mobject: Mobject, text_size: int) -> bytes:
        text = getattr(mobject, "text", "")
        if not text:
            return b""
        requested = getattr(mobject, "font", None) or self.config.font_path or "DejaVuSans.ttf"
        resolved = resolve_font_file(requested)
        key = (resolved, text_size)
        if not hasattr(self._thread_state, "font_cache"):
            self._thread_state.font_cache = {}
            self._thread_state.text_bitmap_cache = {}
        font_cache = self._thread_state.font_cache
        bitmap_cache = self._thread_state.text_bitmap_cache
        bitmap_key = (str(text), *key)
        cached_bitmap = bitmap_cache.get(bitmap_key)
        if cached_bitmap is not None:
            return cached_bitmap
        font = font_cache.get(key)
        if font is None:
            font = load_pillow_font(
                requested,
                text_size,
                required=bool(getattr(mobject, "font", None) or self.config.font_path),
            )
            font_cache[key] = font
        left, top, right, bottom = font.getbbox(text)
        width, height = max(1, right - left + 4), max(1, bottom - top + 4)
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).text((2 - left, 2 - top), text, font=font, fill=255)
        output = BytesIO()
        mask.save(output, format="PNG", optimize=True)
        bitmap = output.getvalue()
        bitmap_cache[bitmap_key] = bitmap
        return bitmap

    def _to_pixel(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        c = self.config
        ppu = c.pixels_per_unit
        return [np.column_stack((c.raster_width / 2 + p[:, 0] * ppu, c.raster_height / 2 - p[:, 1] * ppu)) for p in paths]

    def convert(self, mobject: Mobject, insertion_order=0) -> RenderObject:
        compile_tex = getattr(mobject, "_ensure_tex_compiled", None)
        if compile_tex is not None:
            compile_tex()
        c = self.config
        ppu = c.pixels_per_unit
        world_paths = mobject.get_world_paths()
        fill = color_rgb(mobject.fill_color if mobject.fill_color is not None else mobject.color)
        stroke = color_rgb(mobject.stroke_color if mobject.stroke_color is not None else mobject.color)
        text = getattr(mobject, "text", None)
        text_size = None
        text_bitmap = None
        position = mobject.position
        if text is not None:
            text_size = max(1, round(getattr(mobject, "font_size", 0.55) * ppu * max(abs(mobject.scale_x), abs(mobject.scale_y))))
            text_bitmap = self._rasterize_text(mobject, text_size)
        return RenderObject(
            paths=self._to_pixel(world_paths),
            closed=list(mobject.closed),
            fill_color=fill,
            stroke_color=stroke,
            fill_opacity=max(0.0, min(1.0, mobject.fill_opacity * mobject.opacity)),
            stroke_opacity=max(0.0, min(1.0, mobject.opacity)),
            stroke_width=max(0.0, mobject.stroke_width * c.antialias),
            z_index=mobject.z_index,
            insertion_order=insertion_order,
            reveal=max(0.0, min(1.0, mobject.reveal)),
            text=text,
            text_font=getattr(mobject, "font", None),
            text_size=text_size,
            position=(c.raster_width / 2 + position[0] * ppu, c.raster_height / 2 - position[1] * ppu),
            rotation=mobject.rotation,
            text_bitmap=text_bitmap,
            mesh=mobject.to_render_mesh(self.camera) if isinstance(mobject, Mesh3D) else None,
        )


class FrameBuffer:
    """RGB storage for one high-resolution frame."""

    def __init__(self, width: int, height: int, background=(0, 0, 0)):
        self.width, self.height = int(width), int(height)
        self.pixels = np.empty((height, width, 3), dtype=np.uint8)
        self.clear(background)

    def clear(self, color) -> None:
        self.pixels[:] = color

    def composite(self, x0: int, y0: int, mask: np.ndarray, color, opacity=1.0) -> None:
        if opacity <= 0 or mask.size == 0:
            return
        h, w = mask.shape
        x1, y1 = x0 + w, y0 + h
        left, top = max(0, x0), max(0, y0)
        right, bottom = min(self.width, x1), min(self.height, y1)
        if left >= right or top >= bottom:
            return
        m = mask[top - y0 : bottom - y0, left - x0 : right - x0].astype(np.float32) * float(opacity)
        src = np.asarray(color, dtype=np.float32).reshape((1, 1, 3))
        dst = self.pixels[top:bottom, left:right].astype(np.float32)
        self.pixels[top:bottom, left:right] = np.clip(src * m[..., None] + dst * (1.0 - m[..., None]), 0, 255).astype(np.uint8)


class FrameGenerator:
    """Rasterizes paths with scan-line filling and segment-distance strokes."""

    def __init__(self, config: RenderConfig):
        self.config = config
        self.background = color_rgb(config.background_color)
        self._font_cache = {}

    def _fill_paths(self, buffer: FrameBuffer, paths: list[np.ndarray], color, opacity) -> None:
        paths = [p for p in paths if len(p) >= 3]
        if not paths:
            return
        all_points = np.concatenate(paths)
        y_min = max(0, int(math.floor(all_points[:, 1].min())))
        y_max = min(buffer.height - 1, int(math.ceil(all_points[:, 1].max())))
        x_min = max(0, int(math.floor(all_points[:, 0].min())))
        x_max = min(buffer.width - 1, int(math.ceil(all_points[:, 0].max())))
        if y_min > y_max or x_min > x_max:
            return
        xs = np.arange(x_min, x_max + 1) + 0.5
        edge_starts = np.concatenate(paths, axis=0)
        edge_ends = np.concatenate([np.roll(path, -1, axis=0) for path in paths], axis=0)
        edge_dy = edge_ends[:, 1] - edge_starts[:, 1]
        for y in range(y_min, y_max + 1):
            scan_y = y + 0.5
            crossing = ((edge_starts[:, 1] <= scan_y) & (edge_ends[:, 1] > scan_y)) | ((edge_ends[:, 1] <= scan_y) & (edge_starts[:, 1] > scan_y))
            if not np.any(crossing):
                continue
            intersections = edge_starts[crossing, 0] + (scan_y - edge_starts[crossing, 1]) * (edge_ends[crossing, 0] - edge_starts[crossing, 0]) / edge_dy[crossing]
            intersections.sort()
            mask = np.zeros((1, len(xs)), dtype=np.float32)
            for left, right in zip(intersections[0::2], intersections[1::2]):
                mask[0, (xs >= left) & (xs < right)] = 1.0
            buffer.composite(x_min, y, mask, color, opacity)

    def _stroke_segment(self, buffer: FrameBuffer, a: np.ndarray, b: np.ndarray, color, opacity, width: float) -> None:
        radius = max(0.5, width / 2)
        x0 = max(0, int(math.floor(min(a[0], b[0]) - radius)))
        x1 = min(buffer.width, int(math.ceil(max(a[0], b[0]) + radius + 1)))
        y0 = max(0, int(math.floor(min(a[1], b[1]) - radius)))
        y1 = min(buffer.height, int(math.ceil(max(a[1], b[1]) + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        px, py = xx + 0.5, yy + 0.5
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = dx * dx + dy * dy
        t = np.zeros_like(px) if denom == 0 else np.clip(((px - a[0]) * dx + (py - a[1]) * dy) / denom, 0, 1)
        distance = np.sqrt((px - (a[0] + t * dx)) ** 2 + (py - (a[1] + t * dy)) ** 2)
        mask = np.clip(radius + 0.5 - distance, 0, 1).astype(np.float32)
        buffer.composite(x0, y0, mask, color, opacity)

    @staticmethod
    def _trim(points: np.ndarray, fraction: float) -> np.ndarray:
        if fraction >= 1 or len(points) < 2:
            return points
        if fraction <= 0:
            return points[:1]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        target = float(lengths.sum()) * fraction
        out = [points[0]]
        total = 0.0
        for i, length in enumerate(lengths):
            if total + length >= target:
                portion = 0 if length == 0 else (target - total) / length
                out.append(points[i] * (1 - portion) + points[i + 1] * portion)
                break
            out.append(points[i + 1])
            total += length
        return np.asarray(out)

    def _draw_text(self, buffer: FrameBuffer, obj: RenderObject) -> None:
        if obj.fill_opacity <= 0:
            return
        if obj.text_bitmap:
            mask_image = _decode_text_bitmap(obj.text_bitmap)
            if obj.rotation:
                mask_image = mask_image.rotate(math.degrees(obj.rotation), expand=True, resample=Image.Resampling.BICUBIC)
            x = round(obj.position[0] - mask_image.width / 2)
            y = round(obj.position[1] - mask_image.height / 2)
            mask = np.asarray(mask_image, dtype=np.float32) / 255.0
            buffer.composite(x, y, mask, obj.fill_color, obj.fill_opacity)
            return
        if not obj.text or not obj.text_size:
            return
        requested_font = obj.text_font or self.config.font_path or "DejaVuSans.ttf"
        resolved_font = resolve_font_file(requested_font)
        font_key = (resolved_font, obj.text_size)
        font = self._font_cache.get(font_key)
        if font is None:
            font = load_pillow_font(
                requested_font,
                obj.text_size,
                required=bool(obj.text_font or self.config.font_path),
            )
            self._font_cache[font_key] = font
        left, top, right, bottom = font.getbbox(obj.text)
        w, h = max(1, right - left + 4), max(1, bottom - top + 4)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.text((2 - left, 2 - top), obj.text, font=font, fill=(*obj.fill_color, round(255 * obj.fill_opacity)))
        if obj.rotation:
            layer = layer.rotate(math.degrees(obj.rotation), expand=True, resample=Image.Resampling.BICUBIC)
        x = round(obj.position[0] - layer.width / 2)
        y = round(obj.position[1] - layer.height / 2)
        rgba = np.asarray(layer, dtype=np.uint8)
        mask = rgba[:, :, 3].astype(np.float32) / 255.0
        # Color is already in the RGBA source; composite channel-aware here.
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(buffer.width, x + layer.width), min(buffer.height, y + layer.height)
        if x0 >= x1 or y0 >= y1:
            return
        src = rgba[y0 - y : y1 - y, x0 - x : x1 - x, :3].astype(np.float32)
        alpha = mask[y0 - y : y1 - y, x0 - x : x1 - x, None]
        dst = buffer.pixels[y0:y1, x0:x1].astype(np.float32)
        buffer.pixels[y0:y1, x0:x1] = np.clip(src * alpha + dst * (1 - alpha), 0, 255).astype(np.uint8)

    def generate(
        self,
        objects: list[RenderObject],
        buffer: FrameBuffer | None = None,
        *,
        presorted: bool = False,
    ) -> FrameBuffer:
        c = self.config
        if buffer is None:
            buffer = FrameBuffer(c.raster_width, c.raster_height, self.background)
        ordered_objects = objects if presorted else sorted(
            objects, key=lambda item: (item.z_index, item.insertion_order)
        )
        for obj in ordered_objects:
            if obj.fill_opacity <= 0 and obj.stroke_opacity <= 0 and obj.text is None:
                continue
            if obj.text is not None:
                from dataclasses import replace
                self._draw_text(buffer, replace(obj, fill_opacity=obj.fill_opacity * obj.reveal))
                continue
            fill_paths = [path for path, closed in zip(obj.paths, obj.closed) if closed]
            if fill_paths and obj.fill_opacity > 0 and obj.reveal >= 1:
                self._fill_paths(buffer, fill_paths, obj.fill_color, obj.fill_opacity)
            for points, closed in zip(obj.paths, obj.closed):
                if len(points) == 0:
                    continue
                stroke_points = self._trim(points, obj.reveal)
                if obj.stroke_width > 0 and obj.stroke_opacity > 0 and len(stroke_points) >= 2:
                    for a, b in zip(stroke_points[:-1], stroke_points[1:]):
                        self._stroke_segment(buffer, a, b, obj.stroke_color, obj.stroke_opacity, obj.stroke_width)
                    if closed and obj.reveal >= 1:
                        self._stroke_segment(buffer, points[-1], points[0], obj.stroke_color, obj.stroke_opacity, obj.stroke_width)
        return buffer


class Renderer:
    """Orders cached objects and dispatches frames to GPU or CPU rasterizers."""

    def __init__(self, config: RenderConfig):
        from .renderer3d import Renderer3D

        self.config = config
        self.converter = Converter(config)
        self.frame_generator = FrameGenerator(config)
        self.renderer3d = Renderer3D(config)
        self.gpu_generator = None
        if config.device != "cpu":
            try:
                from .gpu import GPUFrameGenerator

                self.gpu_generator = GPUFrameGenerator(config)
                info = self.gpu_generator.adapter_info
                gpu_name = info.get("device", "unknown GPU")
                vendor = info.get("vendor", "unknown vendor")
                backend = info.get("backend_type", "WebGPU")
                print(f"[INFO GPU] device={vendor} {gpu_name} backend={backend}", flush=True)
            except Exception as exc:
                if config.device == "gpu":
                    raise RuntimeError(f"GPU rendering was requested but no usable GPU backend is available: {exc}") from exc
                print(f"[INFO GPU] hardware unavailable; using CPU ({exc})", flush=True)

    def _disable_gpu(self, error: Exception) -> None:
        if self.config.device == "gpu":
            raise RuntimeError(f"GPU frame rendering failed: {error}") from error
        print(f"[INFO GPU] frame failed; switching to CPU ({error})", flush=True)
        self.gpu_generator = None

    def _compose(self, objects, generator=None, use_gpu=True):
        from itertools import groupby

        generator = generator or self.frame_generator
        gpu = self.gpu_generator if use_gpu else None
        objects = sorted(objects, key=lambda obj: (obj.z_index, obj.insertion_order))
        if not any(obj.mesh is not None for obj in objects):
            return (gpu or generator).generate(objects, presorted=True)
        frame = FrameBuffer(self.config.raster_width, self.config.raster_height, color_rgb(self.config.background_color))
        for _, layer in groupby(objects, key=lambda obj: obj.z_index):
            layer = list(layer)
            meshes = [obj for obj in layer if obj.mesh is not None]
            overlays = [obj for obj in layer if obj.mesh is None]
            if meshes:
                frame = self.renderer3d.generate(meshes, frame, gpu=gpu)
            if overlays:
                frame = (gpu or generator).generate(overlays, frame, presorted=True)
        return frame

    def _generate(self, objects, generator=None, use_gpu=True):
        try:
            return self._compose(objects, generator, use_gpu)
        except Exception as exc:
            if not use_gpu or self.gpu_generator is None:
                raise
            if any(obj.mesh is not None for obj in objects):
                print(f"[INFO 3D Device] GPU frame failed ({exc})", flush=True)
            self._disable_gpu(exc)
            return self._compose(objects, generator, False)

    def render(self, mobjects: list[Mobject], progress: ProgressReporter | None = None, frame_index: int = 1, total_frames: int = 1) -> FrameBuffer:
        converted = [self.converter.convert(mob, i) for i, mob in enumerate(mobjects)]
        if progress is not None:
            progress.update("Converting", frame_index, total_frames, f"objects={len(converted)}")
        converted.sort(key=lambda obj: (obj.z_index, obj.insertion_order))
        if progress is not None:
            progress.update("Calculating", frame_index, total_frames, f"ordered_objects={len(converted)}")
        meshes = sum(obj.mesh is not None for obj in converted)
        if meshes and progress is not None:
            progress.update("3D Calculating", frame_index, total_frames, f"meshes={meshes}")
        frame = self._generate(converted)
        if progress is not None:
            progress.update("In Rendering", frame_index, total_frames, f"pixels={self.config.width}x{self.config.height}")
            if meshes:
                progress.update("3D In Rendering", frame_index, total_frames, f"meshes={meshes}")
        return frame

    def render_cache(
        self,
        object_cache: str | Path,
        frame_cache: str | Path | None,
        progress: ProgressReporter | None = None,
        on_frame=None,
    ) -> int:
        """Render cached frames in order, optionally sending each frame to a viewer."""
        from .object_cache import FrameSequenceWriter, ObjectCacheReader
        from contextlib import nullcontext

        frame_count = 0
        with ObjectCacheReader(object_cache) as reader:
            expected = {
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "antialias": self.config.antialias,
                "frame_width": self.config.frame_width,
                "background_color": color_rgb(self.config.background_color),
            }
            for name, value in expected.items():
                cached = reader.metadata[name]
                if name == "frame_width":
                    matches = math.isclose(float(cached), float(value), rel_tol=1e-7, abs_tol=1e-7)
                else:
                    matches = cached == value
                if not matches:
                    raise ValueError(f".objects cache setting mismatch for {name}")
            frame_context = (
                FrameSequenceWriter(frame_cache)
                if frame_cache is not None
                else nullcontext(None)
            )
            with frame_context as frames:
                worker_count = self.config.effective_workers

                def calculating(frame_index, objects):
                    meshes = [obj.mesh for obj in objects if obj.mesh is not None]
                    if progress is not None:
                        progress.update("Calculating", frame_index + 1, reader.frame_count, f"objects={len(objects)}")
                        if meshes:
                            progress.update("3D Calculating", frame_index + 1, reader.frame_count,
                                            f"meshes={len(meshes)} triangles={sum(len(m.faces) for m in meshes)}")
                    return bool(meshes)

                def write_result(frame, has_3d):
                    nonlocal frame_count
                    if frames is not None:
                        frames.write(frame.pixels)
                    frame_count += 1
                    if on_frame is not None and on_frame(frame) is False:
                        raise OnlineWindowClosed
                    if progress is not None:
                        info = f"pixels={self.config.width}x{self.config.height} device={'GPU' if self.gpu_generator else 'CPU'}"
                        progress.update("In Rendering", frame_count, reader.frame_count, info)
                        if has_3d:
                            progress.update("3D In Rendering", frame_count, reader.frame_count, info)

                if self.gpu_generator is not None or worker_count <= 1:
                    for frame_index, objects in reader:
                        has_3d = calculating(frame_index, objects)
                        write_result(self._generate(objects), has_3d)
                else:
                    thread_state = local()

                    def render_one(objects):
                        generator = getattr(thread_state, "generator", None)
                        if generator is None:
                            generator = FrameGenerator(self.config)
                            thread_state.generator = generator
                        return self._generate(objects, generator, use_gpu=False)

                    pending = deque()
                    max_pending = worker_count

                    def write_next_result():
                        has_3d, future = pending.popleft()
                        write_result(future.result(), has_3d)

                    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pyanimation-render") as pool:
                        for frame_index, objects in reader:
                            has_3d = calculating(frame_index, objects)
                            pending.append((has_3d, pool.submit(render_one, objects)))
                            if len(pending) >= max_pending:
                                write_next_result()
                        while pending:
                            write_next_result()
        return frame_count


class Drawer:
    """Batch-encodes a completed RGB frame sequence as an H.264 MP4."""

    def __init__(self, output: str | Path, config: RenderConfig):
        self.output = Path(output)
        self.config = config
        self.process: subprocess.Popen | None = None
        self.frame_count = 0

    def open(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required. Install the project dependencies or add ffmpeg to PATH.")
        c = self.config
        command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{c.width}x{c.height}", "-r", str(c.fps), "-i", "-", "-an", "-c:v", c.codec, "-preset", c.preset, "-crf", str(c.crf), "-pix_fmt", "yuv420p", str(self.output)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame: FrameBuffer) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Drawer.open() must be called first")
        image = Image.fromarray(frame.pixels, mode="RGB")
        if image.size != (self.config.width, self.config.height):
            image = image.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)
        self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        self.frame_count += 1

    def write_frames(self, frame_path: str | Path, frame_count: int, progress: ProgressReporter | None = None) -> None:
        """Encode a completed sequence in a single MP4 writer session."""
        self.open()
        c = self.config
        raw_size = c.raster_width * c.raster_height * 3
        try:
            with Path(frame_path).open("rb") as stream:
                for index in range(frame_count):
                    raw = stream.read(raw_size)
                    if len(raw) != raw_size:
                        raise ValueError(f"temporary frame sequence ended at frame {index + 1}/{frame_count}")
                    pixels = np.frombuffer(raw, dtype=np.uint8).reshape((c.raster_height, c.raster_width, 3))
                    image = Image.fromarray(pixels, mode="RGB")
                    if image.size != (c.width, c.height):
                        image = image.resize((c.width, c.height), Image.Resampling.LANCZOS)
                    if self.process is None or self.process.stdin is None:
                        raise RuntimeError("FFmpeg writer is not available")
                    self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
                    self.frame_count += 1
                    if progress is not None:
                        progress.update("Writing", self.frame_count, frame_count, "mp4_frame")
                        if progress.has_3d:
                            progress.update("3D Writing", self.frame_count, frame_count, "mixed_2d_3d_mp4_frame")
                if stream.read(1):
                    raise ValueError("temporary frame sequence contains trailing data")
        finally:
            self.close()

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        code = self.process.wait()
        error = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        self.process = None
        if code:
            raise RuntimeError(f"FFmpeg failed ({code}): {error.strip()}")


class OnlineWindowClosed(Exception):
    """Raised when the user closes the live preview to stop rendering."""


class OnlineDrawer:
    """Present rendered RGB frames in a Tk window without opening FFmpeg."""

    def __init__(self, config: RenderConfig, title: str = "PyAnimation live render"):
        self.config = config
        self.title = title
        self.root = None
        self.label = None
        self.photo = None
        self.running = False
        self.frame_count = 0
        self.next_frame_time = 0.0
        self.frame_interval = 1.0 / config.fps

    def open(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError("Online rendering needs Tkinter; install a Python build with Tcl/Tk support.") from exc

        try:
            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.configure(background="#10131c")
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.root.bind("<Escape>", lambda _event: self.close())
            self.label = ttk.Label(self.root)
            self.label.pack(fill="both", expand=True)
            self.root.update_idletasks()
            self.running = True
            self.next_frame_time = time.perf_counter()
            print(
                f"[INFO Online] window=open size={self.config.width}x{self.config.height} "
                f"fps={self.config.fps}",
                flush=True,
            )
            self.root.update()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not open the online render window: {exc}") from exc

    def poll(self) -> bool:
        if not self.running or self.root is None:
            return False
        try:
            self.root.update()
            return self.running
        except Exception:
            self.running = False
            return False

    def write(self, frame: FrameBuffer) -> bool:
        if not self.poll():
            return False

        deadline = max(self.next_frame_time, time.perf_counter())
        while self.running:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if not self.poll():
                return False
            time.sleep(min(remaining, 0.01))

        from PIL import Image, ImageTk

        image = Image.fromarray(frame.pixels, mode="RGB")
        if image.size != (self.config.width, self.config.height):
            image = image.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)
        screen_width = max(320, self.root.winfo_screenwidth() - 64)
        screen_height = max(240, self.root.winfo_screenheight() - 112)
        image.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.label.configure(image=self.photo)
        self.frame_count += 1
        self.next_frame_time = max(deadline, time.perf_counter()) + self.frame_interval
        return self.poll()

    def close(self) -> None:
        was_running = self.running
        self.running = False
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
        if was_running:
            print(f"[INFO Online] window=closed frames={self.frame_count}", flush=True)

    def wait_until_closed(self) -> None:
        """Keep the last frame visible until the user closes the window."""
        while self.poll():
            time.sleep(0.025)


def render_objects_cache(
    object_cache: str | Path,
    output_file: str | Path | None = None,
    config: RenderConfig | None = None,
    *,
    online: bool = False,
) -> Path | None:
    """Render a cache to MP4 or show it frame-by-frame without writing a video."""
    from .object_cache import ObjectCacheReader, render_config_from_cache

    object_cache = Path(object_cache)
    output = Path(output_file) if output_file is not None else None
    if not online and output is None:
        raise ValueError("output_file is required unless online=True")
    if config is None:
        config = render_config_from_cache(object_cache)
    with ObjectCacheReader(object_cache) as reader:
        expected_frames = reader.frame_count

    online_drawer = OnlineDrawer(config, title=f"{object_cache.name} — live render") if online else None
    frame_path = None
    try:
        renderer = Renderer(config)
        progress = ProgressReporter(expected_frames)
        if online_drawer is not None:
            online_drawer.open()
            rendered_frames = renderer.render_cache(object_cache, None, progress, on_frame=online_drawer.write)
        else:
            with tempfile.NamedTemporaryFile(prefix="pyanimation-", suffix=".frames", delete=False) as temp_file:
                frame_path = Path(temp_file.name)
            rendered_frames = renderer.render_cache(object_cache, frame_path, progress)
        if rendered_frames != expected_frames:
            raise RuntimeError(f"frame count mismatch: expected {expected_frames}, got {rendered_frames}")
        if online_drawer is not None:
            online_drawer.wait_until_closed()
        else:
            Drawer(output, config).write_frames(frame_path, rendered_frames, progress)
    except OnlineWindowClosed:
        print("[INFO Online] rendering=stopped by user", flush=True)
    finally:
        if online_drawer is not None:
            online_drawer.close()
        try:
            if frame_path is not None:
                frame_path.unlink()
        except FileNotFoundError:
            pass
    return None if online else output


def describe_pipeline() -> str:
    return "User API -> Converter -> ObjectCache -> Renderer -> FrameGenerator -> FrameBuffer -> Drawer"

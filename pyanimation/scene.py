from __future__ import annotations

import inspect
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import replace

from .animation import ReplacementTransform, flatten_animations
from .config import RenderConfig
from .mobject import Mobject, VGroup
from .geometry3d import Camera3D
from .object_cache import ObjectCacheWriter
from .pipeline import Converter, Drawer, ProgressReporter, Renderer


class Scene:
    """Manim-like scene API backed by the explicit render pipeline."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self.camera = Camera3D()
        self.mobjects: list[Mobject] = []
        self._commands: list[tuple] = []
        self._active: list[Mobject] = []
        self._drawer: Drawer | None = None
        self._converter: Converter | None = None
        self._cache_writer: ObjectCacheWriter | None = None
        self._conversion_executor: ThreadPoolExecutor | None = None
        self._renderer: Renderer | None = None
        self._has_frame = False
        self._rendered = False
        self._progress: ProgressReporter | None = None
        self._total_frames = 1

    def setup(self) -> None:
        """Hook for scene setup; override when needed."""

    def construct(self) -> None:
        raise NotImplementedError("Scene subclasses must implement construct()")

    @staticmethod
    def _expand(items):
        expanded = []
        pending = list(items)
        while pending:
            item = pending.pop(0)
            if isinstance(item, VGroup):
                pending[0:0] = item.submobjects
            elif isinstance(item, Mobject):
                expanded.append(item)
            else:
                raise TypeError(f"expected Mobject or VGroup, got {type(item).__name__}")
        return expanded

    def add(self, *mobjects) -> "Scene":
        items = self._expand(mobjects)
        self._commands.append(("add", items))
        for mob in items:
            if mob not in self.mobjects:
                self.mobjects.append(mob)
        return self

    def remove(self, *mobjects) -> "Scene":
        items = self._expand(mobjects)
        self._commands.append(("remove", items))
        for mob in items:
            if mob in self.mobjects:
                self.mobjects.remove(mob)
        return self

    def clear(self) -> "Scene":
        self._commands.append(("clear",))
        self.mobjects.clear()
        return self

    def play(self, *animations, run_time: float | None = None, rate_func=None) -> "Scene":
        if not animations:
            return self
        self._commands.append(("play", animations, run_time, rate_func))
        for animation, _, _ in flatten_animations(animations):
            for mob in self._expand((animation.mobject,)):
                if mob not in self.mobjects:
                    self.mobjects.append(mob)
                if animation.remover and mob in self.mobjects:
                    self.mobjects.remove(mob)
            if isinstance(animation, ReplacementTransform):
                if animation.mobject in self.mobjects:
                    self.mobjects.remove(animation.mobject)
                if animation.target not in self.mobjects:
                    self.mobjects.append(animation.target)
        return self

    def wait(self, duration: float = 1.0) -> "Scene":
        if duration < 0:
            raise ValueError("wait duration cannot be negative")
        self._commands.append(("wait", float(duration)))
        return self

    def _add_now(self, mobject: Mobject) -> None:
        if mobject not in self._active:
            self._active.append(mobject)

    def _remove_now(self, mobject: Mobject) -> None:
        if mobject in self._active:
            self._active.remove(mobject)

    def _emit(self) -> None:
        assert self._converter is not None and self._cache_writer is not None
        frame_index = self._cache_writer.frame_count + 1
        if self._conversion_executor is not None and len(self._active) > 1:
            converted = list(self._conversion_executor.map(self._converter.convert, self._active, range(len(self._active))))
        else:
            converted = [self._converter.convert(mob, i) for i, mob in enumerate(self._active)]
        self._cache_writer.write_frame(converted)
        if self._progress is not None:
            cache_name = self._cache_writer.path.as_posix()
            self._progress.update("Converting", frame_index, self._total_frames, f"objects={len(converted)} cache={cache_name}")
            meshes = [obj.mesh for obj in converted if obj.mesh is not None]
            if meshes:
                self._progress.update("3D Converting", frame_index, self._total_frames,
                                      f"meshes={len(meshes)} triangles={sum(len(m.faces) for m in meshes)} cache={cache_name}")
        self._has_frame = True

    def _estimate_total_frames(self) -> int:
        total_frames = 0
        has_frame = False
        for command in self._commands:
            kind = command[0]
            if kind == "wait":
                duration = command[1]
                if not has_frame:
                    total_frames += 1
                    has_frame = True
                if duration > 0:
                    total_frames += max(1, int(round(duration * self.config.fps)))
            elif kind == "play":
                _, animations, run_time, _ = command
                schedule = flatten_animations(animations)
                if not schedule:
                    continue
                duration = max(start + length for _, start, length in schedule)
                if run_time is not None:
                    duration = float(run_time)
                if not has_frame:
                    total_frames += 1
                    has_frame = True
                total_frames += max(1, int(round(duration * self.config.fps)))
        return max(1, total_frames)

    def _play_now(self, animations, run_time: float | None, rate_func=None) -> None:
        schedule = flatten_animations(animations)
        if not schedule:
            return
        if rate_func is not None:
            for animation, _, _ in schedule:
                animation.rate_func = rate_func
        total = max(start + duration for _, start, duration in schedule)
        if run_time is not None:
            if run_time <= 0:
                raise ValueError("run_time must be positive")
            factor = run_time / total if total > 0 else 1.0
            schedule = [(animation, start * factor, duration * factor) for animation, start, duration in schedule]
            total = float(run_time)

        pending = sorted(schedule, key=lambda item: item[1])
        running = []
        now = 0.0
        while pending and pending[0][1] <= now:
            animation, start, duration = pending.pop(0)
            animation.begin(self)
            running.append((animation, start, duration))
            animation.interpolate(0.0)
        if not self._has_frame:
            self._emit()

        ticks = max(1, int(round(total * self.config.fps)))
        for tick in range(1, ticks + 1):
            now = min(total, tick / self.config.fps)
            while pending and pending[0][1] <= now:
                animation, start, duration = pending.pop(0)
                animation.begin(self)
                running.append((animation, start, duration))
                animation.interpolate(0.0)
            finished = []
            for animation, start, duration in running:
                alpha = 1.0 if duration <= 0 else max(0.0, min(1.0, (now - start) / duration))
                animation.interpolate(alpha)
                if alpha >= 1:
                    animation.finish()
                    finished.append((animation, start, duration))
            for item in finished:
                running.remove(item)
            self._emit()

    def _wait_now(self, duration: float) -> None:
        if not self._has_frame:
            self._emit()
        if duration <= 0:
            return
        frames = max(1, int(round(duration * self.config.fps)))
        for _ in range(frames):
            self._emit()

    def render(self, output_file: str | Path | None = None, config: RenderConfig | None = None) -> Path:
        if self._rendered:
            raise RuntimeError("a Scene instance can only be rendered once")
        if config is not None:
            self.config = config
        output = Path(output_file or f"{type(self).__name__.lower()}.mp4")
        objects_path = None
        frame_path = None
        try:
            self.setup()
            initial_camera = replace(self.camera)
            self.construct()
            self._renderer = Renderer(self.config)
            self._total_frames = self._estimate_total_frames()
            self._progress = ProgressReporter(self._total_frames)

            try:
                script_path = Path(inspect.getfile(type(self)))
                script_stem = script_path.stem
            except (OSError, TypeError):
                script_stem = type(self).__name__.lower()
            cache_dir = Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            objects_path = cache_dir / f"{script_stem}.objects"
            with tempfile.NamedTemporaryFile(prefix=f"{script_stem}.", suffix=".frames", dir=cache_dir, delete=False) as temp_frames:
                frame_path = Path(temp_frames.name)

            self._converter = Converter(self.config, initial_camera)
            if self.config.effective_workers > 1:
                self._conversion_executor = ThreadPoolExecutor(
                    max_workers=self.config.effective_workers,
                    thread_name_prefix="pyanimation-convert",
                )
            self._cache_writer = ObjectCacheWriter(objects_path, self.config)
            for command in self._commands:
                kind = command[0]
                if kind == "add":
                    for mob in command[1]:
                        self._add_now(mob)
                elif kind == "remove":
                    for mob in command[1]:
                        self._remove_now(mob)
                elif kind == "clear":
                    self._active.clear()
                elif kind == "play":
                    _, animations, run_time, rate_func = command
                    self._play_now(animations, run_time, rate_func)
                elif kind == "wait":
                    self._wait_now(command[1])
                elif kind == "camera":
                    self._converter.camera = command[1]
            if not self._has_frame:
                self._emit()
            self._cache_writer.close()
            self._cache_writer = None
            if self._conversion_executor is not None:
                self._conversion_executor.shutdown(wait=True)
                self._conversion_executor = None

            rendered_count = self._renderer.render_cache(objects_path, frame_path, self._progress)
            if rendered_count != self._total_frames:
                raise RuntimeError(f"frame count mismatch: expected {self._total_frames}, got {rendered_count}")
            self._drawer = Drawer(output, self.config)
            self._drawer.write_frames(frame_path, rendered_count, self._progress)
        finally:
            if self._conversion_executor is not None:
                self._conversion_executor.shutdown(wait=True)
                self._conversion_executor = None
            if self._cache_writer is not None:
                self._cache_writer.abort()
                self._cache_writer = None
            if self._drawer is not None and self._drawer.process is not None:
                self._drawer.close()
            if frame_path is not None:
                try:
                    frame_path.unlink()
                except FileNotFoundError:
                    pass
            self._rendered = True
        return output


class MovingCameraScene(Scene):
    """Compatibility name for the 2D moving-camera scene base.

    Camera-frame objects are not implemented yet; ordinary Mobject movement
    and scaling are supported through the animation API.
    """


class ThreeDScene(Scene):
    """A scene with a perspective camera; ordinary 2D objects stay on screen."""

    def set_camera(self, camera: Camera3D | None = None, **settings):
        """Record a camera change at this point in the scene timeline.

        Example: set_camera(position=(6,4,10), target=(0,0,0)).
        Use projection='orthographic' for a parallel projection.
        """
        self.camera = replace(camera or self.camera, **settings)
        self._commands.append(("camera", self.camera))
        return self

    def add_fixed_in_frame_mobjects(self, *mobjects):
        """Place HUD objects above meshes in screen space; no 3D occlusion."""
        for mob in self._expand(mobjects):
            mob.set_z_index(max(10, mob.z_index))
        return self.add(*mobjects)

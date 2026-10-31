from __future__ import annotations

import copy
import math
from typing import Callable, Iterable

import numpy as np

from .mobject import Mobject, color_rgb
from .geometry3d import Mesh3D, Z_AXIS

RateFunc = Callable[[float], float]


def linear(t: float) -> float:
    return t


def smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def there_and_back(t: float) -> float:
    return smooth(2 * t) if t < 0.5 else smooth(2 - 2 * t)


def _resample(points: np.ndarray, count: int, closed: bool) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((count, 2))
    seq = np.vstack([points, points[:1]]) if closed else points
    if len(seq) == 1:
        return np.repeat(seq, count, axis=0)
    lengths = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    total = float(lengths.sum())
    if total == 0:
        return np.repeat(seq[:1], count, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0, total, count, endpoint=not closed)
    out = []
    for s in samples:
        edge = min(int(np.searchsorted(cumulative, s, side="right") - 1), len(lengths) - 1)
        edge_len = lengths[edge]
        f = 0 if edge_len == 0 else (s - cumulative[edge]) / edge_len
        out.append(seq[edge] * (1 - f) + seq[edge + 1] * f)
    return np.asarray(out)


class Animation:
    def __init__(self, mobject: Mobject, run_time=1.0, rate_func: RateFunc = smooth, remover=False):
        self.mobject = mobject
        self.run_time = float(run_time)
        if self.run_time <= 0:
            raise ValueError("animation run_time must be positive")
        self.rate_func = rate_func
        self.remover = remover
        self.scene = None

    def begin(self, scene) -> None:
        self.scene = scene
        scene._add_now(self.mobject)

    def interpolate(self, alpha: float) -> None:
        pass

    def finish(self) -> None:
        if self.remover and self.scene is not None:
            self.scene._remove_now(self.mobject)


class Transform(Animation):
    def __init__(self, mobject: Mobject, target: Mobject, run_time=1.0, rate_func: RateFunc = smooth):
        super().__init__(mobject, run_time, rate_func)
        self.target = target.copy()

    def begin(self, scene) -> None:
        self.is_mesh = isinstance(self.mobject, Mesh3D)
        if self.is_mesh != isinstance(self.target, Mesh3D):
            raise TypeError("Transform cannot mix 2D paths and 3D meshes")
        if self.is_mesh and (self.mobject.vertices.shape != self.target.vertices.shape
                             or not np.array_equal(self.mobject.faces, self.target.faces)):
            raise ValueError("3D Transform requires matching vertex counts and triangle topology")
        super().begin(scene)
        self.start = self.mobject.copy()
        if self.is_mesh:
            self.start_vertices = self.start.get_world_vertices()
            self.target_vertices = self.target.get_world_vertices()
        start_paths = self.start.get_world_paths()
        target_paths = self.target.get_world_paths()
        count = max(len(start_paths), len(target_paths))
        self.pairs = []
        for i in range(count):
            is_closed = self.start.closed[i] if i < len(self.start.closed) else False
            a = start_paths[i] if i < len(start_paths) else np.asarray([self.start.position], dtype=float)
            b = target_paths[i] if i < len(target_paths) else np.asarray([self.target.position], dtype=float)
            n = max(24, len(a), len(b))
            self.pairs.append((_resample(a, n, is_closed), _resample(b, n, is_closed)))
        self.start_opacity = self.mobject.opacity
        self.target_opacity = self.target.opacity
        self.start_fill = self.mobject.fill_opacity
        self.target_fill = self.target.fill_opacity
        self.start_stroke = self.mobject.stroke_width
        self.target_stroke = self.target.stroke_width
        self.start_position = np.asarray(self.mobject.position, dtype=float)
        self.target_position = np.asarray(self.target.position, dtype=float)
        self.start_rotation = self.mobject.rotation
        self.target_rotation = self.target.rotation
        self.start_scale = (self.mobject.scale_x, self.mobject.scale_y)
        self.target_scale = (self.target.scale_x, self.target.scale_y)
        self.start_font_size = getattr(self.mobject, "font_size", None)
        self.target_font_size = getattr(self.target, "font_size", None)

    def interpolate(self, alpha: float) -> None:
        t = self.rate_func(max(0.0, min(1.0, alpha)))
        self.mobject._world_override = [a * (1 - t) + b * t for a, b in self.pairs]
        if self.is_mesh:
            self.mobject._vertices_override = self.start_vertices * (1-t) + self.target_vertices * t
            self.mobject.position = tuple(self.start_position * (1-t) + self.target_position * t)
        self.mobject.opacity = self.start_opacity * (1 - t) + self.target_opacity * t
        self.mobject.fill_opacity = self.start_fill * (1 - t) + self.target_fill * t
        self.mobject.stroke_width = self.start_stroke * (1 - t) + self.target_stroke * t
        if self.start_font_size is not None and self.target_font_size is not None:
            self.mobject.position = tuple(self.start_position * (1 - t) + self.target_position * t)
            self.mobject.rotation = self.start_rotation * (1 - t) + self.target_rotation * t
            self.mobject.scale_x = self.start_scale[0] * (1 - t) + self.target_scale[0] * t
            self.mobject.scale_y = self.start_scale[1] * (1 - t) + self.target_scale[1] * t
            self.mobject.font_size = self.start_font_size * (1 - t) + self.target_font_size * t
        self.mobject.color = self._blend_color(self.start.color, self.target.color, t)
        self.mobject.fill_color = self._blend_color(self.start.fill_color, self.target.fill_color, t)
        self.mobject.stroke_color = self._blend_color(self.start.stroke_color, self.target.stroke_color, t)
        if alpha >= 1:
            if t <= 0:
                self.mobject.become(self.start)
            elif t >= 1:
                self.mobject.become(self.target)

    @staticmethod
    def _blend_color(first, second, alpha):
        a, b = color_rgb(first), color_rgb(second)
        return tuple(round(x * (1 - alpha) + y * alpha) for x, y in zip(a, b))


class MethodAnimation(Transform):
    def __init__(self, mobject: Mobject, operations, run_time=1.0, rate_func: RateFunc = smooth):
        self.operations = operations
        super().__init__(mobject, mobject.copy(), run_time, rate_func)

    def begin(self, scene) -> None:
        self.target = self.mobject.copy()
        for name, args, kwargs in self.operations:
            getattr(self.target, name)(*args, **kwargs)
        super().begin(scene)


class AnimationBuilder:
    def __init__(self, mobject: Mobject):
        self.mobject = mobject
        self.operations = []
        self.run_time = 1.0
        self.rate_func = smooth

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        target = self.mobject.copy()
        for op_name, args, kwargs in self.operations:
            getattr(target, op_name)(*args, **kwargs)
        method = getattr(target, name)
        if not callable(method):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.operations.append((name, args, kwargs))
            return self

        return call

    def build(self) -> MethodAnimation:
        return MethodAnimation(self.mobject, self.operations, self.run_time, self.rate_func)


class FadeIn(Animation):
    def begin(self, scene) -> None:
        self.target_opacity = self.mobject.opacity
        super().begin(scene)
        self.mobject.opacity = 0.0

    def interpolate(self, alpha: float) -> None:
        self.mobject.opacity = self.target_opacity * self.rate_func(alpha)


class FadeOut(Animation):
    def __init__(self, mobject, run_time=1.0, rate_func=smooth):
        super().__init__(mobject, run_time, rate_func, remover=True)

    def begin(self, scene) -> None:
        self.start_opacity = self.mobject.opacity
        super().begin(scene)

    def interpolate(self, alpha: float) -> None:
        self.mobject.opacity = self.start_opacity * (1 - self.rate_func(alpha))


class Create(Animation):
    def begin(self, scene) -> None:
        super().begin(scene)
        self.mobject.reveal = 0.0

    def interpolate(self, alpha: float) -> None:
        self.mobject.reveal = max(0.0, min(1.0, self.rate_func(alpha)))


class Uncreate(Create):
    def __init__(self, mobject, run_time=1.0, rate_func=smooth):
        super().__init__(mobject, run_time, rate_func, remover=True)

    def begin(self, scene) -> None:
        self.mobject.reveal = 1.0
        Animation.begin(self, scene)

    def interpolate(self, alpha: float) -> None:
        self.mobject.reveal = 1 - max(0.0, min(1.0, self.rate_func(alpha)))


class Write(Create):
    pass


class GrowFromCenter(FadeIn):
    def begin(self, scene) -> None:
        self.target_scale = (self.mobject.scale_x, self.mobject.scale_y)
        self.target_scale_z = getattr(self.mobject, "scale_z", None)
        super().begin(scene)
        self.mobject.scale_x = self.mobject.scale_y = 0.0
        if self.target_scale_z is not None:
            self.mobject.scale_z = 0.0

    def interpolate(self, alpha: float) -> None:
        t = self.rate_func(alpha)
        self.mobject.opacity = self.target_opacity * t
        self.mobject.scale_x = self.target_scale[0] * t
        self.mobject.scale_y = self.target_scale[1] * t
        if self.target_scale_z is not None:
            self.mobject.scale_z = self.target_scale_z * t


class ShrinkToCenter(GrowFromCenter):
    def __init__(self, mobject, run_time=1.0, rate_func=smooth):
        super().__init__(mobject, run_time, rate_func)
        self.remover = True

    def begin(self, scene) -> None:
        self.target_scale = (self.mobject.scale_x, self.mobject.scale_y)
        self.target_scale_z = getattr(self.mobject, "scale_z", None)
        self.target_opacity = self.mobject.opacity
        Animation.begin(self, scene)

    def interpolate(self, alpha: float) -> None:
        t = 1 - self.rate_func(alpha)
        self.mobject.opacity = self.target_opacity * t
        self.mobject.scale_x = self.target_scale[0] * t
        self.mobject.scale_y = self.target_scale[1] * t
        if self.target_scale_z is not None:
            self.mobject.scale_z = self.target_scale_z * t


class Indicate(Animation):
    def __init__(self, mobject, scale_factor=1.2, color="#ffff00", run_time=1.0, rate_func=there_and_back):
        super().__init__(mobject, run_time, rate_func)
        self.scale_factor = float(scale_factor)
        self.highlight = color

    def begin(self, scene) -> None:
        super().begin(scene)
        self.start_scale = (self.mobject.scale_x, self.mobject.scale_y)
        self.start_scale_z = getattr(self.mobject, "scale_z", None)
        self.start_color = self.mobject.color
        self.start_fill_color = self.mobject.fill_color
        self.start_stroke_color = self.mobject.stroke_color

    def interpolate(self, alpha: float) -> None:
        pulse = max(0.0, min(1.0, self.rate_func(alpha)))
        factor = 1 + (self.scale_factor - 1) * pulse
        self.mobject.scale_x = self.start_scale[0] * factor
        self.mobject.scale_y = self.start_scale[1] * factor
        if self.start_scale_z is not None:
            self.mobject.scale_z = self.start_scale_z * factor
        self.mobject.color = Transform._blend_color(self.start_color, self.highlight, pulse)
        self.mobject.fill_color = Transform._blend_color(self.start_fill_color, self.highlight, pulse)
        self.mobject.stroke_color = Transform._blend_color(self.start_stroke_color, self.highlight, pulse)


class Rotate(Animation):
    def __init__(self, mobject, angle=math.pi, run_time=1.0, rate_func=smooth, *, axis=Z_AXIS, about_point=None):
        super().__init__(mobject, run_time, rate_func)
        self.angle = float(angle)
        self.axis = axis
        self.about_point = about_point

    def begin(self, scene) -> None:
        super().begin(scene)
        self.start_rotation = self.mobject.rotation
        self.start = self.mobject.copy()

    def interpolate(self, alpha: float) -> None:
        angle = self.angle * self.rate_func(alpha)
        if isinstance(self.mobject, Mesh3D):
            self.mobject.orientation = self.start.orientation.copy()
            self.mobject.position = self.start.position
            self.mobject._vertices_override = (None if self.start._vertices_override is None
                                               else self.start._vertices_override.copy())
            self.mobject.rotate(angle, about_point=self.about_point, axis=self.axis)
        elif self.about_point is not None:
            self.mobject.position = self.start.position
            self.mobject.rotation = self.start_rotation
            self.mobject._world_override = self.start.copy()._world_override
            self.mobject.rotate(angle, about_point=self.about_point)
        else:
            self.mobject.rotation = self.start_rotation + angle


class AnimationGroup(Animation):
    def __init__(self, *animations: Animation, lag_ratio=0.0, run_time=None, rate_func: RateFunc = linear):
        if not animations:
            raise ValueError("AnimationGroup requires at least one animation")
        self.animations = [_prepare(animation) for animation in animations]
        self.lag_ratio = float(lag_ratio)
        if self.lag_ratio < 0:
            raise ValueError("lag_ratio cannot be negative")
        natural = max((i * self.lag_ratio + a.run_time for i, a in enumerate(self.animations)), default=0.0)
        self.run_time = float(run_time) if run_time is not None else natural
        if self.run_time <= 0:
            raise ValueError("animation group run_time must be positive")
        self.rate_func = rate_func
        self.mobject = self.animations[0].mobject

    def schedule(self):
        natural = max((i * self.lag_ratio + a.run_time for i, a in enumerate(self.animations)), default=0.0)
        factor = self.run_time / natural if natural > 0 else 1.0
        return [(anim, i * self.lag_ratio * factor, anim.run_time * factor) for i, anim in enumerate(self.animations)]


class LaggedStart(AnimationGroup):
    def __init__(self, *animations, lag_ratio=0.05, run_time=None):
        super().__init__(*animations, lag_ratio=lag_ratio, run_time=run_time)


class Succession(AnimationGroup):
    def __init__(self, *animations, run_time=None):
        super().__init__(*animations, lag_ratio=1.0, run_time=run_time)


class DrawBorderThenFill(AnimationGroup):
    def __init__(self, mobject: Mobject, run_time=1.0, rate_func: RateFunc = smooth):
        super().__init__(Create(mobject, run_time=run_time, rate_func=rate_func), FadeIn(mobject, run_time=run_time, rate_func=rate_func), lag_ratio=0.0, run_time=run_time)


class ReplacementTransform(Transform):
    def finish(self) -> None:
        if self.scene is not None:
            self.scene._remove_now(self.mobject)
            self.scene._add_now(self.target)


class TransformFromCopy(Transform):
    def __init__(self, source: Mobject, target: Mobject, run_time=1.0, rate_func: RateFunc = smooth):
        self.source = source
        self.copy_object = source.copy()
        super().__init__(self.copy_object, target, run_time, rate_func)


def _prepare(animation):
    if isinstance(animation, AnimationBuilder):
        return animation.build()
    if isinstance(animation, Animation):
        return animation
    raise TypeError(f"play() expected an Animation, got {type(animation).__name__}")


def flatten_animations(animations: Iterable, offset=0.0):
    result = []

    def expand(animation, start, duration):
        if isinstance(animation, AnimationGroup):
            ratio = duration / animation.run_time if animation.run_time > 0 else 1.0
            for child, child_offset, child_duration in animation.schedule():
                expand(child, start + child_offset * ratio, child_duration * ratio)
        elif hasattr(animation.mobject, "_leaves") and not isinstance(animation.mobject, Mobject):
            members = list(animation.mobject._leaves())
            target = getattr(animation, "target", None)
            targets = list(target._leaves()) if hasattr(target, "_leaves") else None
            if targets is not None and len(targets) != len(members):
                raise ValueError("group animation source and target must contain the same number of objects")
            pivot = animation.mobject.get_center() if isinstance(animation, Rotate) and animation.about_point is None else None
            for index, member in enumerate(members):
                child = copy.copy(animation)
                child.mobject = member
                child.scene = None
                if targets is not None:
                    child.target = targets[index]
                if pivot is not None:
                    child.about_point = pivot
                result.append((child, start, duration))
        else:
            result.append((animation, start, duration))

    for raw in animations:
        animation = _prepare(raw)
        expand(animation, offset, animation.run_time)
    return result

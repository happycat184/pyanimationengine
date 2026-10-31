from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]


def _point(value: Sequence[float]) -> np.ndarray:
    if len(value) < 2:
        raise ValueError("a point needs at least x and y")
    return np.array([float(value[0]), float(value[1])], dtype=np.float64)


def color_rgb(color: str | Sequence[float]) -> tuple[int, int, int]:
    if isinstance(color, str):
        value = color.strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6:
            raise ValueError(f"unsupported color: {color!r}")
        try:
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"unsupported color: {color!r}") from exc
    vals = tuple(float(v) for v in color[:3])
    if len(vals) != 3:
        raise ValueError("RGB colors need three channels")
    if max(vals) <= 1.0:
        vals = tuple(v * 255 for v in vals)
    return tuple(max(0, min(255, round(v))) for v in vals)  # type: ignore[return-value]


@dataclass(eq=False)
class Mobject:
    """A drawable object described in local coordinates.

    Shapes store paths rather than backend-specific drawing instructions.
    The converter maps these paths to pixels later in the render pipeline.
    """

    paths: list[np.ndarray]
    closed: list[bool]
    color: str | Sequence[float] = "#ffffff"
    fill_color: str | Sequence[float] | None = None
    stroke_color: str | Sequence[float] | None = None
    fill_opacity: float = 0.0
    stroke_width: float = 2.0
    opacity: float = 1.0
    position: Point = (0.0, 0.0)
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    z_index: int = 0
    reveal: float = 1.0
    name: str = "Mobject"
    _world_override: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.paths = [np.asarray(p, dtype=np.float64).reshape((-1, 2)) for p in self.paths]
        if len(self.closed) != len(self.paths):
            raise ValueError("closed must have one value per path")
        self.position = (float(self.position[0]), float(self.position[1]))
        if self.fill_color is None:
            self.fill_color = self.color
        if self.stroke_color is None:
            self.stroke_color = self.color

    def copy(self) -> "Mobject":
        import copy

        return copy.deepcopy(self)

    @property
    def animate(self):
        from .animation import AnimationBuilder

        return AnimationBuilder(self)

    @property
    def width(self) -> float:
        if not self.paths and hasattr(self, "_text_width"):
            return float(self._text_width * abs(self.scale_x))
        points = np.concatenate(self.paths) if self.paths else np.zeros((1, 2))
        return float((points[:, 0].max() - points[:, 0].min()) * abs(self.scale_x))

    @property
    def height(self) -> float:
        if not self.paths and hasattr(self, "_text_height"):
            return float(self._text_height * abs(self.scale_y))
        points = np.concatenate(self.paths) if self.paths else np.zeros((1, 2))
        return float((points[:, 1].max() - points[:, 1].min()) * abs(self.scale_y))

    def get_center(self) -> Point:
        return self.position

    def get_left(self) -> Point:
        return (self.position[0] - self.width / 2, self.position[1])

    def get_right(self) -> Point:
        return (self.position[0] + self.width / 2, self.position[1])

    def get_top(self) -> Point:
        return (self.position[0], self.position[1] + self.height / 2)

    def get_bottom(self) -> Point:
        return (self.position[0], self.position[1] - self.height / 2)

    def to_edge(self, direction=(0.0, 1.0), buff=0.5, frame_width=14.2222222222, frame_height=8.0) -> "Mobject":
        d = _point(direction)
        center = _point(self.position)
        if d[0]:
            center[0] = math.copysign(frame_width / 2 - self.width / 2 - buff, d[0])
        if d[1]:
            center[1] = math.copysign(frame_height / 2 - self.height / 2 - buff, d[1])
        return self.move_to(center)

    def get_world_paths(self) -> list[np.ndarray]:
        if self._world_override is not None:
            return [p.copy() for p in self._world_override]
        angle = self.rotation
        c, s = math.cos(angle), math.sin(angle)
        matrix = np.array([[c, -s], [s, c]])
        scale = np.diag([self.scale_x, self.scale_y])
        transform = matrix @ scale
        offset = np.asarray(self.position)
        return [(path @ transform.T) + offset for path in self.paths]

    def move_to(self, point: Sequence[float]) -> "Mobject":
        if hasattr(point, "get_center"):
            point = point.get_center()
        self.position = (float(point[0]), float(point[1]))
        self._world_override = None
        return self

    def next_to(self, other: "Mobject", direction: Sequence[float] = (1.0, 0.0), buff: float = 0.25) -> "Mobject":
        unit = _point(direction)
        norm = np.linalg.norm(unit)
        if norm == 0:
            raise ValueError("direction must not be zero")
        unit /= norm
        own_extent = abs(unit[0]) * self.width / 2 + abs(unit[1]) * self.height / 2
        other_extent = abs(unit[0]) * other.width / 2 + abs(unit[1]) * other.height / 2
        center = _point(other.get_center()) + unit * (own_extent + other_extent + float(buff))
        return self.move_to(center)

    def set_width(self, width: float) -> "Mobject":
        if self.width == 0:
            raise ValueError("cannot set width on a zero-width object")
        return self.scale(float(width) / self.width)

    def set_height(self, height: float) -> "Mobject":
        if self.height == 0:
            raise ValueError("cannot set height on a zero-height object")
        return self.scale(float(height) / self.height)

    def set_x(self, x: float) -> "Mobject":
        return self.move_to((float(x), self.position[1]))

    def set_y(self, y: float) -> "Mobject":
        return self.move_to((self.position[0], float(y)))

    def shift(self, vector: Sequence[float]) -> "Mobject":
        self.position = (self.position[0] + float(vector[0]), self.position[1] + float(vector[1]))
        if self._world_override is not None:
            delta = _point(vector)
            self._world_override = [p + delta for p in self._world_override]
        return self

    def scale(self, factor: float, about_point: Sequence[float] | None = None) -> "Mobject":
        origin = _point(about_point) if about_point is not None else _point(self.position)
        pos = _point(self.position)
        self.position = tuple(origin + (pos - origin) * factor)  # type: ignore[assignment]
        self.scale_x *= factor
        self.scale_y *= factor
        if self._world_override is not None:
            self._world_override = [origin + (p - origin) * factor for p in self._world_override]
        return self

    def stretch(self, factor: float, dim: int = 0) -> "Mobject":
        if dim not in (0, 1):
            raise ValueError("dim must be 0 (x) or 1 (y)")
        if dim == 0:
            self.scale_x *= factor
        else:
            self.scale_y *= factor
        return self

    def rotate(self, angle: float, about_point: Sequence[float] | None = None) -> "Mobject":
        origin = _point(about_point) if about_point is not None else _point(self.position)
        c, s = math.cos(angle), math.sin(angle)
        matrix = np.array([[c, -s], [s, c]])
        pos = _point(self.position)
        self.position = tuple(origin + matrix @ (pos - origin))  # type: ignore[assignment]
        self.rotation += angle
        if self._world_override is not None:
            self._world_override = [origin + (p - origin) @ matrix.T for p in self._world_override]
        return self

    def set_color(self, color: str | Sequence[float]) -> "Mobject":
        self.color = color
        self.fill_color = color
        self.stroke_color = color
        return self

    def set_fill(self, color: str | Sequence[float] | None = None, opacity: float = 1.0) -> "Mobject":
        if color is not None:
            self.fill_color = color
        self.fill_opacity = max(0.0, min(1.0, float(opacity)))
        return self

    def set_stroke(self, color: str | Sequence[float] | None = None, width: float | None = None) -> "Mobject":
        if color is not None:
            self.stroke_color = color
        if width is not None:
            self.stroke_width = max(0.0, float(width))
        return self

    def set_opacity(self, opacity: float) -> "Mobject":
        self.opacity = max(0.0, min(1.0, float(opacity)))
        return self

    def set_z_index(self, z_index: int) -> "Mobject":
        self.z_index = int(z_index)
        return self

    def become(self, other: "Mobject") -> "Mobject":
        replacement = other.copy()
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self


def _normalize(points: Iterable[Sequence[float]]) -> tuple[np.ndarray, Point]:
    arr = np.asarray([_point(p) for p in points], dtype=np.float64)
    if len(arr) == 0:
        return arr, (0.0, 0.0)
    center = (arr.min(axis=0) + arr.max(axis=0)) / 2
    return arr - center, (float(center[0]), float(center[1]))


class Polygon(Mobject):
    def __init__(self, *vertices: Sequence[float], color="#ffffff", fill_opacity=0.0, stroke_width=2.0, **kwargs):
        if len(vertices) < 3:
            raise ValueError("Polygon needs at least three vertices")
        points, center = _normalize(vertices)
        super().__init__([points], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, position=center, name="Polygon", **kwargs)


class Circle(Mobject):
    def __init__(self, radius=1.0, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, segments=128, **kwargs):
        if radius <= 0 or segments < 8:
            raise ValueError("radius must be positive and segments must be at least 8")
        points = [(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
        super().__init__([np.asarray(points)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Circle", **kwargs)
        self.radius = float(radius)

    def get_radius(self) -> float:
        return self.radius * max(abs(self.scale_x), abs(self.scale_y))


class Ellipse(Mobject):
    def __init__(self, width=4.0, height=2.0, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, segments=128, **kwargs):
        pts = [(width / 2 * math.cos(2 * math.pi * i / segments), height / 2 * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
        super().__init__([np.asarray(pts)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Ellipse", **kwargs)


class Rectangle(Mobject):
    def __init__(self, width=4.0, height=2.0, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, **kwargs):
        if width <= 0 or height <= 0:
            raise ValueError("rectangle dimensions must be positive")
        w, h = width / 2, height / 2
        pts = [(-w, -h), (w, -h), (w, h), (-w, h)]
        super().__init__([np.asarray(pts)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Rectangle", **kwargs)


class Square(Rectangle):
    def __init__(self, side_length=2.0, **kwargs):
        super().__init__(side_length, side_length, **kwargs)
        self.name = "Square"
        self.side_length = float(side_length)


class RoundedRectangle(Mobject):
    def __init__(self, width=4.0, height=2.0, corner_radius=0.25, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, **kwargs):
        r = min(max(0.0, corner_radius), width / 2, height / 2)
        points = []
        for cx, cy, start in [(width / 2 - r, -height / 2 + r, -90), (width / 2 - r, height / 2 - r, 0), (-width / 2 + r, height / 2 - r, 90), (-width / 2 + r, -height / 2 + r, 180)]:
            for i in range(9):
                a = math.radians(start + 90 * i / 8)
                points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        super().__init__([np.asarray(points)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="RoundedRectangle", **kwargs)


class RegularPolygon(Mobject):
    def __init__(self, n=6, radius=1.0, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, **kwargs):
        if n < 3:
            raise ValueError("a regular polygon needs at least three sides")
        pts = [(radius * math.cos(math.pi / 2 + 2 * math.pi * i / n), radius * math.sin(math.pi / 2 + 2 * math.pi * i / n)) for i in range(n)]
        super().__init__([np.asarray(pts)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="RegularPolygon", **kwargs)
        self.n = n


class Triangle(RegularPolygon):
    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)
        self.name = "Triangle"


class Star(Mobject):
    def __init__(self, n=5, outer_radius=1.0, inner_radius=None, color="#ffffff", fill_opacity=0.0, stroke_width=2.0, **kwargs):
        inner_radius = inner_radius if inner_radius is not None else outer_radius * 0.45
        points = []
        for i in range(2 * n):
            r = outer_radius if i % 2 == 0 else inner_radius
            a = math.pi / 2 + math.pi * i / n
            points.append((r * math.cos(a), r * math.sin(a)))
        super().__init__([np.asarray(points)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Star", **kwargs)


class Line(Mobject):
    def __init__(self, start=(-1.0, 0.0), end=(1.0, 0.0), color="#ffffff", stroke_width=3.0, **kwargs):
        pts, center = _normalize([start, end])
        super().__init__([pts], [False], color=color, stroke_width=stroke_width, position=center, name="Line", **kwargs)
        self.start = tuple(_point(start))
        self.end = tuple(_point(end))


class Arrow(Line):
    def __init__(self, start=(-1.0, 0.0), end=(1.0, 0.0), tip_length=0.25, **kwargs):
        a, b = _point(start), _point(end)
        direction = b - a
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("arrow endpoints must differ")
        u = direction / norm
        base = b - u * tip_length
        perp = np.array([-u[1], u[0]]) * tip_length * 0.45
        # One open path keeps the arrow head and shaft under one style.
        super().__init__(start, end, **kwargs)
        center = (a + b) / 2
        shaft = np.asarray([a - center, base - center])
        head = np.asarray([p - center for p in (base - perp, b, base + perp)])
        self.paths = [shaft, head]
        self.closed = [False, False]
        self.position = tuple(center)
        self.name = "Arrow"


class DashedLine(Line):
    def __init__(self, start=(-1.0, 0.0), end=(1.0, 0.0), dash_length=0.15, **kwargs):
        if dash_length <= 0:
            raise ValueError("dash_length must be positive")
        a, b = _point(start), _point(end)
        d = b - a
        length = np.linalg.norm(d)
        if length == 0:
            raise ValueError("line endpoints must differ")
        unit = d / length
        paths = []
        t = 0.0
        while t < length:
            stop = min(length, t + dash_length)
            paths.append(np.asarray([a + t * unit, a + stop * unit]))
            t += 2 * dash_length
        center = (a + b) / 2
        super().__init__(start, end, **kwargs)
        self.paths = [p - center for p in paths]
        self.closed = [False] * len(paths)
        self.name = "DashedLine"


class Dot(Circle):
    def __init__(self, point=(0.0, 0.0), radius=0.08, **kwargs):
        super().__init__(radius=radius, fill_opacity=1.0, stroke_width=0.0, **kwargs)
        self.move_to(point)
        self.name = "Dot"


class Arc(Mobject):
    def __init__(self, radius=1.0, angle=math.pi / 2, start_angle=0.0, color="#ffffff", stroke_width=2.0, segments=64, **kwargs):
        pts = [(radius * math.cos(start_angle + angle * i / segments), radius * math.sin(start_angle + angle * i / segments)) for i in range(segments + 1)]
        super().__init__([np.asarray(pts)], [False], color=color, stroke_width=stroke_width, name="Arc", **kwargs)


class Sector(Mobject):
    def __init__(self, radius=1.0, angle=math.pi / 2, start_angle=0.0, color="#ffffff", fill_opacity=0.5, stroke_width=2.0, segments=64, **kwargs):
        pts = [(0.0, 0.0)] + [(radius * math.cos(start_angle + angle * i / segments), radius * math.sin(start_angle + angle * i / segments)) for i in range(segments + 1)]
        super().__init__([np.asarray(pts)], [True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Sector", **kwargs)


class Annulus(Mobject):
    def __init__(self, inner_radius=0.5, outer_radius=1.0, color="#ffffff", fill_opacity=1.0, stroke_width=1.0, segments=128, **kwargs):
        outer = [(outer_radius * math.cos(2 * math.pi * i / segments), outer_radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
        inner = [(inner_radius * math.cos(-2 * math.pi * i / segments), inner_radius * math.sin(-2 * math.pi * i / segments)) for i in range(segments)]
        super().__init__([np.asarray(outer), np.asarray(inner)], [True, True], color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, name="Annulus", **kwargs)


class Text(Mobject):
    def __init__(
        self,
        text: str,
        font_size=0.55,
        color="#ffffff",
        font: str | None = None,
        font_path: str | None = None,
        font_family: str | None = None,
        **kwargs,
    ):
        super().__init__([], [], color=color, fill_opacity=1.0, stroke_width=0.0, name="Text", **kwargs)
        self.text = str(text)
        self.font_size = float(font_size)
        self.font = str(font_path or font_family or font) if (font_path or font_family or font) else None
        self._text_width = self.font_size * sum(1.0 if ord(char) > 127 else 0.58 for char in self.text)
        self._text_height = self.font_size


class VGroup:
    """A lightweight collection with layout helpers; scenes accept its members."""

    def __init__(self, *mobjects: Mobject):
        self.submobjects = list(mobjects)

    def __iter__(self):
        return iter(self.submobjects)

    def __len__(self):
        return len(self.submobjects)

    def __getitem__(self, item):
        result = self.submobjects[item]
        return VGroup(*result) if isinstance(item, slice) else result

    def add(self, *mobjects: Mobject) -> "VGroup":
        self.submobjects.extend(mobjects)
        return self

    def _leaves(self):
        for mob in self.submobjects:
            if isinstance(mob, VGroup):
                yield from mob._leaves()
            elif isinstance(mob, Mobject):
                yield mob

    def _bounds(self):
        leaves = list(self._leaves())
        if not leaves:
            return (-0.0, 0.0, 0.0, 0.0)
        left = min(m.position[0] - m.width / 2 for m in leaves)
        right = max(m.position[0] + m.width / 2 for m in leaves)
        bottom = min(m.position[1] - m.height / 2 for m in leaves)
        top = max(m.position[1] + m.height / 2 for m in leaves)
        return left, right, bottom, top

    @property
    def width(self) -> float:
        left, right, _, _ = self._bounds()
        return right - left

    @property
    def height(self) -> float:
        _, _, bottom, top = self._bounds()
        return top - bottom

    def get_center(self) -> Point:
        left, right, bottom, top = self._bounds()
        return ((left + right) / 2, (bottom + top) / 2)

    def get_left(self) -> Point:
        left, _, bottom, top = self._bounds()
        return (left, (bottom + top) / 2)

    def get_right(self) -> Point:
        _, right, bottom, top = self._bounds()
        return (right, (bottom + top) / 2)

    def get_top(self) -> Point:
        left, right, _, top = self._bounds()
        return ((left + right) / 2, top)

    def get_bottom(self) -> Point:
        left, right, bottom, _ = self._bounds()
        return ((left + right) / 2, bottom)

    def move_to(self, point) -> "VGroup":
        if hasattr(point, "get_center"):
            point = point.get_center()
        target = _point(point)
        delta = target - _point(self.get_center())
        for mob in self._leaves():
            mob.shift(delta)
        return self

    def next_to(self, other, direction=(1.0, 0.0), buff=0.25) -> "VGroup":
        unit = _point(direction)
        norm = np.linalg.norm(unit)
        if norm == 0:
            raise ValueError("direction must not be zero")
        unit /= norm
        own_extent = abs(unit[0]) * self.width / 2 + abs(unit[1]) * self.height / 2
        other_extent = abs(unit[0]) * other.width / 2 + abs(unit[1]) * other.height / 2
        return self.move_to(_point(other.get_center()) + unit * (own_extent + other_extent + buff))

    def to_edge(self, direction=(0.0, 1.0), buff=0.5, frame_width=14.2222222222, frame_height=8.0) -> "VGroup":
        d = _point(direction)
        center = _point(self.get_center())
        if d[0]:
            center[0] = math.copysign(frame_width / 2 - self.width / 2 - buff, d[0])
        if d[1]:
            center[1] = math.copysign(frame_height / 2 - self.height / 2 - buff, d[1])
        return self.move_to(center)

    def scale(self, factor: float, about_point=None) -> "VGroup":
        origin = _point(about_point) if about_point is not None else _point(self.get_center())
        for mob in self._leaves():
            mob.scale(factor, about_point=origin)
        return self

    def set_width(self, width: float) -> "VGroup":
        if self.width == 0:
            raise ValueError("cannot set width on an empty group")
        return self.scale(float(width) / self.width)

    def set_height(self, height: float) -> "VGroup":
        if self.height == 0:
            raise ValueError("cannot set height on an empty group")
        return self.scale(float(height) / self.height)

    def set_color(self, color) -> "VGroup":
        for mob in self._leaves():
            mob.set_color(color)
        return self

    def set_opacity(self, opacity: float) -> "VGroup":
        for mob in self._leaves():
            mob.set_opacity(opacity)
        return self

    def set_z_index(self, z_index: int) -> "VGroup":
        for mob in self._leaves():
            mob.set_z_index(z_index)
        return self

    def arrange(self, direction=(1.0, 0.0), buff=0.25, aligned_edge=None) -> "VGroup":
        if not self.submobjects:
            return self
        d = _point(direction)
        norm = np.linalg.norm(d)
        if norm == 0:
            raise ValueError("direction must not be zero")
        d /= norm
        extents = [abs(d[0]) * mob.width / 2 + abs(d[1]) * mob.height / 2 for mob in self.submobjects]
        total = sum(extents) + max(0, len(extents) - 1) * buff
        cursor = -total / 2
        centers = []
        for extent in extents:
            centers.append(d * (cursor + extent / 2))
            cursor += extent + buff

        if aligned_edge is not None:
            edge = _point(aligned_edge)
            if np.linalg.norm(edge) == 0:
                raise ValueError("aligned_edge must not be zero")
            axis = 0 if abs(edge[0]) >= abs(edge[1]) else 1
            sign = math.copysign(1.0, edge[axis])
            projections = [center[axis] + sign * (mob.width if axis == 0 else mob.height) / 2 for center, mob in zip(centers, self.submobjects)]
            target = max(projections) if sign > 0 else min(projections)
            for center, mob, projection in zip(centers, self.submobjects, projections):
                extent = (mob.width if axis == 0 else mob.height) / 2
                center[axis] += target - projection
        for mob, center in zip(self.submobjects, centers):
            mob.move_to(center)
        return self


ORIGIN = (0.0, 0.0)
UP = (0.0, 1.0)
DOWN = (0.0, -1.0)
LEFT = (-1.0, 0.0)
RIGHT = (1.0, 0.0)
UL = (-1.0, 1.0)
UR = (1.0, 1.0)
DL = (-1.0, -1.0)
DR = (1.0, -1.0)
PI = math.pi
TAU = 2 * math.pi
DEGREES = math.pi / 180
WHITE = "#ffffff"
BLACK = "#000000"
RED = "#ef476f"
GREEN = "#06d6a0"
BLUE = "#118ab2"
YELLOW = "#ffd166"
ORANGE = "#f78c32"
PURPLE = "#9b5de5"
PINK = "#f15bb5"
TEAL = "#00b4d8"

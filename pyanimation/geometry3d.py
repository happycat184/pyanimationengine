"""Indexed triangle meshes and explicit world/view coordinate mathematics.

Coordinates are right handed: X right, Y up, Z towards the default camera.
No external geometry or 3D rendering library is used.
"""
from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

from .mobject import Mobject, VGroup, color_rgb

X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)
Z_AXIS = OUT = (0.0, 0.0, 1.0)
IN = (0.0, 0.0, -1.0)


def point3(value) -> np.ndarray:
    """Convert an XY or XYZ sequence into a finite float64 3-vector."""
    a = np.asarray(value, dtype=float)
    if a.shape == (2,):
        a = np.append(a, 0.0)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("expected a finite 2D or 3D point")
    return a


def unit3(value) -> np.ndarray:
    """Return a normalized direction, rejecting a zero-length vector."""
    a = point3(value)
    length = np.linalg.norm(a)
    if length < 1e-12:
        raise ValueError("direction must be nonzero")
    return a / length


def rotation_matrix(angle, axis=Z_AXIS):
    """Rodrigues' formula; vectors are multiplied by the matrix on the left."""
    x, y, z = unit3(axis)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    c, s = math.cos(angle), math.sin(angle)
    return c * np.eye(3) + (1 - c) * np.outer((x, y, z), (x, y, z)) + s * skew


def _positive(value, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _segments(value, minimum: int = 3) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"resolution must be an integer >= {minimum}")
    return value


@dataclass
class Camera3D:
    position: tuple = (6.0, 4.0, 10.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = Y_AXIS
    projection: str = "perspective"
    focal_distance: float | None = None
    near: float = 0.1
    far: float = 1000.0
    light_direction: tuple = (-3.0, 5.0, 7.0)
    ambient: float = 0.3

    def __post_init__(self) -> None:
        self.position = tuple(point3(self.position))
        self.target = tuple(point3(self.target))
        self.up = tuple(unit3(self.up))
        self.light_direction = tuple(unit3(self.light_direction))
        if self.projection not in {"perspective", "orthographic"}:
            raise ValueError("projection must be perspective or orthographic")
        _positive(self.near, "near")
        _positive(self.far, "far")
        if self.far <= self.near:
            raise ValueError("far must be greater than near")
        if self.focal_distance is not None:
            _positive(self.focal_distance, "focal_distance")
        if not 0 <= self.ambient <= 1:
            raise ValueError("ambient must be in [0, 1]")
        self.basis()  # reject coincident eye/target or parallel up direction

    def basis(self) -> np.ndarray:
        """Return right, up, and forward as rows of a view matrix."""
        forward = unit3(point3(self.target) - point3(self.position))
        right = unit3(np.cross(forward, self.up))
        up = np.cross(right, forward)
        return np.stack((right, up, forward))

    @property
    def focal_length(self):
        return self.focal_distance or float(np.linalg.norm(point3(self.target) - self.position))

    def world_to_view(self, vertices: np.ndarray) -> np.ndarray:
        return (np.asarray(vertices) - self.position) @ self.basis().T


@dataclass
class RenderMesh:
    """Portable snapshot of camera-space vertices, faces, and lit RGB colors."""

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    perspective: bool
    focal_length: float
    near: float
    far: float


class Mesh3D(Mobject):
    """An indexed triangle mesh transformed in world space before projection."""

    def __init__(
        self,
        vertices,
        faces,
        color="#58a6ff",
        fill_opacity=1.0,
        face_colors=None,
        position=(0, 0, 0),
        **kwargs,
    ):
        # Mesh edges are actual geometry (Line3D); 2D stroke_width is not used.
        kwargs.setdefault("stroke_width", 0.0)
        super().__init__([], [], color=color, fill_opacity=fill_opacity, **kwargs)
        self.vertices = np.array(vertices, dtype=float, copy=True)
        raw_faces = np.asarray(faces)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise ValueError("vertices must be an N x 3 array with N >= 3")
        if not np.isfinite(self.vertices).all():
            raise ValueError("mesh vertices must be finite")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
            raise ValueError("faces must be a nonempty M x 3 triangle index array")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("face indices must be integers")
        if raw_faces.min() < 0 or raw_faces.max() >= len(self.vertices):
            raise ValueError("face index outside vertex array")
        self.faces = raw_faces.astype(np.uint32, copy=True)
        if face_colors is not None and len(face_colors) != len(self.faces):
            raise ValueError("face_colors must have one color per triangle")
        self.face_colors = None if face_colors is None else np.array([color_rgb(c) for c in face_colors], dtype=np.uint8)
        self.position = tuple(point3(position))
        self.scale_z = 1.0
        self.orientation = np.eye(3)
        self._vertices_override = None
        self.name = type(self).__name__

    def get_world_vertices(self) -> np.ndarray:
        """Apply local scale and orientation, then move vertices into world space."""
        if self._vertices_override is not None:
            return self._vertices_override.copy()
        scale = np.array((self.scale_x, self.scale_y, self.scale_z))
        return (self.vertices * scale) @ self.orientation.T + self.position

    def _extent(self, axis: int) -> float:
        return float(np.ptp(self.get_world_vertices()[:, axis]))

    @property
    def width(self):
        return self._extent(0)

    @property
    def height(self):
        return self._extent(1)

    def get_center(self):
        vertices = self.get_world_vertices()
        return tuple((vertices.min(axis=0) + vertices.max(axis=0)) / 2)

    def get_left(self):
        vertices = self.get_world_vertices()
        return (float(vertices[:, 0].min()), self.get_center()[1], self.get_center()[2])

    def get_right(self):
        vertices = self.get_world_vertices()
        return (float(vertices[:, 0].max()), self.get_center()[1], self.get_center()[2])

    def get_top(self):
        vertices = self.get_world_vertices()
        return (self.get_center()[0], float(vertices[:, 1].max()), self.get_center()[2])

    def get_bottom(self):
        vertices = self.get_world_vertices()
        return (self.get_center()[0], float(vertices[:, 1].min()), self.get_center()[2])

    @property
    def depth(self):
        return self._extent(2)

    def move_to(self, point):
        if hasattr(point, "get_center"):
            point = point.get_center()
        return self.shift(point3(point) - point3(self.get_center()))

    def shift(self, vector):
        delta = point3(vector)
        self.position = tuple(point3(self.position) + delta)
        if self._vertices_override is not None:
            self._vertices_override += delta
        return self

    def scale(self, factor, about_point=None):
        if not math.isfinite(factor):
            raise ValueError("scale must be finite")
        origin = point3(self.get_center() if about_point is None else about_point)
        self.position = tuple(origin + (point3(self.position) - origin) * factor)
        self.scale_x *= factor
        self.scale_y *= factor
        self.scale_z *= factor
        if self._vertices_override is not None:
            self._vertices_override = origin + (self._vertices_override - origin) * factor
        return self

    def stretch(self, factor, dim=0):
        if dim not in (0, 1, 2) or not math.isfinite(factor):
            raise ValueError("dim must be 0, 1 or 2 and factor must be finite")
        name = ("scale_x", "scale_y", "scale_z")[dim]
        setattr(self, name, getattr(self, name) * factor)
        return self

    def rotate(self, angle, axis=Z_AXIS, about_point=None):
        matrix = rotation_matrix(angle, axis)
        origin = point3(self.get_center() if about_point is None else about_point)
        self.position = tuple(origin + matrix @ (point3(self.position) - origin))
        self.orientation = matrix @ self.orientation
        if self._vertices_override is not None:
            self._vertices_override = (self._vertices_override - origin) @ matrix.T + origin
        return self

    def set_x(self, x):
        center = self.get_center()
        return self.move_to((x, center[1], center[2]))

    def set_y(self, y):
        center = self.get_center()
        return self.move_to((center[0], y, center[2]))

    def set_z(self, z):
        center = self.get_center()
        return self.move_to((center[0], center[1], z))

    def set_color(self, color):
        self.face_colors = None
        return super().set_color(color)

    def to_render_mesh(self, camera: Camera3D) -> RenderMesh:
        """Transform vertices to camera space and compute one Lambert color per face."""
        world = self.get_world_vertices()
        triangles = world[self.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-20)
        diffuse = np.maximum(0, normals @ np.asarray(camera.light_direction))
        intensity = camera.ambient + (1 - camera.ambient) * diffuse
        colors = self.face_colors if self.face_colors is not None else np.asarray(color_rgb(self.fill_color))
        colors = np.clip(colors * intensity[:, None], 0, 255).astype(np.uint8)
        return RenderMesh(
            vertices=camera.world_to_view(world).astype(np.float32),
            faces=self.faces.copy(),
            colors=colors,
            perspective=camera.projection == "perspective",
            focal_length=camera.focal_length,
            near=camera.near,
            far=camera.far,
        )


class Polyhedron(Mesh3D):
    """Convex polygon faces are triangulated as fans; winding faces outwards."""
    def __init__(self, vertices, faces, **kwargs):
        triangles = []
        for face in faces:
            if len(face) < 3:
                raise ValueError("each face needs at least three vertex indices")
            triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
        super().__init__(vertices, triangles, **kwargs)


class Cuboid(Polyhedron):
    def __init__(self, width=2.0, height=2.0, depth=2.0, **kwargs):
        w, h, d = [_positive(v, n) / 2 for v, n in zip((width, height, depth), ("width", "height", "depth"))]
        vertices = [(-w,-h,-d),(w,-h,-d),(w,h,-d),(-w,h,-d),(-w,-h,d),(w,-h,d),(w,h,d),(-w,h,d)]
        faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(3,7,6,2),(0,4,7,3),(1,2,6,5)]
        super().__init__(vertices, faces, **kwargs)


class Cube(Cuboid):
    def __init__(self, side_length=2.0, **kwargs):
        super().__init__(side_length, side_length, side_length, **kwargs)


class Sphere(Mesh3D):
    def __init__(self, radius=1.0, resolution=(16, 32), **kwargs):
        r = _positive(radius, "radius")
        rings, segments = (_segments(n) for n in resolution)
        vertices = [(0, r, 0)]
        for i in range(1, rings):
            phi = math.pi * i / rings
            for j in range(segments):
                t = 2 * math.pi * j / segments
                vertices.append((r*math.sin(phi)*math.cos(t), r*math.cos(phi), r*math.sin(phi)*math.sin(t)))
        bottom = len(vertices)
        vertices.append((0, -r, 0))
        faces = [(0, 1+(j+1)%segments, 1+j) for j in range(segments)]
        for i in range(rings - 2):
            for j in range(segments):
                a, b = 1+i*segments+j, 1+i*segments+(j+1)%segments
                faces.extend(((a,b,b+segments),(a,b+segments,a+segments)))
        start = 1 + (rings-2)*segments
        faces.extend((bottom, start+j, start+(j+1)%segments) for j in range(segments))
        super().__init__(vertices, faces, **kwargs)


class Cylinder(Mesh3D):
    def __init__(self, radius=1.0, height=2.0, resolution=32, **kwargs):
        r, h = _positive(radius, "radius"), _positive(height, "height") / 2
        n = _segments(resolution)
        vertices = [(r*math.cos(2*math.pi*j/n), y, r*math.sin(2*math.pi*j/n)) for y in (-h,h) for j in range(n)]
        vertices += [(0,-h,0),(0,h,0)]
        faces = []
        for a in range(n):
            b = (a+1)%n
            faces.extend(((a,a+n,b+n),(a,b+n,b),(2*n,a,b),(2*n+1,b+n,a+n)))
        super().__init__(vertices, faces, **kwargs)


class Cone(Mesh3D):
    def __init__(self, radius=1.0, height=2.0, resolution=32, **kwargs):
        r, h = _positive(radius, "radius"), _positive(height, "height") / 2
        n = _segments(resolution)
        vertices = [(r*math.cos(2*math.pi*j/n), -h, r*math.sin(2*math.pi*j/n)) for j in range(n)]
        vertices += [(0,h,0),(0,-h,0)]
        faces = []
        for a in range(n):
            b = (a+1)%n
            faces.extend(((a,n,b),(n+1,a,b)))
        super().__init__(vertices, faces, **kwargs)


class Torus(Mesh3D):
    def __init__(self, major_radius=1.0, minor_radius=0.35, resolution=(32, 12), **kwargs):
        r, tube = _positive(major_radius, "major_radius"), _positive(minor_radius, "minor_radius")
        if tube >= r:
            raise ValueError("minor_radius must be smaller than major_radius")
        nu, nv = (_segments(n) for n in resolution)
        vertices, faces = [], []
        for i in range(nu):
            u = 2*math.pi*i/nu
            for j in range(nv):
                v = 2*math.pi*j/nv
                vertices.append(((r+tube*math.cos(v))*math.cos(u), tube*math.sin(v), (r+tube*math.cos(v))*math.sin(u)))
                a, b = i*nv+j, ((i+1)%nu)*nv+j
                c, d = ((i+1)%nu)*nv+(j+1)%nv, i*nv+(j+1)%nv
                faces.extend(((a,d,c),(a,c,b)))
        super().__init__(vertices, faces, **kwargs)


class Plane(Polyhedron):
    """An XY plane, facing +Z. Rotate it to make a floor."""
    def __init__(self, width=4.0, height=4.0, **kwargs):
        w, h = _positive(width, "width")/2, _positive(height, "height")/2
        super().__init__([(-w,-h,0),(w,-h,0),(w,h,0),(-w,h,0)], [(0,1,2,3)], **kwargs)


class Tetrahedron(Polyhedron):
    def __init__(self, radius=1.0, **kwargs):
        vertices = np.array([(1,1,1),(-1,-1,1),(-1,1,-1),(1,-1,-1)]) * (_positive(radius, "radius")/math.sqrt(3))
        super().__init__(vertices, [(0,2,1),(0,1,3),(0,3,2),(1,2,3)], **kwargs)


class Octahedron(Polyhedron):
    def __init__(self, radius=1.0, **kwargs):
        vertices = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]) * _positive(radius, "radius")
        super().__init__(vertices, [(2,4,0),(2,1,4),(2,5,1),(2,0,5),(3,0,4),(3,4,1),(3,1,5),(3,5,0)], **kwargs)


class Surface(Mesh3D):
    """Sample a function f(u,v)->(x,y,z); normals follow df/du cross df/dv."""
    def __init__(self, function, u_range=(-2,2), v_range=(-2,2), resolution=(24,24), **kwargs):
        nu, nv = (_segments(n, 1) for n in resolution)
        vertices = [point3(function(u,v)) for u in np.linspace(*u_range, nu+1) for v in np.linspace(*v_range, nv+1)]
        faces = []
        for i in range(nu):
            for j in range(nv):
                a = i*(nv+1)+j
                faces.extend(((a,a+nv+1,a+nv+2),(a,a+nv+2,a+1)))
        super().__init__(vertices, faces, **kwargs)


class Line3D(Cylinder):
    """A solid cylindrical segment, with a world-space radius."""
    def __init__(self, start=(-1,0,0), end=(1,0,0), radius=0.025, resolution=8, **kwargs):
        a, b = point3(start), point3(end)
        direction = unit3(b-a)
        super().__init__(radius, float(np.linalg.norm(b-a)), resolution, **kwargs)
        axis = np.cross(Y_AXIS, direction)
        if np.linalg.norm(axis) > 1e-12:
            self.rotate(math.acos(np.clip(direction[1], -1, 1)), axis=axis)
        elif direction[1] < 0:
            self.rotate(math.pi, axis=X_AXIS)
        self.move_to((a+b)/2)


class Group3D(VGroup):
    def __init__(self, *meshes):
        super().__init__(*meshes)
        if any(not isinstance(m, Mesh3D) for m in self._leaves()):
            raise TypeError("Group3D accepts meshes and nested Group3D collections")

    def add(self, *meshes):
        if any(not isinstance(mesh, Mesh3D) and not isinstance(mesh, Group3D) for mesh in meshes):
            raise TypeError("Group3D only accepts Mesh3D objects or nested Group3D collections")
        self.submobjects.extend(meshes)
        return self

    def get_center(self):
        leaves = list(self._leaves())
        if not leaves:
            return (0., 0., 0.)
        points = np.concatenate([m.get_world_vertices() for m in leaves])
        return tuple((points.min(axis=0)+points.max(axis=0))/2)

    @property
    def width(self):
        leaves = list(self._leaves())
        return float(np.ptp(np.concatenate([m.get_world_vertices() for m in leaves])[:,0])) if leaves else 0.0

    @property
    def height(self):
        leaves = list(self._leaves())
        return float(np.ptp(np.concatenate([m.get_world_vertices() for m in leaves])[:,1])) if leaves else 0.0

    @property
    def depth(self):
        leaves = list(self._leaves())
        return float(np.ptp(np.concatenate([m.get_world_vertices() for m in leaves])[:,2])) if leaves else 0.0

    def copy(self):
        import copy
        return copy.deepcopy(self)

    @property
    def animate(self):
        from .animation import AnimationBuilder
        return AnimationBuilder(self)

    def move_to(self, point):
        return self.shift(point3(point)-self.get_center())

    def next_to(self, other, direction=(1, 0, 0), buff=0.25):
        direction = unit3(direction)
        own_points = np.concatenate([m.get_world_vertices() for m in self._leaves()])
        if isinstance(other, Group3D):
            other_points = np.concatenate([m.get_world_vertices() for m in other._leaves()])
        elif isinstance(other, Mesh3D):
            other_points = other.get_world_vertices()
        else:
            other_points = point3(other.get_center())[None, :]
        separation = np.max(other_points @ direction) - np.min(own_points @ direction) + float(buff)
        return self.shift(direction * separation)

    def shift(self, vector):
        for m in self._leaves():
            m.shift(vector)
        return self

    def scale(self, factor, about_point=None):
        origin = self.get_center() if about_point is None else about_point
        for m in self._leaves():
            m.scale(factor, about_point=origin)
        return self

    def rotate(self, angle, axis=Z_AXIS, about_point=None):
        origin = self.get_center() if about_point is None else about_point
        for m in self._leaves():
            m.rotate(angle, about_point=origin, axis=axis)
        return self

    def arrange_in_grid(self, columns=3, buff=0.75):
        """Arrange meshes in a centered XY grid while keeping each Z value."""
        if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
            raise ValueError("columns must be a positive integer")
        if buff < 0 or not math.isfinite(buff):
            raise ValueError("buff must be a finite nonnegative number")
        leaves = list(self._leaves())
        if not leaves:
            return self
        rows = math.ceil(len(leaves) / columns)
        cell_width = max(m.width for m in leaves)
        cell_height = max(m.height for m in leaves)
        spacing_x, spacing_y = cell_width + buff, cell_height + buff
        for index, mesh in enumerate(leaves):
            row, column = divmod(index, columns)
            row_count = min(columns, len(leaves) - row * columns)
            x = (column - (row_count - 1) / 2) * spacing_x
            y = ((rows - 1) / 2 - row) * spacing_y
            mesh.move_to((x, y, mesh.get_center()[2]))
        return self

    def to_edge(self, direction=(0, 1), buff=0.5, frame_width=14.2222222222, frame_height=8.0):
        leaves = list(self._leaves())
        if not leaves:
            return self
        direction = point3(direction)
        center = point3(self.get_center())
        bounds = np.concatenate([m.get_world_vertices() for m in leaves])
        if direction[0]:
            center[0] = math.copysign(frame_width/2 - np.ptp(bounds[:,0])/2 - buff, direction[0])
        if direction[1]:
            center[1] = math.copysign(frame_height/2 - np.ptp(bounds[:,1])/2 - buff, direction[1])
        return self.move_to(center)


class ThreeDAxes(Group3D):
    def __init__(self, length=3.0, radius=0.02):
        length = _positive(length, "length")
        super().__init__(Line3D((0,0,0),(length,0,0),radius,color="#ff5555"),
                         Line3D((0,0,0),(0,length,0),radius,color="#55dd88"),
                         Line3D((0,0,0),(0,0,length),radius,color="#5599ff"))

"""A low-level 2D/3D animation engine with a Manim-inspired API."""

from .animation import (
    Animation,
    AnimationGroup,
    Create,
    DrawBorderThenFill,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    MethodAnimation,
    ReplacementTransform,
    Rotate,
    Succession,
    ShrinkToCenter,
    Transform,
    TransformFromCopy,
    Uncreate,
    Write,
    linear,
    smooth,
    there_and_back,
)
from .config import RenderConfig
from .mobject import (
    Annulus,
    Arc,
    Arrow,
    BLACK,
    BLUE,
    Circle,
    DashedLine,
    DEGREES,
    DL,
    Dot,
    DOWN,
    DR,
    Ellipse,
    GREEN,
    LEFT,
    Line,
    Mobject,
    ORANGE,
    ORIGIN,
    PI,
    PINK,
    Polygon,
    PURPLE,
    RED,
    Rectangle,
    RegularPolygon,
    RIGHT,
    RoundedRectangle,
    Sector,
    Square,
    Star,
    TAU,
    TEAL,
    Text,
    Triangle,
    UL,
    UP,
    UR,
    VGroup,
    WHITE,
    YELLOW,
)
from .pipeline import Converter, Drawer, FrameBuffer, FrameGenerator, Renderer, render_objects_cache
from .gpu import GPUFrameGenerator
from .object_cache import ObjectCacheReader, ObjectCacheWriter, render_config_from_cache
from .scene import MovingCameraScene, Scene, ThreeDScene
from .geometry3d import (
    Camera3D, Mesh3D, Polyhedron, Cube, Cuboid, Sphere, Cylinder, Cone, Torus,
    Plane, Tetrahedron, Octahedron, Surface, Line3D, Group3D, ThreeDAxes,
    X_AXIS, Y_AXIS, Z_AXIS, IN, OUT,
)
from .renderer3d import Renderer3D
from .tex import MathTex, Tex, TexCompileError


__all__ = [name for name in globals() if not name.startswith("_")]

"""TeX-to-vector integration using LaTeX/XeLaTeX and dvisvgm."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .mobject import Mobject

_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_PATH_TOKEN = re.compile(rf"[AaCcHhLlMmQqSsTtVvZz]|{_NUMBER}")
_TRANSFORM = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")


class TexCompileError(RuntimeError):
    """Raised when the TeX toolchain cannot produce vector output."""


def _is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def _parse_path(data: str, curve_steps: int) -> tuple[list[np.ndarray], list[bool]]:
    tokens = _PATH_TOKEN.findall(data.replace(",", " "))
    paths: list[np.ndarray] = []
    closed: list[bool] = []
    points: list[tuple[float, float]] | None = None
    current = np.array([0.0, 0.0])
    start = current.copy()
    last_cubic = None
    last_quad = None
    previous_command = ""
    command = None
    index = 0

    def finish(is_closed=False):
        nonlocal points
        if points and len(points) >= 2:
            paths.append(np.asarray(points, dtype=np.float64))
            closed.append(is_closed)
        points = None

    def number() -> float:
        nonlocal index
        if index >= len(tokens) or _is_command(tokens[index]):
            raise TexCompileError(f"invalid SVG path near {data[:120]!r}")
        value = float(tokens[index])
        index += 1
        return value

    def pair(relative: bool) -> np.ndarray:
        value = np.array([number(), number()], dtype=np.float64)
        return current + value if relative else value

    def add_curve(samples):
        nonlocal current
        points.extend(tuple(p) for p in samples)
        current = np.asarray(samples[-1], dtype=np.float64)

    while index < len(tokens):
        if _is_command(tokens[index]):
            command = tokens[index]
            index += 1
            lower = command.lower()
            if lower == "z":
                finish(True)
                current = start.copy()
                last_cubic = last_quad = None
                previous_command = command
                command = None
                continue
        elif command is None:
            raise TexCompileError(f"missing SVG path command near {data[:120]!r}")

        relative = command.islower()
        lower = command.lower()
        if lower == "m":
            target = pair(relative)
            finish(False)
            current = target
            start = target.copy()
            points = [tuple(current)]
            command = "l" if relative else "L"
            last_cubic = last_quad = None
            previous_command = "m" if relative else "M"
            continue

        if points is None:
            points = [tuple(current)]

        if lower == "l":
            current = pair(relative)
            points.append(tuple(current))
            last_cubic = last_quad = None
        elif lower == "h":
            x = number()
            current = np.array([current[0] + x if relative else x, current[1]])
            points.append(tuple(current))
            last_cubic = last_quad = None
        elif lower == "v":
            y = number()
            current = np.array([current[0], current[1] + y if relative else y])
            points.append(tuple(current))
            last_cubic = last_quad = None
        elif lower == "c":
            p0 = current.copy()
            c1, c2, end = pair(relative), pair(relative), pair(relative)
            ts = np.linspace(0.0, 1.0, curve_steps + 1)[1:]
            samples = [((1-t)**3*p0 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*end) for t in ts]
            add_curve(samples)
            last_cubic, last_quad = c2, None
        elif lower == "s":
            p0 = current.copy()
            c1 = 2 * current - last_cubic if previous_command.lower() in {"c", "s"} and last_cubic is not None else current.copy()
            c2, end = pair(relative), pair(relative)
            ts = np.linspace(0.0, 1.0, curve_steps + 1)[1:]
            samples = [((1-t)**3*p0 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*end) for t in ts]
            add_curve(samples)
            last_cubic, last_quad = c2, None
        elif lower == "q":
            p0 = current.copy()
            control, end = pair(relative), pair(relative)
            ts = np.linspace(0.0, 1.0, curve_steps + 1)[1:]
            samples = [((1-t)**2*p0 + 2*(1-t)*t*control + t**2*end) for t in ts]
            add_curve(samples)
            last_quad, last_cubic = control, None
        elif lower == "t":
            p0 = current.copy()
            control = 2 * current - last_quad if previous_command.lower() in {"q", "t"} and last_quad is not None else current.copy()
            end = pair(relative)
            ts = np.linspace(0.0, 1.0, curve_steps + 1)[1:]
            samples = [((1-t)**2*p0 + 2*(1-t)*t*control + t**2*end) for t in ts]
            add_curve(samples)
            last_quad, last_cubic = control, None
        elif lower == "a":
            # dvisvgm font outlines are normally lines and Bézier curves. For
            # SVG arc commands retain the endpoint, keeping the path usable.
            number()  # rx
            number()  # ry
            number()  # x-axis rotation
            number()  # large-arc flag
            number()  # sweep flag
            current = pair(relative)
            points.append(tuple(current))
            last_cubic = last_quad = None
        else:
            raise TexCompileError(f"unsupported SVG path command {command!r}")
        previous_command = command

    finish(False)
    return paths, closed


def _transform_matrix(spec: str | None) -> np.ndarray:
    result = np.identity(3)
    if not spec:
        return result
    for name, raw_args in _TRANSFORM.findall(spec):
        args = [float(value) for value in re.findall(_NUMBER, raw_args)]
        matrix = np.identity(3)
        lower = name.casefold()
        if lower == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            matrix = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif lower == "translate" and args:
            matrix[0, 2] = args[0]
            matrix[1, 2] = args[1] if len(args) > 1 else 0.0
        elif lower == "scale" and args:
            matrix[0, 0] = args[0]
            matrix[1, 1] = args[1] if len(args) > 1 else args[0]
        elif lower == "rotate" and args:
            angle = math.radians(args[0])
            c, s = math.cos(angle), math.sin(angle)
            matrix[:2, :2] = [[c, -s], [s, c]]
            if len(args) >= 3:
                to_origin = np.identity(3)
                to_origin[:2, 2] = [-args[1], -args[2]]
                back = np.identity(3)
                back[:2, 2] = [args[1], args[2]]
                matrix = back @ matrix @ to_origin
        elif lower == "skewx" and args:
            matrix[0, 1] = math.tan(math.radians(args[0]))
        elif lower == "skewy" and args:
            matrix[1, 0] = math.tan(math.radians(args[0]))
        else:
            continue
        result = result @ matrix
    return result


def _svg_paths(svg_path: Path, curve_steps: int) -> tuple[list[np.ndarray], list[bool]]:
    try:
        root = ET.parse(svg_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise TexCompileError(f"dvisvgm did not produce valid SVG: {exc}") from exc

    all_paths: list[np.ndarray] = []
    all_closed: list[bool] = []
    excluded = {"defs", "clipPath", "mask", "pattern", "metadata", "title", "desc", "style"}

    def visit(node, parent_matrix, hidden=False):
        tag = node.tag.rsplit("}", 1)[-1]
        hidden = hidden or tag in excluded
        if hidden:
            return
        matrix = parent_matrix @ _transform_matrix(node.attrib.get("transform"))
        if tag == "path":
            contours, closures = _parse_path(node.attrib.get("d", ""), curve_steps)
            for contour, is_closed in zip(contours, closures):
                homogeneous = np.column_stack((contour, np.ones(len(contour))))
                points = (homogeneous @ matrix.T)[:, :2]
                all_paths.append(points)
                all_closed.append(is_closed)
        elif tag == "rect":
            x, y = float(node.attrib.get("x", 0)), float(node.attrib.get("y", 0))
            width, height = float(node.attrib.get("width", 0)), float(node.attrib.get("height", 0))
            points = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
            homogeneous = np.column_stack((points, np.ones(len(points))))
            all_paths.append((homogeneous @ matrix.T)[:, :2])
            all_closed.append(True)
        elif tag in {"polygon", "polyline"}:
            values = [float(value) for value in re.findall(_NUMBER, node.attrib.get("points", ""))]
            if len(values) >= 4 and len(values) % 2 == 0:
                points = np.asarray(values).reshape((-1, 2))
                homogeneous = np.column_stack((points, np.ones(len(points))))
                all_paths.append((homogeneous @ matrix.T)[:, :2])
                all_closed.append(tag == "polygon")
        elif tag == "use":
            raise TexCompileError("dvisvgm emitted an SVG <use>; use a newer dvisvgm with --no-fonts=1")
        for child in node:
            visit(child, matrix, hidden)

    visit(root, np.identity(3))
    if not all_paths:
        raise TexCompileError("TeX output contains no vector paths")
    return all_paths, all_closed


class Tex(Mobject):
    """A LaTeX text object converted to vector paths during scene conversion.

    ``tex_compiler`` may be ``auto``, a DVI/XDV engine, a PDF engine, Tectonic,
    latexmk, or a path to one of those executables.
    XeLaTeX supports system fonts through a preamble such as
    ``\\usepackage{fontspec}``.
    """

    def __init__(
        self,
        tex_string: str,
        *,
        font_size: float = 0.6,
        color="#ffffff",
        stroke_width: float = 0.65,
        preamble: str = "",
        tex_compiler: str = "auto",
        dvisvgm_path: str | None = None,
        timeout: float = 60.0,
        curve_steps: int = 12,
        math_mode: bool = False,
        **kwargs,
    ):
        super().__init__([], [], color=color, fill_opacity=1.0, stroke_width=stroke_width, name="MathTex" if math_mode else "Tex", **kwargs)
        self.tex_string = str(tex_string)
        self.font_size = float(font_size)
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")
        self.preamble = str(preamble)
        self.tex_compiler = str(tex_compiler)
        self.dvisvgm_path = str(dvisvgm_path) if dvisvgm_path else None
        self.tex_timeout = float(timeout)
        if self.tex_timeout <= 0:
            raise ValueError("timeout must be positive")
        self.curve_steps = max(3, int(curve_steps))
        self.math_mode = bool(math_mode)
        self._compiled = False

    @property
    def width(self) -> float:
        self._ensure_tex_compiled()
        return super().width

    @property
    def height(self) -> float:
        self._ensure_tex_compiled()
        return super().height

    def copy(self) -> "Tex":
        self._ensure_tex_compiled()
        return super().copy()

    def get_world_paths(self) -> list[np.ndarray]:
        self._ensure_tex_compiled()
        return super().get_world_paths()

    def _ensure_tex_compiled(self) -> None:
        if self._compiled:
            return
        compilers = self._resolve_compilers()
        dvisvgm_name = self.dvisvgm_path or "dvisvgm"
        dvisvgm = shutil.which(dvisvgm_name)
        if dvisvgm is None and Path(dvisvgm_name).is_file():
            dvisvgm = str(Path(dvisvgm_name).resolve())
        if dvisvgm is None:
            raise TexCompileError("dvisvgm not found. Install it with TeX Live or MiKTeX and add it to PATH.")

        content = rf"\[\displaystyle {self.tex_string}\]" if self.math_mode else self.tex_string
        source = "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage{amsmath,amssymb}",
                r"\pagestyle{empty}",
                self.preamble,
                r"\begin{document}",
                r"\noindent\leavevmode",
                content,
                r"\end{document}",
            ]
        )
        errors = []
        paths = closed = None
        for compiler, compiler_name in compilers:
            try:
                paths, closed = self._compile_svg(compiler, compiler_name, dvisvgm, source)
                break
            except TexCompileError as exc:
                errors.append(f"{compiler_name}: {exc}")
        if paths is None or closed is None:
            diagnostics = "\n\n".join(errors[-3:])
            raise TexCompileError(f"all available TeX toolchains failed:\n{diagnostics}")

        all_points = np.concatenate(paths)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        center = (minimum + maximum) / 2
        svg_height = max(float(maximum[1] - minimum[1]), 1e-9)
        scale = self.font_size / svg_height
        self.paths = [np.column_stack(((path[:, 0] - center[0]) * scale, (center[1] - path[:, 1]) * scale)) for path in paths]
        self.closed = closed
        self._compiled = True

    def _resolve_compilers(self) -> list[tuple[str, str]]:
        requested = self.tex_compiler
        if requested.casefold() == "auto":
            candidates = ["latex", "xelatex", "platex", "uplatex", "ptex", "uptex", "latexmk", "pdflatex", "lualatex", "tectonic"]
            if any(ord(char) > 127 for char in self.tex_string) or "fontspec" in self.preamble.casefold():
                candidates.remove("xelatex")
                candidates.insert(0, "xelatex")
                candidates.remove("uplatex")
                candidates.insert(1, "uplatex")
            available = []
            for candidate in candidates:
                resolved = shutil.which(candidate)
                if resolved:
                    available.append((resolved, candidate))
            if not available:
                raise TexCompileError("no TeX engine found; install TeX Live, MiKTeX, MacTeX, or Tectonic and add it to PATH")
            return available

        path = Path(requested)
        compiler = shutil.which(requested) or (str(path.resolve()) if path.is_file() else None)
        if compiler is None:
            raise TexCompileError(f"TeX compiler not found: {requested}")
        name = Path(requested).name.casefold().removesuffix(".exe")
        supported = {"latex", "xelatex", "platex", "uplatex", "ptex", "uptex", "latexmk", "pdflatex", "lualatex", "tectonic"}
        if name not in supported:
            raise TexCompileError("unsupported TeX engine; use auto, latex/xelatex, pTeX, latexmk, pdflatex/lualatex, or tectonic")
        return [(compiler, name)]

    def _compile_svg(self, compiler: str, compiler_name: str, dvisvgm: str, source: str):
        with tempfile.TemporaryDirectory(prefix="pyanimation-tex-") as temp_dir:
            directory = Path(temp_dir)
            tex_file = directory / "formula.tex"
            tex_file.write_text(source, encoding="utf-8")
            compiler_args = self._compiler_command(compiler, compiler_name, tex_file.name)
            try:
                compile_result = subprocess.run(
                    compiler_args,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.tex_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TexCompileError(f"compilation timed out after {self.tex_timeout:g}s") from exc
            except OSError as exc:
                raise TexCompileError(f"could not start compiler: {exc}") from exc
            if compile_result.returncode != 0:
                diagnostic = (compile_result.stdout + "\n" + compile_result.stderr)[-4000:]
                raise TexCompileError(f"compilation failed:\n{diagnostic}")

            extension = self._output_extension(compiler_name)
            intermediate = directory / f"formula.{extension}"
            if not intermediate.is_file():
                raise TexCompileError(f"compiler did not create {intermediate.suffix} output")
            svg_file = directory / "formula.svg"
            svg_command = [dvisvgm, "--no-fonts=1", "--exact-bbox", "--bbox=min", f"--output={svg_file.name}"]
            if extension == "pdf":
                svg_command.append("--pdf")
            svg_command.append(intermediate.name)
            try:
                svg_result = subprocess.run(
                    svg_command,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.tex_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TexCompileError(f"dvisvgm conversion timed out after {self.tex_timeout:g}s") from exc
            except OSError as exc:
                raise TexCompileError(f"could not start dvisvgm: {exc}") from exc
            if svg_result.returncode != 0 or not svg_file.is_file():
                diagnostic = (svg_result.stdout + "\n" + svg_result.stderr)[-4000:]
                raise TexCompileError(f"dvisvgm conversion failed:\n{diagnostic}")
            try:
                return _svg_paths(svg_file, self.curve_steps)
            except TexCompileError as exc:
                raise TexCompileError(str(exc)) from exc

    @staticmethod
    def _output_extension(compiler_name: str) -> str:
        if compiler_name in {"latex", "platex", "uplatex", "ptex", "uptex", "latexmk"}:
            return "dvi"
        if compiler_name == "xelatex":
            return "xdv"
        return "pdf"

    @staticmethod
    def _compiler_command(compiler: str, compiler_name: str, tex_filename: str) -> list[str]:
        if compiler_name == "tectonic":
            return [compiler, "--keep-logs", tex_filename]
        if compiler_name == "latexmk":
            return [compiler, "-dvi", "-interaction=nonstopmode", "-halt-on-error", tex_filename]
        command = [compiler, "-interaction=nonstopmode", "-halt-on-error"]
        if compiler_name == "xelatex":
            command.append("-no-pdf")
        command.append(tex_filename)
        return command


class MathTex(Tex):
    """A math-mode TeX object. Multiple parts are joined by ``separator``."""

    def __init__(self, *tex_strings: str, separator: str = r"\,", **kwargs):
        if not tex_strings:
            raise ValueError("MathTex needs at least one TeX string")
        super().__init__(separator.join(tex_strings), math_mode=True, **kwargs)

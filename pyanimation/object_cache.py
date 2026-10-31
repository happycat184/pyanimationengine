"""Versioned binary cache for converted objects and temporary raster frames."""

from __future__ import annotations

import os
import struct
import uuid
from pathlib import Path

import numpy as np

from .config import RenderConfig
from .pipeline import RenderObject
from .geometry3d import RenderMesh

_MAGIC = b"PYAO"
_VERSION = 4
_HEADER = struct.Struct("<4sHH5If3B")
_FRAME_HEADER = struct.Struct("<I")
_OBJECT_HEADER_V1 = struct.Struct("<ii6B7f4I")
_OBJECT_HEADER_V2 = struct.Struct("<ii6B7f5I")
_OBJECT_HEADER = struct.Struct("<ii6B7f6I")
_PATH_HEADER = struct.Struct("<IB3x")
_MESH_FLAG = struct.Struct("<B")
_MESH_HEADER = struct.Struct("<IIB3f")


def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated .objects cache")
    return data


class ObjectCacheWriter:
    """Writes frame snapshots as a compact little-endian binary stream."""

    def __init__(self, path: str | Path, config: RenderConfig):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        self.config = config
        self.frame_count = 0
        self._bitmap_ids: dict[bytes, int] = {}
        self._next_bitmap_id = 1
        self._stream = self.temp_path.open("w+b")
        self._write_header(0)

    def _write_header(self, frame_count: int) -> None:
        from .mobject import color_rgb

        rgb = color_rgb(self.config.background_color)
        self._stream.seek(0)
        self._stream.write(
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                0,
                self.config.width,
                self.config.height,
                self.config.fps,
                self.config.antialias,
                frame_count,
                self.config.frame_width,
                *rgb,
            )
        )
        self._stream.seek(0, os.SEEK_END)

    def write_frame(self, objects: list[RenderObject]) -> None:
        if self._stream.closed:
            raise RuntimeError("object cache writer is closed")
        self._stream.write(_FRAME_HEADER.pack(len(objects)))
        for obj in objects:
            font_bytes = (obj.text_font or "").encode("utf-8")
            text_bytes = (obj.text or "").encode("utf-8")
            bitmap_bytes = obj.text_bitmap or b""
            bitmap_id = 0
            bitmap_length = 0
            if bitmap_bytes:
                bitmap_id = self._bitmap_ids.get(bitmap_bytes, 0)
                if bitmap_id == 0:
                    bitmap_id = self._next_bitmap_id
                    self._next_bitmap_id += 1
                    self._bitmap_ids[bitmap_bytes] = bitmap_id
                    bitmap_length = len(bitmap_bytes)
            self._stream.write(
                _OBJECT_HEADER.pack(
                    int(obj.z_index),
                    int(obj.insertion_order),
                    *obj.fill_color,
                    *obj.stroke_color,
                    float(obj.fill_opacity),
                    float(obj.stroke_opacity),
                    float(obj.stroke_width),
                    float(obj.reveal),
                    float(obj.rotation),
                    float(obj.position[0]),
                    float(obj.position[1]),
                    len(obj.paths),
                    int(obj.text_size or 0),
                    len(font_bytes),
                    len(text_bytes),
                    bitmap_id,
                    bitmap_length,
                )
            )
            self._stream.write(font_bytes)
            self._stream.write(text_bytes)
            if bitmap_length:
                self._stream.write(bitmap_bytes)
            for points, closed in zip(obj.paths, obj.closed):
                points = np.asarray(points, dtype="<f4").reshape((-1, 2))
                self._stream.write(_PATH_HEADER.pack(len(points), int(bool(closed))))
                self._stream.write(points.tobytes(order="C"))
            self._stream.write(_MESH_FLAG.pack(int(obj.mesh is not None)))
            if obj.mesh is not None:
                mesh = obj.mesh
                self._stream.write(_MESH_HEADER.pack(len(mesh.vertices), len(mesh.faces), int(mesh.perspective),
                                                    mesh.focal_length, mesh.near, mesh.far))
                self._stream.write(np.asarray(mesh.vertices, dtype="<f4").tobytes())
                self._stream.write(np.asarray(mesh.faces, dtype="<u4").tobytes())
                self._stream.write(np.asarray(mesh.colors, dtype=np.uint8).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self._stream.closed:
            return
        self._write_header(self.frame_count)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        os.replace(self.temp_path, self.path)

    def abort(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        try:
            self.temp_path.unlink()
        except FileNotFoundError:
            pass


class ObjectCacheReader:
    """Reads the versioned .objects format one frame at a time."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = None
        self.frame_count = 0
        self.metadata: dict[str, object] = {}
        self._text_bitmaps: dict[int, bytes] = {}

    def __enter__(self) -> "ObjectCacheReader":
        self._stream = self.path.open("rb")
        values = _HEADER.unpack(_read_exact(self._stream, _HEADER.size))
        magic, version, _flags = values[:3]
        if magic != _MAGIC:
            raise ValueError("not a PyAnimationEngine .objects cache")
        if version not in {1, 2, 3, _VERSION}:
            raise ValueError(f"unsupported .objects cache version: {version}")
        self.version = version
        width, height, fps, antialias, self.frame_count, frame_width, *background = values[3:]
        self.metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "antialias": antialias,
            "frame_width": frame_width,
            "background_color": tuple(background),
            "version": version,
            "portable_text": version >= 2,
        }
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __iter__(self):
        if self._stream is None:
            raise RuntimeError("use ObjectCacheReader as a context manager")
        for frame_index in range(self.frame_count):
            object_count = _FRAME_HEADER.unpack(_read_exact(self._stream, _FRAME_HEADER.size))[0]
            objects = [self._read_object() for _ in range(object_count)]
            yield frame_index, objects

    def _read_object(self) -> RenderObject:
        if self.version >= 3:
            object_header = _OBJECT_HEADER
        elif self.version == 2:
            object_header = _OBJECT_HEADER_V2
        else:
            object_header = _OBJECT_HEADER_V1
        values = object_header.unpack(_read_exact(self._stream, object_header.size))
        z_index, insertion_order = values[:2]
        fill_color = tuple(values[2:5])
        stroke_color = tuple(values[5:8])
        floats = values[8:15]
        if self.version >= 3:
            path_count, text_size, font_length, text_length, bitmap_id, bitmap_length = values[15:21]
        elif self.version == 2:
            path_count, text_size, font_length, text_length, bitmap_length = values[15:20]
            bitmap_id = 0
        else:
            path_count, text_size, font_length, text_length = values[15:19]
            bitmap_id = 0
            bitmap_length = 0
        font = _read_exact(self._stream, font_length).decode("utf-8") if font_length else None
        text = _read_exact(self._stream, text_length).decode("utf-8") if text_length else None
        if bitmap_length:
            text_bitmap = _read_exact(self._stream, bitmap_length)
            if bitmap_id:
                self._text_bitmaps[bitmap_id] = text_bitmap
        elif bitmap_id:
            text_bitmap = self._text_bitmaps.get(bitmap_id)
            if text_bitmap is None:
                raise ValueError(f".objects cache references missing text bitmap {bitmap_id}")
        else:
            text_bitmap = None
        paths, closed = [], []
        for _ in range(path_count):
            point_count, is_closed = _PATH_HEADER.unpack(_read_exact(self._stream, _PATH_HEADER.size))
            raw = _read_exact(self._stream, point_count * 2 * 4)
            points = np.frombuffer(raw, dtype="<f4").reshape((point_count, 2)).copy()
            paths.append(points)
            closed.append(bool(is_closed))
        mesh = None
        if self.version >= 4:
            mesh_flag = _MESH_FLAG.unpack(_read_exact(self._stream, _MESH_FLAG.size))[0]
            if mesh_flag not in (0, 1):
                raise ValueError("invalid mesh record in .objects cache")
            if mesh_flag:
                nv, nf, perspective, focal, near, far = _MESH_HEADER.unpack(_read_exact(self._stream, _MESH_HEADER.size))
                vertices = np.frombuffer(_read_exact(self._stream, nv*12), dtype="<f4").reshape(nv,3).copy()
                faces = np.frombuffer(_read_exact(self._stream, nf*12), dtype="<u4").reshape(nf,3).copy()
                colors = np.frombuffer(_read_exact(self._stream, nf*3), dtype=np.uint8).reshape(nf,3).copy()
                if (not nv or not nf or faces.max() >= nv or not np.isfinite(vertices).all()
                        or not np.isfinite([focal,near,far]).all() or not 0 < near < far
                        or focal <= 0 or perspective not in (0,1)):
                    raise ValueError("invalid mesh geometry or camera in .objects cache")
                mesh = RenderMesh(vertices, faces, colors, bool(perspective), focal, near, far)
        return RenderObject(
            paths=paths,
            closed=closed,
            fill_color=fill_color,
            stroke_color=stroke_color,
            fill_opacity=floats[0],
            stroke_opacity=floats[1],
            stroke_width=floats[2],
            reveal=floats[3],
            rotation=floats[4],
            position=(floats[5], floats[6]),
            z_index=z_index,
            insertion_order=insertion_order,
            text=text,
            text_font=font,
            text_size=text_size or None,
            text_bitmap=text_bitmap,
            mesh=mesh,
        )


def render_config_from_cache(path: str | Path, *, codec="libx264", crf=18, preset="medium") -> RenderConfig:
    """Recreate the geometry-affecting render settings stored in a cache."""
    with ObjectCacheReader(path) as reader:
        metadata = reader.metadata.copy()
    background = "#" + "".join(f"{channel:02x}" for channel in metadata["background_color"])
    return RenderConfig(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        fps=int(metadata["fps"]),
        antialias=int(metadata["antialias"]),
        frame_width=float(metadata["frame_width"]),
        background_color=background,
        codec=codec,
        crf=crf,
        preset=preset,
    )


class FrameSequenceWriter:
    """Stores supersampled RGB frames temporarily before batch MP4 encoding."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = self.path.open("wb")
        self.frame_count = 0

    def __enter__(self) -> "FrameSequenceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def write(self, pixels: np.ndarray) -> None:
        self._stream.write(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()

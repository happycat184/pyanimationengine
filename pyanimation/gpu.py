"""Cross-vendor WebGPU compute rasterizer for converted scene frames."""

from __future__ import annotations

import math
import struct

import numpy as np
from PIL import Image

from .config import RenderConfig
from .mobject import color_rgb
from .pipeline import FrameBuffer, FrameGenerator, RenderObject, _decode_text_bitmap

_OBJECT_STRIDE = 20
_PATH_STRIDE = 4
_WORKGROUP = 8

_WGSL = r"""
@group(0) @binding(0) var<storage, read> object_data: array<u32>;
@group(0) @binding(1) var<storage, read> path_data: array<u32>;
@group(0) @binding(2) var<storage, read> point_data: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bitmap_data: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_pixels: array<u32>;
@group(0) @binding(6) var<storage, read> background_pixels: array<u32>;

fn as_f32(value: u32) -> f32 {
    return bitcast<f32>(value);
}

fn unpack_rgb(value: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(value & 255u),
        f32((value >> 8u) & 255u),
        f32((value >> 16u) & 255u),
    );
}

fn blend(dst: vec3<f32>, color: u32, opacity: f32) -> vec3<f32> {
    let alpha = clamp(opacity, 0.0, 1.0);
    return unpack_rgb(color) * alpha + dst * (1.0 - alpha);
}

fn segment_distance(point: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let delta = b - a;
    let denominator = dot(delta, delta);
    var t = 0.0;
    if denominator > 0.0 {
        t = clamp(dot(point - a, delta) / denominator, 0.0, 1.0);
    }
    return length(point - (a + t * delta));
}

@compute @workgroup_size(8, 8, 1)
fn render_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params[0];
    let height = params[1];
    let object_count = params[2];
    let background = params[3];
    if gid.x >= width || gid.y >= height {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    var color = unpack_rgb(background);
    if params[4] != 0u {
        color = unpack_rgb(background_pixels[gid.y * width + gid.x]);
    }
    for (var object_index = 0u; object_index < object_count; object_index = object_index + 1u) {
        let base = object_index * 20u;
        let flags = object_data[base];
        let path_start = object_data[base + 1u];
        let path_count = object_data[base + 2u];
        let fill_color = object_data[base + 3u];
        let stroke_color = object_data[base + 4u];
        let fill_opacity = as_f32(object_data[base + 5u]);
        let stroke_opacity = as_f32(object_data[base + 6u]);
        let stroke_width = as_f32(object_data[base + 7u]);
        let reveal = as_f32(object_data[base + 8u]);
        let bitmap_offset = object_data[base + 9u];
        let bitmap_width = object_data[base + 10u];
        let bitmap_height = object_data[base + 11u];
        let position = vec2<f32>(as_f32(object_data[base + 12u]), as_f32(object_data[base + 13u]));
        let bbox = vec4<f32>(
            as_f32(object_data[base + 14u]),
            as_f32(object_data[base + 15u]),
            as_f32(object_data[base + 16u]),
            as_f32(object_data[base + 17u]),
        );
        if pixel.x < bbox.x || pixel.y < bbox.y || pixel.x > bbox.z || pixel.y > bbox.w {
            continue;
        }

        if flags == 1u {
            let origin_x = i32(round(position.x - f32(bitmap_width) * 0.5));
            let origin_y = i32(round(position.y - f32(bitmap_height) * 0.5));
            let local_x = i32(gid.x) - origin_x;
            let local_y = i32(gid.y) - origin_y;
            if local_x >= 0 && local_y >= 0 && local_x < i32(bitmap_width) && local_y < i32(bitmap_height) {
                let alpha_index = u32(local_y) * bitmap_width + u32(local_x);
                let packed = bitmap_data[bitmap_offset + alpha_index / 4u];
                let shift = (alpha_index % 4u) * 8u;
                let coverage = f32((packed >> shift) & 255u) / 255.0;
                color = blend(color, fill_color, coverage * fill_opacity * reveal);
            }
            continue;
        }

        if fill_opacity > 0.0 && reveal >= 1.0 {
            var inside = false;
            for (var local_path = 0u; local_path < path_count; local_path = local_path + 1u) {
                let path_base = (path_start + local_path) * 4u;
                let first_point = path_data[path_base];
                let count = path_data[path_base + 1u];
                let is_closed = path_data[path_base + 2u] != 0u;
                if !is_closed || count < 3u {
                    continue;
                }
                var previous = point_data[first_point + count - 1u];
                for (var i = 0u; i < count; i = i + 1u) {
                    let current = point_data[first_point + i];
                    if (current.y > pixel.y) != (previous.y > pixel.y) {
                        let x_intersection = (previous.x - current.x) * (pixel.y - current.y) / (previous.y - current.y) + current.x;
                        if pixel.x < x_intersection {
                            inside = !inside;
                        }
                    }
                    previous = current;
                }
            }
            if inside {
                color = blend(color, fill_color, fill_opacity);
            }
        }

        if stroke_opacity > 0.0 && stroke_width > 0.0 {
            var coverage = 0.0;
            let radius = stroke_width * 0.5;
            for (var local_path = 0u; local_path < path_count; local_path = local_path + 1u) {
                let path_base = (path_start + local_path) * 4u;
                let first_point = path_data[path_base];
                let count = path_data[path_base + 1u];
                let is_closed = path_data[path_base + 2u] != 0u;
                if count < 2u {
                    continue;
                }
                for (var i = 0u; i + 1u < count; i = i + 1u) {
                    let distance = segment_distance(pixel, point_data[first_point + i], point_data[first_point + i + 1u]);
                    coverage = max(coverage, clamp(radius + 0.5 - distance, 0.0, 1.0));
                }
                if is_closed && reveal >= 1.0 {
                    let distance = segment_distance(pixel, point_data[first_point + count - 1u], point_data[first_point]);
                    coverage = max(coverage, clamp(radius + 0.5 - distance, 0.0, 1.0));
                }
            }
            if coverage > 0.0 {
                color = blend(color, stroke_color, coverage * stroke_opacity);
            }
        }
    }

    let red = u32(clamp(color.r, 0.0, 255.0));
    let green = u32(clamp(color.g, 0.0, 255.0));
    let blue = u32(clamp(color.b, 0.0, 255.0));
    output_pixels[gid.y * width + gid.x] = red | (green << 8u) | (blue << 16u) | 0xff000000u;
}
"""


def _float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _pack_rgb(color: tuple[int, int, int]) -> int:
    return int(color[0]) | (int(color[1]) << 8) | (int(color[2]) << 16)


def _adapter_rank(info: dict) -> int:
    device = str(info.get("device", "")).casefold()
    vendor = str(info.get("vendor", "")).casefold()
    device_type = str(info.get("adapter_type", info.get("device_type", ""))).casefold().replace("-", "")
    if "cpu" in device_type or any(name in device for name in ("software", "llvmpipe", "basic render driver")):
        return 0
    if any(name in device or name in vendor for name in ("nvidia", "amd", "radeon", "geforce")):
        return 4
    if "discretegpu" in device_type:
        return 3
    if "integratedgpu" in device_type:
        return 2
    return 1


class GPUFrameGenerator:
    """A WebGPU compute rasterizer; wgpu-native selects Vulkan/DX12/Metal."""

    def __init__(self, config: RenderConfig):
        try:
            import wgpu
        except ImportError as exc:
            raise RuntimeError("wgpu is not installed") from exc

        adapters = wgpu.gpu.enumerate_adapters_sync()
        ranked = sorted(((_adapter_rank(adapter.info), adapter) for adapter in adapters), key=lambda item: item[0], reverse=True)
        selected = next(((rank, adapter) for rank, adapter in ranked if rank > 0), None)
        if selected is None:
            raise RuntimeError("no hardware WebGPU adapter was found")
        self.adapter = selected[1]
        self.adapter_info = dict(self.adapter.info)
        self.device = self.adapter.request_device_sync()
        self.config = config
        module = self.device.create_shader_module(code=_WGSL)
        self.pipeline = self.device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={"module": module, "entry_point": "render_frame"},
        )
        self._wgpu = wgpu

    def _pack_frame(self, objects: list[RenderObject]):
        object_words: list[int] = []
        path_words: list[int] = []
        point_arrays: list[np.ndarray] = []
        point_count = 0
        bitmap_bytes = bytearray()
        bitmap_word_count = 0

        for obj in objects:
            if obj.text is not None and obj.text_bitmap:
                mask = _decode_text_bitmap(obj.text_bitmap)
                if obj.rotation:
                    mask = mask.rotate(math.degrees(obj.rotation), expand=True, resample=Image.Resampling.BICUBIC)
                bitmap = np.asarray(mask, dtype=np.uint8)
                bitmap_h, bitmap_w = bitmap.shape
                raw_bitmap = bitmap.tobytes(order="C")
                pad = (-len(raw_bitmap)) % 4
                bitmap_offset = bitmap_word_count
                bitmap_bytes.extend(raw_bitmap)
                if pad:
                    bitmap_bytes.extend(b"\x00" * pad)
                bitmap_word_count += (len(raw_bitmap) + pad) // 4
                origin_x = round(obj.position[0] - bitmap_w / 2)
                origin_y = round(obj.position[1] - bitmap_h / 2)
                bbox = (origin_x, origin_y, origin_x + bitmap_w, origin_y + bitmap_h)
                flags = 1
                path_start = 0
                path_count = 0
                fill_opacity = obj.fill_opacity * obj.reveal
                stroke_opacity = 0.0
                stroke_width = 0.0
            else:
                if obj.text is not None and obj.text:
                    raise RuntimeError("GPU rendering needs a portable text bitmap; regenerate this legacy cache")
                flags = 0
                path_start = len(path_words) // _PATH_STRIDE
                path_count = 0
                clipped_paths = []
                all_points = []
                for points, closed in zip(obj.paths, obj.closed):
                    points = np.asarray(points, dtype=np.float32).reshape((-1, 2))
                    if obj.reveal < 1.0:
                        points = FrameGenerator._trim(points, obj.reveal).astype(np.float32, copy=False)
                        closed = False
                    if len(points) == 0:
                        continue
                    point_start = point_count
                    clipped_paths.append((points, closed))
                    all_points.append(points)
                    path_words.extend((point_start, len(points), int(bool(closed)), 0))
                    point_count += len(points)
                    path_count += 1
                if clipped_paths:
                    for path_points, _ in clipped_paths:
                        point_arrays.append(path_points)
                    bbox_points = np.concatenate(all_points, axis=0)
                    padding = obj.stroke_width * 0.5 + 1.0
                    bbox = (
                        float(bbox_points[:, 0].min() - padding),
                        float(bbox_points[:, 1].min() - padding),
                        float(bbox_points[:, 0].max() + padding),
                        float(bbox_points[:, 1].max() + padding),
                    )
                else:
                    bbox = (0.0, 0.0, 0.0, 0.0)
                bitmap_offset = bitmap_w = bitmap_h = 0
                fill_opacity = obj.fill_opacity if obj.reveal >= 1.0 else 0.0
                stroke_opacity = obj.stroke_opacity
                stroke_width = obj.stroke_width

            object_words.extend((
                flags,
                path_start,
                path_count,
                _pack_rgb(obj.fill_color),
                _pack_rgb(obj.stroke_color),
                _float_word(fill_opacity),
                _float_word(stroke_opacity),
                _float_word(stroke_width),
                _float_word(obj.reveal),
                bitmap_offset,
                bitmap_w,
                bitmap_h,
                _float_word(obj.position[0]),
                _float_word(obj.position[1]),
                _float_word(bbox[0]),
                _float_word(bbox[1]),
                _float_word(bbox[2]),
                _float_word(bbox[3]),
                0,
                0,
            ))

        if point_arrays:
            points = np.concatenate(point_arrays, axis=0).astype("<f4", copy=False)
        else:
            points = np.zeros((1, 2), dtype="<f4")
        if not object_words:
            object_words = [0] * _OBJECT_STRIDE
        if not path_words:
            path_words = [0] * _PATH_STRIDE
        if not bitmap_bytes:
            bitmap_bytes.extend(b"\x00\x00\x00\x00")
        return (
            np.asarray(object_words, dtype="<u4"),
            np.asarray(path_words, dtype="<u4"),
            points,
            np.frombuffer(bitmap_bytes, dtype="<u4").copy(),
        )

    def generate(
        self,
        objects: list[RenderObject],
        initial_frame: FrameBuffer | None = None,
        *,
        presorted: bool = False,
    ) -> FrameBuffer:
        wgpu = self._wgpu
        c = self.config
        if not presorted:
            objects = sorted(objects, key=lambda obj: (obj.z_index, obj.insertion_order))
        packed_objects, packed_paths, packed_points, packed_bitmaps = self._pack_frame(objects)
        background = _pack_rgb(color_rgb(c.background_color))
        params = np.asarray([c.raster_width, c.raster_height, len(objects), background, int(initial_frame is not None)], dtype="<u4")
        if initial_frame is None:
            background_data = np.zeros(1, dtype="<u4")
        else:
            background_data = np.empty((c.raster_height, c.raster_width, 4), dtype=np.uint8)
            background_data[:, :, :3] = initial_frame.pixels
            background_data[:, :, 3] = 255
        usage = wgpu.BufferUsage.STORAGE

        object_buffer = self.device.create_buffer_with_data(label="scene objects", data=packed_objects, usage=usage)
        path_buffer = self.device.create_buffer_with_data(label="path descriptors", data=packed_paths, usage=usage)
        point_buffer = self.device.create_buffer_with_data(label="path points", data=packed_points, usage=usage)
        bitmap_buffer = self.device.create_buffer_with_data(label="text masks", data=packed_bitmaps, usage=usage)
        params_buffer = self.device.create_buffer_with_data(label="frame parameters", data=params, usage=usage)
        background_buffer = self.device.create_buffer_with_data(label="composite background", data=background_data, usage=usage)
        output_buffer = self.device.create_buffer(
            label="raster output",
            size=c.raster_width * c.raster_height * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": object_buffer},
                {"binding": 1, "resource": path_buffer},
                {"binding": 2, "resource": point_buffer},
                {"binding": 3, "resource": bitmap_buffer},
                {"binding": 4, "resource": params_buffer},
                {"binding": 5, "resource": output_buffer},
                {"binding": 6, "resource": background_buffer},
            ],
        )
        encoder = self.device.create_command_encoder(label="PyAnimation frame")
        compute_pass = encoder.begin_compute_pass(label="GPU rasterization")
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(
            (c.raster_width + _WORKGROUP - 1) // _WORKGROUP,
            (c.raster_height + _WORKGROUP - 1) // _WORKGROUP,
            1,
        )
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        raw = self.device.queue.read_buffer(output_buffer)
        rgba = np.frombuffer(raw, dtype="<u4").view(np.uint8).reshape((c.raster_height, c.raster_width, 4))
        frame = FrameBuffer(c.raster_width, c.raster_height, color_rgb(c.background_color))
        frame.pixels[:] = rgba[:, :, :3]
        for buffer in (object_buffer, path_buffer, point_buffer, bitmap_buffer, params_buffer, output_buffer, background_buffer):
            buffer.destroy()
        return frame

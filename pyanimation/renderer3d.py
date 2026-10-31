"""Low-level tiled triangle rasterizer, shared by CPU and WebGPU backends."""
from __future__ import annotations

import math

import numpy as np

from .pipeline import FrameBuffer
from .mobject import color_rgb

TILE_SIZE = 16


def _clip_polygon(polygon, plane):
    """Sutherland-Hodgman clipping against a view-space half plane."""
    result = []
    previous = polygon[-1]
    dp = np.dot(previous, plane[:3]) + plane[3]
    for current in polygon:
        dc = np.dot(current, plane[:3]) + plane[3]
        if (dc >= 0) != (dp >= 0):
            result.append(previous + (current - previous) * (dp / (dp - dc)))
        if dc >= 0:
            result.append(current)
        previous, dp = current, dc
    return result


class Renderer3D:
    """Clip, project, bin, depth-test and shade meshes into an RGB frame.

    Each z_index is a separate compositing layer. Meshes within that layer
    share depth; 2D objects are screen-space overlays, drawn after its meshes.
    Lighting is flat Lambert shading baked into portable mesh snapshots.
    """
    def __init__(self, config):
        """Create a renderer for the requested output and supersampling size."""
        self.config = config
        self._gpu_pipeline = None

    def prepare(self, objects):
        """Clip, project, triangulate clipped polygons, and build screen tiles."""
        c = self.config
        half_w, half_h = c.raster_width / 2, c.raster_height / 2
        result = []
        for obj in objects:
            mesh = obj.mesh
            if mesh is None or obj.fill_opacity <= 0 or obj.reveal <= 0:
                continue
            face_count = min(len(mesh.faces), int(math.ceil(len(mesh.faces) * obj.reveal)))
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            triangles = vertices[mesh.faces[:face_count]]
            if mesh.perspective:
                # A point survives the left/right plane when +/-x + z*sx >= 0.
                sx = half_w / (c.pixels_per_unit * mesh.focal_length)
                sy = half_h / (c.pixels_per_unit * mesh.focal_length)
                planes = [
                    (0, 0, 1, -mesh.near),
                    (0, 0, -1, mesh.far),
                    (1, 0, sx, 0),
                    (-1, 0, sx, 0),
                    (0, 1, sy, 0),
                    (0, -1, sy, 0),
                ]
            else:
                sx = half_w / c.pixels_per_unit
                sy = half_h / c.pixels_per_unit
                planes = [
                    (0, 0, 1, -mesh.near),
                    (0, 0, -1, mesh.far),
                    (1, 0, 0, sx),
                    (-1, 0, 0, sx),
                    (0, 1, 0, sy),
                    (0, -1, 0, sy),
                ]
            planes = np.asarray(planes)
            distances = triangles @ planes[:, :3].T + planes[:, 3]
            outside = (distances < 0).all(axis=1).any(axis=1)
            inside = (distances >= 0).all(axis=(1,2))
            clipped, colors = [], []
            for i in np.flatnonzero(~outside):
                if inside[i]:
                    clipped.append(triangles[i])
                    colors.append(mesh.colors[i])
                else:
                    polygon = list(triangles[i])
                    for plane in planes:
                        polygon = _clip_polygon(polygon, plane)
                        if len(polygon) < 3:
                            break
                    for j in range(1, len(polygon) - 1):
                        clipped.append((polygon[0], polygon[j], polygon[j + 1]))
                        colors.append(mesh.colors[i])
            if not clipped:
                continue
            xyz = np.asarray(clipped)
            projected = np.zeros((len(xyz), 4, 4), dtype=np.float32)
            multiplier = c.pixels_per_unit * (mesh.focal_length / xyz[:,:,2] if mesh.perspective else 1.0)
            projected[:,:3,0] = half_w + xyz[:,:,0] * multiplier
            projected[:,:3,1] = half_h - xyz[:,:,1] * multiplier
            projected[:,:3,2] = 1.0/xyz[:,:,2] if mesh.perspective else -xyz[:,:,2]
            # Fixed subpixel coordinates make shared edge ownership repeatable.
            projected[:,:3,:2] = np.round(projected[:,:3,:2] * 256) / 256
            projected[:,3,:3] = colors
            projected[:,3,3] = obj.fill_opacity
            a, b, d = projected[:,0], projected[:,1], projected[:,2]
            areas = (b[:,0]-a[:,0])*(d[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(d[:,0]-a[:,0])
            negative = areas < 0
            projected[negative,1:3] = projected[negative,1:3][:,::-1]
            result.append(projected[np.abs(areas) > 1e-8])
        triangles = np.concatenate(result) if result else np.empty((0,4,4), dtype=np.float32)
        columns = (c.raster_width + TILE_SIZE - 1) // TILE_SIZE
        rows = (c.raster_height + TILE_SIZE - 1) // TILE_SIZE
        bins = [[] for _ in range(columns * rows)]
        for index, triangle in enumerate(triangles):
            bounds_min = np.floor(triangle[:3, :2].min(axis=0) / TILE_SIZE).astype(int)
            bounds_max = np.floor(triangle[:3, :2].max(axis=0) / TILE_SIZE).astype(int)
            for y in range(max(0, bounds_min[1]), min(rows - 1, bounds_max[1]) + 1):
                for x in range(max(0, bounds_min[0]), min(columns - 1, bounds_max[0]) + 1):
                    bins[y * columns + x].append(index)
        return triangles, bins, columns

    @staticmethod
    def _sample(triangles, px, py):
        a, b, c = (triangles[:,i,:3] for i in range(3))
        weights = []
        covered = np.ones((len(triangles), len(px)), dtype=bool)
        for v, w in ((b,c),(c,a),(a,b)):
            dx, dy = (w[:,0]-v[:,0])[:,None], (w[:,1]-v[:,1])[:,None]
            edge = dx*(py-v[:,1,None]) - dy*(px-v[:,0,None])
            top_left = (dy < 0) | ((dy == 0) & (dx > 0))
            covered &= (edge > 0) | ((edge == 0) & top_left)
            weights.append(edge)
        area = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
        depth = (weights[0]*a[:,2,None] + weights[1]*b[:,2,None] + weights[2]*c[:,2,None]) / area[:,None]
        return np.where(covered, depth, -np.inf)

    def _cpu(self, triangles, bins, columns, frame):
        for tile, indices in enumerate(bins):
            if not indices:
                continue
            x0, y0 = (tile%columns)*TILE_SIZE, (tile//columns)*TILE_SIZE
            x1, y1 = min(x0+TILE_SIZE, frame.width), min(y0+TILE_SIZE, frame.height)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            px, py = xx.ravel().astype(np.float32)+0.5, yy.ravel().astype(np.float32)+0.5
            colors = frame.pixels[y0:y1,x0:x1].reshape(-1,3).astype(np.float32)
            nearest = np.full(len(px), -np.inf, dtype=np.float32)
            candidates = triangles[indices]
            opaque = candidates[candidates[:,3,3] >= 1.0]
            # Bound the temporary triangle x pixel array even for dense meshes.
            for start in range(0,len(opaque),128):
                batch = opaque[start:start+128]
                depths = self._sample(batch,px,py)
                winner = np.argmax(depths,axis=0)
                depth = depths[winner,np.arange(len(px))]
                visible = depth > nearest
                colors[visible] = batch[winner[visible],3,:3]
                nearest[visible] = depth[visible]
            transparent = candidates[candidates[:,3,3] < 1.0]
            if len(transparent):
                depths = self._sample(transparent,px,py)
                depths[depths < nearest] = -np.inf
                # Per-pixel sorting handles intersecting transparent surfaces.
                order = np.argsort(depths, axis=0, kind="stable")
                pixel_ids = np.arange(len(px))
                for ids in order:
                    valid = np.isfinite(depths[ids,pixel_ids])
                    alpha = np.where(valid,transparent[ids,3,3],0)[:,None]
                    colors = transparent[ids,3,:3]*alpha + colors*(1-alpha)
            frame.pixels[y0:y1,x0:x1] = np.clip(colors,0,255).astype(np.uint8).reshape(y1-y0,x1-x0,3)
        return frame

    def generate(self, objects, frame=None, gpu=None):
        c = self.config
        if frame is None:
            frame = FrameBuffer(c.raster_width,c.raster_height,color_rgb(c.background_color))
        triangles, bins, columns = self.prepare(objects)
        if not len(triangles):
            return frame
        if gpu is None:
            return self._cpu(triangles,bins,columns,frame)
        return self._gpu(triangles,bins,columns,frame,gpu)

    def _gpu(self, triangles, bins, columns, frame, gpu):
        device, wgpu = gpu.device, gpu._wgpu
        if self._gpu_pipeline is None:
            module = device.create_shader_module(code=_WGSL3D)
            self._gpu_pipeline = device.create_compute_pipeline(layout="auto",compute={"module":module,"entry_point":"render3d"})
        offsets = np.zeros(len(bins)+1,dtype="<u4")
        offsets[1:] = np.cumsum([len(b) for b in bins],dtype=np.uint32)
        ids = np.array([i for b in bins for i in b],dtype="<u4")
        rgba = np.empty((frame.height,frame.width,4),dtype=np.uint8)
        rgba[:,:,:3], rgba[:,:,3] = frame.pixels, 255
        params = np.array([frame.width,frame.height,columns,int(np.any(triangles[:,3,3]<1))],dtype="<u4")
        buffers = []
        try:
            for data in (triangles.astype("<f4"),offsets,ids if len(ids) else np.zeros(1,dtype="<u4"),rgba,params):
                buffers.append(device.create_buffer_with_data(data=data,usage=wgpu.BufferUsage.STORAGE))
            output = device.create_buffer(size=rgba.nbytes,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
            buffers.append(output)
            group = device.create_bind_group(layout=self._gpu_pipeline.get_bind_group_layout(0),
                entries=[{"binding":i,"resource":buffer} for i,buffer in enumerate(buffers)])
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self._gpu_pipeline)
            compute.set_bind_group(0,group)
            compute.dispatch_workgroups((frame.width+7)//8,(frame.height+7)//8,1)
            compute.end()
            device.queue.submit([encoder.finish()])
            raw = device.queue.read_buffer(output)
            frame.pixels[:] = np.frombuffer(raw,dtype=np.uint8).reshape(frame.height,frame.width,4)[:,:,:3]
        finally:
            for buffer in buffers:
                buffer.destroy()
        return frame


_WGSL3D = r"""
@group(0) @binding(0) var<storage, read> triangles: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> background: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_pixels: array<u32>;

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    let d = b-a;
    return d.x*(p.y-a.y)-d.y*(p.x-a.x);
}
fn owns(e: f32, a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = b-a;
    return e > 0.0 || (e == 0.0 && (d.y < 0.0 || (d.y == 0.0 && d.x > 0.0)));
}
fn depth_at(id: u32, p: vec2<f32>) -> f32 {
    let a = triangles[id*4u].xyz;
    let b = triangles[id*4u+1u].xyz;
    let c = triangles[id*4u+2u].xyz;
    let wa = edge(b.xy,c.xy,p);
    let wb = edge(c.xy,a.xy,p);
    let wc = edge(a.xy,b.xy,p);
    if !owns(wa,b.xy,c.xy) || !owns(wb,c.xy,a.xy) || !owns(wc,a.xy,b.xy) { return -1e30; }
    return (wa*a.z+wb*b.z+wc*c.z)/edge(a.xy,b.xy,c.xy);
}
fn unpack_rgb(v: u32) -> vec3<f32> {
    return vec3<f32>(f32(v&255u),f32((v>>8u)&255u),f32((v>>16u)&255u));
}
@compute @workgroup_size(8,8,1)
fn render3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params[0] || gid.y >= params[1] { return; }
    let pixel_id = gid.y*params[0]+gid.x;
    let tile = (gid.y/16u)*params[2]+gid.x/16u;
    let first = offsets[tile];
    let end = offsets[tile+1u];
    let p = vec2<f32>(f32(gid.x)+0.5,f32(gid.y)+0.5);
    var color = unpack_rgb(background[pixel_id]);
    var nearest = -1e30;
    for (var k=first; k<end; k=k+1u) {
        let id = indices[k];
        let material = triangles[id*4u+3u];
        if material.a < 1.0 { continue; }
        let depth = depth_at(id,p);
        if depth > nearest { nearest=depth; color=material.rgb; }
    }
    if params[3] != 0u {
        // Depth peeling in ascending reciprocal depth; no fixed layer limit.
        var previous = -1e30;
        var previous_id = 0u;
        for (var layer=first; layer<end; layer=layer+1u) {
            var next_depth = 1e30;
            var next_id = 0xffffffffu;
            for (var k=first; k<end; k=k+1u) {
                let id = indices[k];
                let material = triangles[id*4u+3u];
                if material.a >= 1.0 { continue; }
                let depth = depth_at(id,p);
                if depth <= -1e30 || depth < nearest { continue; }
                if depth < previous || (depth == previous && id <= previous_id) { continue; }
                if depth < next_depth || (depth == next_depth && id < next_id) { next_depth=depth; next_id=id; }
            }
            if next_id == 0xffffffffu { break; }
            let material = triangles[next_id*4u+3u];
            color = material.rgb*material.a+color*(1.0-material.a);
            previous=next_depth; previous_id=next_id;
        }
    }
    let rgb = vec3<u32>(clamp(color,vec3<f32>(0.0),vec3<f32>(255.0)));
    output_pixels[pixel_id] = rgb.r|(rgb.g<<8u)|(rgb.b<<16u)|0xff000000u;
}
"""

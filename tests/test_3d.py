"""Regression checks for geometry, clipping, depth, cache and mixed rendering."""
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyanimation import (
    Camera3D, Converter, RenderConfig, Renderer, Cube, Cuboid, Sphere, Cylinder,
    Cone, Torus, Plane, Tetrahedron, Octahedron, Surface, Line3D, Group3D,
    Text, Rectangle, ThreeDScene, Scene, Rotate, Transform, GrowFromCenter,
    ShrinkToCenter, Indicate, ObjectCacheWriter, ObjectCacheReader, X_AXIS,
    Y_AXIS, Z_AXIS, linear,
)
from pyanimation.geometry3d import RenderMesh, rotation_matrix
from pyanimation.pipeline import ProgressReporter


def configuration(device="cpu", workers=1):
    return RenderConfig(width=96, height=64, antialias=1, frame_width=6,
                        background_color="#000000", device=device, workers=workers)


def renderer(device="cpu", workers=1):
    r = Renderer(configuration(device, workers))
    r.converter.camera = Camera3D(position=(0,0,6), ambient=1)
    return r


class GeometryTests(unittest.TestCase):
    def test_solids_have_outward_nondegenerate_faces(self):
        for shape in (Cube(), Cuboid(), Sphere(), Cylinder(), Cone(), Tetrahedron(), Octahedron()):
            with self.subTest(shape=shape.name):
                tri = shape.get_world_vertices()[shape.faces]
                normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                self.assertTrue((np.linalg.norm(normal,axis=1)>1e-8).all())
                self.assertTrue((np.sum(normal*tri.mean(axis=1),axis=1)>0).all())

    def test_torus_surface_and_line(self):
        torus = Torus()
        self.assertEqual(len(torus.faces),32*12*2)
        surface = Surface(lambda u,v:(u,v,u*v), resolution=(3,4))
        self.assertEqual(len(surface.faces),24)
        line = Line3D((1,2,3),(1,-2,3))
        np.testing.assert_allclose(line.get_center(),(1,0,3))
        self.assertAlmostEqual(line.height,4)

    def test_rotation_and_group_pivot(self):
        np.testing.assert_allclose(rotation_matrix(np.pi/2,Z_AXIS)@X_AXIS,Y_AXIS,atol=1e-12)
        cube = Cube().move_to((2,0,1))
        Group3D(cube).rotate(np.pi/2,about_point=(0,0,1),axis=Z_AXIS)
        np.testing.assert_allclose(cube.get_center(),(0,2,1),atol=1e-12)
        self.assertEqual(Group3D().get_center(),(0,0,0))

    def test_camera_rejects_invalid_basis_and_clipping(self):
        for kwargs in ({"position":(0,0,0)}, {"up":(0,0,0)}, {"near":-1},
                       {"far":0.01}, {"projection":"invalid"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Camera3D(**kwargs)

    def test_animation_preserves_all_three_dimensions(self):
        cube, scene = Cube(), Scene(configuration())
        rotate = Rotate(cube, angle=2*np.pi, axis=Y_AXIS, rate_func=linear)
        rotate.begin(scene)
        rotate.interpolate(0.25)
        np.testing.assert_allclose(cube.orientation, rotation_matrix(np.pi/2,Y_AXIS))
        rotate.interpolate(1)
        np.testing.assert_allclose(cube.orientation,np.eye(3),atol=1e-12)
        grow = GrowFromCenter(cube, rate_func=linear)
        grow.begin(scene)
        grow.interpolate(0.5)
        np.testing.assert_allclose((cube.width,cube.height,cube.depth),(1,1,1),atol=1e-12)
        grow.interpolate(1)
        for animation in (ShrinkToCenter(cube), Indicate(cube)):
            animation.begin(scene)
            animation.interpolate(0.4)
            self.assertAlmostEqual(cube.scale_x,cube.scale_z)

    def test_mesh_transform_and_incompatible_topology(self):
        cube, scene = Cube(), Scene(configuration())
        target = cube.copy().shift((0,0,4))
        anim = Transform(cube,target,rate_func=linear)
        anim.begin(scene)
        anim.interpolate(0.5)
        np.testing.assert_allclose(cube.get_world_vertices(),target.vertices+(0,0,2))
        anim.interpolate(1)
        np.testing.assert_allclose(cube.get_world_vertices(),target.get_world_vertices())
        with self.assertRaisesRegex(ValueError,"topology"):
            Transform(cube,Sphere()).begin(scene)


class RasterTests(unittest.TestCase):
    def test_occlusion_does_not_depend_on_submission_order(self):
        r = renderer()
        near = Cube(2,color="#ff0000").move_to((0,0,1))
        far = Sphere(1.2,color="#0000ff").move_to((0,0,-1))
        a, b = r.render([near,far]).pixels, r.render([far,near]).pixels
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[32,48],(255,0,0))

    def test_intersecting_triangles_use_pixel_depth(self):
        r = renderer()
        a = r.converter.convert(Plane(color="#ff0000"))
        b = r.converter.convert(Plane(color="#0000ff"))
        # Same projected coverage; front/back order reverses across the image.
        xy = np.array([[-2,-1.5],[2,-1.5],[0,1.5]])
        a.mesh = RenderMesh(np.column_stack((xy,[2,6,4])),np.array([[0,1,2]]),np.array([[255,0,0]]),False,6,.1,20)
        b.mesh = RenderMesh(np.column_stack((xy,[6,2,4])),np.array([[0,1,2]]),np.array([[0,0,255]]),False,6,.1,20)
        frame = r._generate([a,b]).pixels
        np.testing.assert_array_equal(frame[40,32],(255,0,0))
        np.testing.assert_array_equal(frame[40,64],(0,0,255))

    def test_transparent_shared_edge_is_blended_once(self):
        r = renderer()
        r.converter.camera.projection = "orthographic"
        frame = r.render([Plane(2,2,color="#ffffff",fill_opacity=.5)]).pixels
        np.testing.assert_array_equal(frame[18:46,34:62],np.full((28,28,3),127,dtype=np.uint8))

    def test_near_clipping_and_behind_camera(self):
        r = renderer()
        obj = r.converter.convert(Plane())
        obj.mesh = RenderMesh(np.array([[-.1,-.1,.05],[.5,-.5,1],[0,.5,1]]),np.array([[0,1,2]]),np.array([[255,255,255]]),True,1,.1,10)
        tri, _, _ = r.renderer3d.prepare([obj])
        self.assertGreater(len(tri),0)
        self.assertTrue(np.isfinite(tri).all())
        self.assertTrue((tri[:,:3,2] <= 10.0001).all())
        obj.mesh.vertices[:,2] = -1
        tri, _, _ = r.renderer3d.prepare([obj])
        self.assertEqual(len(tri),0)

    def test_2d_layer_over_and_under_3d(self):
        r = renderer()
        mesh = Cube(color="#ff0000")
        overlay = Rectangle(1,1,color="#00ff00",fill_opacity=1,stroke_width=0)
        np.testing.assert_array_equal(r.render([overlay,mesh]).pixels[32,48],(0,255,0))
        overlay.set_z_index(-1)
        np.testing.assert_array_equal(r.render([mesh,overlay]).pixels[32,48],(255,0,0))

    def test_perspective_scales_with_distance(self):
        r = renderer()
        r.converter.camera = Camera3D(position=(0,0,0),target=(0,0,1),ambient=1)
        near = r.converter.convert(Plane(2,2).move_to((0,0,2)))
        far = r.converter.convert(Plane(2,2).move_to((0,0,4)))
        a = r.renderer3d.prepare([near])[0][:,:3,0]
        b = r.renderer3d.prepare([far])[0][:,:3,0]
        self.assertAlmostEqual(np.ptp(a),2*np.ptp(b))

    def test_cache_roundtrip_and_parallel_frames(self):
        r = renderer()
        objects = [r.converter.convert(Cube()),r.converter.convert(Text("portable",font_size=.3).set_z_index(10))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mixed.objects"
            writer = ObjectCacheWriter(path,r.config)
            for _ in range(4):
                writer.write_frame(objects)
            writer.close()
            with ObjectCacheReader(path) as reader:
                frames = list(reader)
                self.assertEqual(reader.version,4)
            recovered = frames[0][1]
            np.testing.assert_array_equal(objects[0].mesh.faces,recovered[0].mesh.faces)
            np.testing.assert_array_equal(r._generate(objects).pixels,r._generate(recovered).pixels)
            single, parallel = Path(directory)/"single.frames", Path(directory)/"parallel.frames"
            r.render_cache(path,single)
            renderer(workers=16).render_cache(path,parallel)
            self.assertEqual(single.read_bytes(),parallel.read_bytes())

    def test_legacy_v3_cache_is_readable(self):
        r = renderer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.objects"
            writer = ObjectCacheWriter(path,r.config)
            writer.write_frame([r.converter.convert(Rectangle())])
            writer.close()
            raw = bytearray(path.read_bytes())
            raw[4:6] = (3).to_bytes(2,"little")
            del raw[-1]  # only v4 adds the final no-mesh flag
            path.write_bytes(raw)
            with ObjectCacheReader(path) as reader:
                self.assertIsNone(list(reader)[0][1][0].mesh)

    def test_progress_prefix(self):
        stream = io.StringIO()
        r = renderer()
        r.render([Cube()],progress=ProgressReporter(1,stream=stream))
        self.assertIn("[INFO 3D Calculating] 1/1",stream.getvalue())
        self.assertIn("[INFO 3D In Rendering] 1/1",stream.getvalue())


class GPUParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import/adapter absence is a skip; shader or rendering errors are failures.
        try:
            import wgpu
        except ImportError:
            raise unittest.SkipTest("wgpu is not installed")
        adapters = wgpu.gpu.enumerate_adapters_sync()
        from pyanimation.gpu import _adapter_rank
        if not any(_adapter_rank(a.info)>0 for a in adapters):
            raise unittest.SkipTest("no hardware GPU available")
        cls.gpu = renderer("gpu")

    def test_gpu_matches_cpu_for_mixed_opaque_and_transparent_meshes(self):
        objects = [Cube(color="#aa44ff").rotate(.43,axis=Y_AXIS),
                   Torus(.9,.25,resolution=(12,8),color="#44ffaa",fill_opacity=.55).rotate(.8,axis=X_AXIS),
                   Text("HUD",font_size=.4).move_to((0,1.6)).set_z_index(5)]
        cpu = renderer().render(objects).pixels
        gpu = self.gpu.render(objects).pixels
        difference = np.abs(cpu.astype(int)-gpu.astype(int))
        self.assertLessEqual(difference.max(),2)
        self.assertLess(difference.mean(),.05)


if __name__ == "__main__":
    unittest.main()

"""Run: python examples/three_d.py --width 960 --height 540 --fps 30

Six generated solids, three-axis rotation, real depth overlap, and a fixed
2D title. The default device is auto (hardware GPU, then CPU fallback).
"""
from pathlib import Path
import argparse
import math
import sys
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyanimation import (
    BLUE, GREEN, ORANGE, PURPLE, TEAL, YELLOW, Cube, Sphere, Cylinder, Cone,
    Torus, Octahedron, ThreeDScene, Text, RoundedRectangle, FadeIn, Create,
    Rotate, RenderConfig, X_AXIS, Y_AXIS, Z_AXIS, linear,
)
from pyanimation.fonts import resolve_font_file


class ThreeDGallery(ThreeDScene):
    def setup(self):
        # Pick an installed scalable font so the labels retain their size on
        # Windows (where DejaVuSans.ttf is usually not installed) and Linux.
        if self.config.font_path is None:
            for family in ("Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"):
                path = resolve_font_file(family)
                try:
                    ImageFont.truetype(path, 24)
                except OSError:
                    continue
                self.config.font_path = path
                break

    def construct(self):
        self.set_camera(position=(0, 3, 14), target=(0, 0, 0), ambient=0.32)
        panel = RoundedRectangle(width=12.8, height=0.85, corner_radius=0.15,
                                 color="#202c42", fill_opacity=1, stroke_width=0)
        panel.move_to((0, 3.25)).set_z_index(10)
        title = Text("3D geometry  |  2D overlay", font_size=0.43).move_to((0, 3.25)).set_z_index(11)
        footer = Text("Triangle meshes / depth buffer / perspective / lighting", font_size=0.23,
                      color="#aebed3").move_to((0, -3.45)).set_z_index(11)
        self.add(panel, title, footer)

        solids = [
            Cube(1.45, color=BLUE).rotate(0.35, axis=X_AXIS),
            Sphere(0.9, resolution=(12,24), color=TEAL),
            Torus(0.68, 0.27, resolution=(28,12), color=ORANGE).rotate(0.65, axis=X_AXIS),
            Cylinder(0.68, 1.5, resolution=24, color=PURPLE),
            Cone(0.85, 1.65, resolution=24, color=GREEN),
            Octahedron(1.05, color=YELLOW),
        ]
        positions = [(-4,1.25,0),(0,1.25,0),(4,1.25,0),(-4,-1.6,0),(0,-1.6,0),(4,-1.6,0)]
        names = ("Cube", "Sphere", "Torus", "Cylinder", "Cone", "Octahedron")
        for solid, position, name in zip(solids, positions, names):
            solid.move_to(position)
            label_y = -0.05 if position[1] > 0 else -2.9
            self.add(Text(name, font_size=0.25, color="#b8c7dc").move_to((position[0], label_y)).set_z_index(10))

        self.play(*(Create(solid) for solid in solids), run_time=1.25)
        self.play(*(Rotate(solid, angle=2*math.pi, axis=(1,1,0.4), rate_func=linear)
                    for solid in solids), run_time=4)
        # This smaller sphere overlaps the cube and is correctly hidden by it.
        satellite = Sphere(0.42, resolution=(10,20), color=YELLOW).move_to((-4,1.25,1.5))
        self.play(FadeIn(satellite), run_time=0.5)
        self.play(Rotate(satellite, angle=2*math.pi, axis=Y_AXIS,
                         about_point=(-4,1.25,0), rate_func=linear),
                  Rotate(solids[2], angle=2*math.pi, axis=Z_AXIS, rate_func=linear), run_time=3)
        self.wait(0.75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--antialias", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("-o", "--output", default="three_d.mp4")
    args = parser.parse_args()
    config = RenderConfig(width=args.width, height=args.height, fps=args.fps,
                          antialias=args.antialias, workers=args.workers, device=args.device)
    ThreeDGallery(config).render(args.output)

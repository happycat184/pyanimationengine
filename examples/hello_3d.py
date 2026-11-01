"""A short, editable 3D scene. Run with: python examples/hello_3d.py"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyanimation import BLUE, ORANGE, PI, Create, Cube, Rotate, Sphere, Text, ThreeDScene, Y_AXIS


class Hello3D(ThreeDScene):
    def construct(self):
        self.set_camera(position=(5, 4, 9), target=(0, 0, 0))

        cube = Cube(1.8, color=BLUE).move_to((-1.2, 0, 0))
        ball = Sphere(1.0, color=ORANGE).move_to((1.2, 0, 0))
        caption = Text("My first 3D scene", font_size=0.45).move_to((0, 3.1))
        self.add_fixed_in_frame_mobjects(caption)

        self.play(Create(cube), Create(ball), run_time=1.5)
        self.play(
            Rotate(cube, angle=2*PI, axis=Y_AXIS),
            Rotate(ball, angle=2*PI, axis=Y_AXIS),
            run_time=3,
        )
        self.wait(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="preview frames live without writing an MP4")
    parser.add_argument("-o", "--output", help="output MP4 path")
    args = parser.parse_args()
    if args.online and args.output:
        parser.error("--output cannot be used with --online")
    if args.online:
        Hello3D().render_online()
    else:
        Hello3D().render(args.output or "hello_3d.mp4")

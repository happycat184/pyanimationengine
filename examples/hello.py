from pathlib import Path
import sys

# Let this example run directly from the repository without pip installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyanimation import BLUE, GREEN, Circle, Create, FadeIn, Rectangle, Scene, Text
class Hello(Scene):
    def construct(self):
        cir = Circle(radius=1.05, color=BLUE)
        self.play(Create(cir), run_time=2)
        self.wait(2)
hello = Hello()
hello.render("Hello.mp4")

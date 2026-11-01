from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

from .config import RenderConfig
from .object_cache import render_config_from_cache
from .pipeline import render_objects_cache
from .scene import Scene


def _load_scene(source: Path, name: str | None):
    spec = importlib.util.spec_from_file_location("user_animation_scene", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scene file: {source}")
    module = importlib.util.module_from_spec(spec)
    # inspect.getfile(SceneClass) needs the module to recover the script stem
    # for cache/<script>.objects, including when launched through this CLI.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scenes = [obj for _, obj in inspect.getmembers(module, inspect.isclass) if issubclass(obj, Scene) and obj is not Scene and obj.__module__ == module.__name__]
    if name:
        for scene_class in scenes:
            if scene_class.__name__ == name:
                return scene_class
        raise ValueError(f"scene {name!r} was not found in {source}")
    if len(scenes) != 1:
        choices = ", ".join(cls.__name__ for cls in scenes) or "(none found)"
        raise ValueError(f"select a scene class with a name; available: {choices}")
    return scenes[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render a Python animation scene to MP4 or a live window")
    parser.add_argument("source", nargs="?", type=Path, help="Python file defining a Scene subclass")
    parser.add_argument("scene", nargs="?", help="Scene class name (optional when the file has one scene)")
    parser.add_argument("-o", "--output", type=Path, help="output MP4 path")
    parser.add_argument("--objects-cache", type=Path, help="render a portable .objects cache without the Python scene")
    parser.add_argument("--online", action="store_true", help="show frames live in a window; do not create an MP4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--antialias", type=int, default=2, help="supersampling factor")
    parser.add_argument("--frame-width", type=float, default=14.2222222222)
    parser.add_argument("--background", default="#10131c")
    parser.add_argument("--font", help="default font family or font file for Text objects")
    parser.add_argument("--workers", type=int, default=16, help="parallel worker threads (default: 16; 0 selects up to 4 automatically)")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto", help="GPU backend selection; auto prefers a hardware GPU and falls back to CPU")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args(argv)
    if args.online and args.output:
        parser.error("--output cannot be used with --online; no MP4 is written")

    if args.objects_cache:
        if args.source or args.scene:
            parser.error("--objects-cache cannot be combined with a Python scene source")
        config = render_config_from_cache(args.objects_cache, crf=args.crf, preset=args.preset)
        config.workers = args.workers
        config.device = args.device
        if args.online:
            render_objects_cache(args.objects_cache, config=config, online=True)
            print("Online cache preview finished")
        else:
            output = args.output or args.objects_cache.with_suffix(".mp4")
            result = render_objects_cache(args.objects_cache, output, config)
            print(f"Rendered cache: {result.resolve()}")
        return 0
    if args.source is None:
        parser.error("provide a Python scene source or --objects-cache")

    config = RenderConfig(width=args.width, height=args.height, fps=args.fps, antialias=args.antialias, frame_width=args.frame_width, background_color=args.background, crf=args.crf, preset=args.preset, font_path=args.font, workers=args.workers, device=args.device)
    scene_class = _load_scene(args.source.resolve(), args.scene)
    scene = scene_class(config=config)
    output = args.output or Path(f"{scene_class.__name__.lower()}.mp4")
    result = scene.render(output, online=args.online)
    if args.online:
        print("Online preview finished; no MP4 was written")
    else:
        print(f"Rendered: {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

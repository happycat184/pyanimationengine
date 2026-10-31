"""Font family and font-file resolution for Pillow text objects."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _windows_font_family(spec: str) -> str | None:
    try:
        import winreg
    except ImportError:
        return None

    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
    ]
    wanted = _key(spec)
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    local_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    partial_match = None
    for root, key_path in roots:
        try:
            with winreg.OpenKey(root, key_path) as key:
                index = 0
                while True:
                    try:
                        display_name, filename, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    index += 1
                    display_family = _key(display_name.split("(", 1)[0].strip())
                    if display_family != wanted and not display_family.startswith(wanted):
                        continue
                    for folder in (windows_dir / "Fonts", local_fonts):
                        candidate = folder / filename
                        if candidate.is_file():
                            if display_family == wanted:
                                return str(candidate)
                            if partial_match is None:
                                partial_match = str(candidate)
                    candidate = Path(filename)
                    if candidate.is_file():
                        if display_family == wanted:
                            return str(candidate.resolve())
                        if partial_match is None:
                            partial_match = str(candidate.resolve())
        except OSError:
            continue
    if partial_match:
        return partial_match

    font_dir = windows_dir / "Fonts"
    if font_dir.is_dir():
        for candidate in font_dir.iterdir():
            if candidate.suffix.casefold() in {".ttf", ".otf", ".ttc"} and wanted in _key(candidate.stem):
                return str(candidate)
    return None


@lru_cache(maxsize=128)
def resolve_font_file(font_spec: str) -> str:
    """Resolve an explicit font path or an installed family name to a file."""
    spec = str(font_spec).strip()
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    if os.name == "nt":
        resolved = _windows_font_family(spec)
        if resolved:
            return resolved
    else:
        matcher = shutil.which("fc-match")
        if matcher:
            try:
                result = subprocess.run(
                    [matcher, "-f", "%{file}", spec],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                match = result.stdout.strip()
                if result.returncode == 0 and match and Path(match).is_file():
                    return match
            except (OSError, subprocess.TimeoutExpired):
                pass
    # Pillow can also resolve standard font names on some platforms.
    return spec


def load_pillow_font(font_spec: str, size: int, *, required: bool = False):
    """Load a requested font, or try common sans-serif families portably."""
    families = [font_spec]
    if not required:
        families.extend(("Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans", "sans-serif"))

    tried = set()
    for family in families:
        font_file = resolve_font_file(family)
        if font_file in tried:
            continue
        tried.add(font_file)
        try:
            return ImageFont.truetype(font_file, size)
        except OSError:
            continue

    if required:
        raise RuntimeError(f"could not load requested font {font_spec!r}")

    try:
        # Newer Pillow versions scale the built-in bitmap font to the request.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

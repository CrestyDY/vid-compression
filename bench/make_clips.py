"""Generate the standard benchmark clips.

Every results document is measured on these exact five clips, so numbers stay comparable
across milestones. They are regenerated rather than committed — deterministic seeds mean
you get byte-identical files, so `docs/m1-results.md`'s figures are reproducible from a
clean checkout.

    uv run python bench/make_clips.py

Each clip isolates one case:

    screen      near-static UI, tiny sparse changes      PIXEL_DELTA's best case
    slide       a large rigid object translating         motion's best case, delta's worst
    cut         three unrelated scenes spliced           FULL's case
    noise       real-ish video with sensor-like grain    everyone's worst case
    mixed       four quadrants wanting four answers      per-tile selection's case

They are encoded with `libx264rgb -qp 0`, which is lossless *in RGB*. Ordinary
`libx264 -qp 0` converts to YUV first and that conversion rounds, sprinkling ±1 noise
everywhere and making a "static" clip look completely changed.

The `-preset veryslow` is load-bearing for reproducibility: a different preset produces a
different (still lossless) bitstream, so the clip bytes — and therefore the source-file
sizes quoted in the results docs — would change.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H, FPS = 320, 240, 25
OUT = Path(__file__).resolve().parent / "clips"


def write_clip(path: Path, frames: list[np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    err = proc.stderr.read() if proc.stderr else b""
    if proc.wait() != 0:
        raise RuntimeError(f"encode failed:\n{err.decode(errors='replace')}")
    return path


def gradient() -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    f = np.zeros((H, W, 3), np.uint8)
    f[..., 0] = (xs * 255 // (W - 1)).astype(np.uint8)
    f[..., 1] = (ys * 255 // (H - 1)).astype(np.uint8)
    f[..., 2] = 96
    return f


def clip_screen(n: int = 60) -> list[np.ndarray]:
    """A UI: static chrome, text appearing a character at a time, a blinking cursor."""
    base = np.full((H, W, 3), 24, np.uint8)
    base[0:20, :] = (45, 45, 52)
    base[20:, 0:70] = (32, 32, 40)
    for i in range(5):
        base[30 + i * 22 : 44 + i * 22, 8:62] = (60, 60, 72)

    frames = []
    for k in range(n):
        f = base.copy()
        typed = min(k // 2, 20)
        for c in range(typed):
            f[80:90, 90 + c * 9 : 96 + c * 9] = (200, 200, 208)
        if (k // 10) % 2 == 0:
            f[78:92, 90 + typed * 9 : 92 + typed * 9] = (255, 255, 255)
        frames.append(f)
    return frames


def clip_slide(n: int = 60) -> list[np.ndarray]:
    """A textured rectangle translating 3px/frame over a textured background.

    Deliberately textured: a flat box on a flat background is trivially cheap for any
    scheme, which would flatter motion estimation unfairly.
    """
    rng = np.random.default_rng(4)
    bg = (rng.integers(0, 60, (H, W, 3)) + 30).astype(np.uint8)
    box = (rng.integers(120, 240, (80, 100, 3))).astype(np.uint8)

    frames = []
    for k in range(n):
        f = bg.copy()
        x = min(10 + 3 * k, W - 100)
        f[80:160, x : x + 100] = box
        frames.append(f)
    return frames


def clip_cut(n: int = 60) -> list[np.ndarray]:
    """Three unrelated scenes. Two hard cuts where nearly every pixel changes."""
    rng = np.random.default_rng(9)
    a = gradient()
    b = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    c = np.full((H, W, 3), 200, np.uint8)
    c[H // 3 : 2 * H // 3] = (20, 90, 160)

    third = n // 3
    return (
        [a.copy() for _ in range(third)]
        + [b.copy() for _ in range(third)]
        + [c.copy() for _ in range(n - 2 * third)]
    )


def clip_noise(n: int = 60) -> list[np.ndarray]:
    """A moving gradient plus low-amplitude per-pixel grain.

    The honest stand-in for camera footage: nothing is ever *exactly* unchanged, so the
    changed-pixel count sits near 100% no matter what actually moved.
    """
    rng = np.random.default_rng(17)
    base = gradient().astype(np.int16)
    frames = []
    for k in range(n):
        shifted = np.roll(base, k * 2, axis=1)
        grain = rng.integers(-4, 5, (H, W, 3))
        frames.append(np.clip(shifted + grain, 0, 255).astype(np.uint8))
    return frames


def clip_mixed(n: int = 60) -> list[np.ndarray]:
    """Four quadrants that each want a *different* representation.

        top-left      static UI chrome, a few pixels blink   -> PIXEL_DELTA
        top-right     a textured block sliding 3px/frame     -> MOTION
        bottom-left   completely unchanged                   -> anything, near-free
        bottom-right  abrupt whole-region repaints           -> BLOCK_DELTA / FULL

    A single mode per frame must compromise across all four, which is the only honest way
    to test whether per-tile selection helps.

    The bottom-right quadrant is deliberately *compressible* (smooth gradients that jump
    phase) rather than random noise: incompressible data is a fixed floor no mode can
    improve, and it would dominate the file and make every gain elsewhere look like
    rounding error.
    """
    rng = np.random.default_rng(23)
    half_h, half_w = H // 2, W // 2

    ui = np.full((half_h, half_w, 3), 28, np.uint8)
    ui[0:16, :] = (48, 48, 56)
    for i in range(4):
        ui[26 + i * 20 : 38 + i * 20, 8:60] = (64, 64, 78)

    bg = (rng.integers(0, 50, (half_h, half_w, 3)) + 40).astype(np.uint8)
    block = rng.integers(120, 240, (40, 50, 3)).astype(np.uint8)
    still = (rng.integers(0, 256, (half_h, half_w, 3))).astype(np.uint8)

    qy, qx = np.mgrid[0 : H - half_h, 0 : W - half_w]
    palettes = []
    for phase in range(4):
        panel = np.zeros((H - half_h, W - half_w, 3), np.uint8)
        panel[..., 0] = ((qx * 3 + phase * 60) % 256).astype(np.uint8)
        panel[..., 1] = ((qy * 4 + phase * 90) % 256).astype(np.uint8)
        panel[..., 2] = ((qx + qy + phase * 40) % 256).astype(np.uint8)
        palettes.append(panel)

    frames = []
    for k in range(n):
        f = np.zeros((H, W, 3), np.uint8)

        f[:half_h, :half_w] = ui
        if (k // 8) % 2 == 0:  # blinking cursor: a handful of changed pixels
            f[60:72, 70:74] = (255, 255, 255)

        right = bg.copy()
        x = min(3 * k, half_w - 50)
        right[20:60, x : x + 50] = block
        f[:half_h, half_w:] = right

        f[half_h:, :half_w] = still  # never changes

        f[half_h:, half_w:] = palettes[(k // 6) % len(palettes)]
        frames.append(f)
    return frames


CLIPS = {
    "screen": clip_screen,
    "slide": clip_slide,
    "cut": clip_cut,
    "noise": clip_noise,
    "mixed": clip_mixed,
}


def main(argv: list[str] | None = None) -> int:
    wanted = argv if argv else list(CLIPS)
    unknown = [name for name in wanted if name not in CLIPS]
    if unknown:
        print(f"unknown clip(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(CLIPS)}", file=sys.stderr)
        return 2

    for name in wanted:
        path = write_clip(OUT / f"{name}.mp4", CLIPS[name]())
        print(f"  {name:<8} {path}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

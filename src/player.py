"""Watching a .myv directly, without exporting an MP4 first.

This matters more than it sounds. `vidc decode` gives you an H.264 *re-encode* of the
codec's output, so when a frame looks wrong you cannot tell whether the codec or the
re-encode broke it. Here the decoder's own reconstructed frames go straight to the
renderer, so what is on screen is what the codec actually produced.

Rendering is ffplay's job. Writing a window, an event loop and a frame clock would be a
lot of code that has nothing to do with compression, and ffmpeg already ships one.

The two pieces of real logic — building the renderer command and choosing which frames to
send — are separated from the subprocess plumbing so they can be tested without opening a
window.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src import pixfmt
from src.decoder import open_myv
from src.models.header import Header
from src.utils.errors import FFmpegError


def _require_ffplay() -> str:
    path = shutil.which("ffplay")
    if path is None:
        raise FFmpegError(
            "ffplay not found on PATH. It ships with ffmpeg — or use "
            "`vidc decode` to export an MP4 and open that instead."
        )
    return path


def build_command(
    header: Header, *, ffplay: str, title: str, width: int | None, speed: float
) -> list[str]:
    """The ffplay invocation for a raw stream of this header's frames."""
    fps = header.fps * speed
    cmd = [
        ffplay,
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", header.pix_fmt.ffmpeg_name,
        "-video_size", f"{header.width}x{header.height}",
        "-framerate", f"{fps:.6f}",
        "-window_title", title,
        # Close when the stream ends. Without it ffplay holds the window open waiting for
        # a keypress, so `vidc play` would hang after the last frame instead of returning.
        "-autoexit",
        # Without this ffplay buffers a large queue before showing anything, which on a
        # slow-decoding frame stalls the window rather than the pipe.
        "-fflags", "nobuffer",
        "-i", "-",
    ]
    if width:
        scaled = round(header.height * width / header.width)
        cmd[-2:-2] = ["-x", str(width), "-y", str(max(1, scaled))]
    return cmd


def select_frames(
    frames: Iterable[tuple[np.ndarray, ...]], *, start: int, count: int
) -> Iterator[tuple[np.ndarray, ...]]:
    """Skip to ``start`` and yield at most ``count`` frames (0 meaning all of them).

    Skipping means decoding and discarding: reconstruction is sequential, so frame N needs
    frame N−1. Real seeking needs keyframes and an index, which is M7.

    Asking for more frames than exist, or starting past the end, yields what there is
    rather than raising — a short video is not a user error.
    """
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")

    sent = 0
    for index, frame in enumerate(frames):
        if index < start:
            continue
        if count and sent >= count:
            return
        yield frame
        sent += 1


class _Renderer:
    """Thin wrapper over the ffplay child, so the player has one seam to stand in for."""

    def __init__(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, data: bytes) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(data)

    def close(self) -> int:
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        return self._proc.wait()


def _open_renderer(cmd: list[str]):
    """A seam, so tests can stand in for ffplay without patching subprocess."""
    return _Renderer(cmd)


def play(
    path: str | Path,
    *,
    start: int = 0,
    count: int = 0,
    speed: float = 1.0,
    width: int | None = None,
    loop: bool = False,
) -> int:
    """Render a .myv's own reconstructed frames. Returns the number of frames shown."""
    path = Path(path)
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")

    ffplay = _require_ffplay()
    shown = 0

    while True:
        with open_myv(path) as (header, frames):
            cmd = build_command(
                header, ffplay=ffplay, title=f"vidc — {path.name}",
                width=width, speed=speed,
            )
            renderer = _open_renderer(cmd)
            try:
                for frame in select_frames(frames, start=start, count=count):
                    try:
                        renderer.write(pixfmt.join(frame))
                    except BrokenPipeError:
                        # The window was closed. That is how you stop a player, not a
                        # failure worth a traceback.
                        return shown
                    shown += 1
            finally:
                renderer.close()
        if not loop:
            return shown

"""The bridge to the outside world.

The codec never needs to understand MP4 — that's a container problem ffmpeg already solved.
This module is the only place that knows subprocesses exist; everything upstream of it deals
in numpy frames.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src.models.pixel_format import RGB24, PixelFormat
from src.models.video import VideoInfo
from src.utils.errors import FFmpegError


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegError(
            f"{tool} not found on PATH. This codec uses ffmpeg to read and write MP4; "
            f"install it from https://ffmpeg.org/download.html"
        )
    return path


def probe(path: str | Path) -> VideoInfo:
    """Read dimensions and frame rate from a media file's first video stream."""
    path = Path(path)
    if not path.exists():
        raise FFmpegError(f"no such file: {path}")

    cmd = [
        _require("ffprobe"), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed on {path}:\n{result.stderr.decode(errors='replace')}"
        )

    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise FFmpegError(f"{path} has no video stream")
    stream = streams[0]

    # A rational, not a float: 30000/1001 must survive the round-trip exactly.
    rate = stream.get("r_frame_rate") or "25/1"
    num, _, den = rate.partition("/")
    try:
        fps_num, fps_den = int(num), int(den or 1)
    except ValueError:
        fps_num, fps_den = 25, 1
    if fps_num <= 0 or fps_den <= 0:
        fps_num, fps_den = 25, 1  # the container lied; pick something sane and move on

    # nb_frames is absent on plenty of containers and "N/A" on plenty more.
    nb = stream.get("nb_frames")
    num_frames = int(nb) if nb not in (None, "N/A", "") else None

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        num_frames=num_frames,
    )


@contextmanager
def decode_frames(
    path: str | Path, pix_fmt: PixelFormat = RGB24, info: VideoInfo | None = None
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of decoded frames as ``(h, w, channels)`` uint8 arrays.

    A context manager so the ffmpeg process is always reaped, even if the caller stops
    iterating early::

        with decode_frames("in.mp4") as frames:
            for frame in frames:
                ...
    """
    path = Path(path)
    info = info or probe(path)
    nbytes = pix_fmt.frame_nbytes(info.width, info.height)
    shape = pix_fmt.frame_shape(info.width, info.height)

    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin",
        "-i", str(path),
        "-map", "0:v:0",            # video only; audio is out of scope
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt.ffmpeg_name,
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    drained = False  # did we read all the way to end-of-stream?

    def read_frames() -> Iterator[np.ndarray]:
        nonlocal drained
        assert proc.stdout is not None
        index = 0
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                drained = True
                return
            if len(buf) != nbytes:
                # A partial frame means the stream was cut mid-frame. Never pad it out;
                # that would silently corrupt the last frame and break losslessness.
                raise FFmpegError(
                    f"{path}: frame {index} is short — got {len(buf)} of {nbytes} bytes"
                )
            # Zero-copy, and each frame owns its own buffer, so callers may retain it.
            yield np.frombuffer(buf, dtype=np.uint8).reshape(shape)
            index += 1

    try:
        yield read_frames()
    finally:
        if not drained:
            # The caller stopped early. Closing the pipe would make ffmpeg die of SIGPIPE
            # and report a nonzero exit, which isn't a failure worth reporting — we asked
            # it to stop. Kill it deliberately and don't check the exit code.
            proc.kill()
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.stderr:
            proc.stderr.close()
        code = proc.wait()
        if drained and code != 0:
            raise FFmpegError(
                f"ffmpeg failed decoding {path} (exit {code}):\n"
                f"{stderr.decode(errors='replace')}"
            )


def encode_frames(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    pix_fmt: PixelFormat = RGB24,
    lossless: bool = False,
) -> int:
    """Pipe frames into ffmpeg to produce a playable MP4. Returns the frame count.

    With ``lossless=False`` the output is a re-encode and is not byte-identical to what went
    in — it's for watching. Proving the round-trip is exact is ``verify``'s job, and it does
    that by comparing arrays in memory instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if lossless:
        # libx264rgb encodes RGB directly. Plain libx264 -qp 0 is only lossless *after* the
        # RGB->YUV conversion, and that conversion rounds — so it would quietly cost a pixel
        # value here and there, which defeats the point of asking for lossless.
        codec = ["-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24"]
    else:
        # yuv420p so ordinary players and browsers can open the result.
        codec = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]

    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt.ffmpeg_name,
        "-s", f"{width}x{height}",
        "-r", f"{fps_num}/{fps_den}",
        "-i", "-",
        "-preset", "medium", *codec,
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    written = 0
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        # ffmpeg died early; its stderr says why, and that's far more useful than the
        # BrokenPipeError traceback.
        stderr = proc.stderr.read() if proc.stderr else b""
        proc.wait()
        raise FFmpegError(
            f"ffmpeg closed the pipe while writing {path}:\n"
            f"{stderr.decode(errors='replace')}"
        ) from None

    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.stderr:
        proc.stderr.close()
    code = proc.wait()
    if code != 0:
        raise FFmpegError(
            f"ffmpeg failed writing {path} (exit {code}):\n{stderr.decode(errors='replace')}"
        )
    return written

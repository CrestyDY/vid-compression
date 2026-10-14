"""The bridge to the outside world.

The codec never needs to understand MP4 — that's a container problem ffmpeg already solved.
This module is the only place that knows subprocesses exist; everything upstream of it deals
in numpy frames.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

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

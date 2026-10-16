"""The bridge to the outside world.

The codec never needs to understand MP4 — that's a container problem ffmpeg already solved.
This module is the only place that knows subprocesses exist; everything upstream of it deals
in numpy frames.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src import pixfmt
from src.models.pixel_format import RGB24, PixelFormat
from src.models.video import VideoInfo
from src.utils.errors import FFmpegError


def _drain_stderr(proc: subprocess.Popen) -> threading.Thread | None:
    """Read ffmpeg's stderr on a thread for as long as it lives.

    Reading stderr only *after* the data pipe is finished is a deadlock: if ffmpeg emits
    more stderr than the pipe buffer holds (a damaged input emits one line per bad packet,
    which exceeds a 64 KiB buffer easily) it blocks on that write, stops touching the data
    pipe, and we block on ours. A hang with no diagnostic is the worst failure to debug
    later, so the pipe is drained continuously instead.
    """
    if proc.stderr is None:
        return None
    chunks: list[bytes] = []

    def pump() -> None:
        try:
            chunks.append(proc.stderr.read())
        except (ValueError, OSError):
            pass  # pipe closed under us during teardown

    thread = threading.Thread(target=pump, daemon=True)
    thread.chunks = chunks  # type: ignore[attr-defined]
    thread.start()
    return thread


def _collect(drain: threading.Thread | None) -> bytes:
    if drain is None:
        return b""
    drain.join(timeout=10)
    return b"".join(getattr(drain, "chunks", []))


def _reap(proc: subprocess.Popen, drain: threading.Thread | None) -> tuple[int, bytes]:
    """Close stdin, wait for the child, and return its exit code and stderr.

    Closing stdin first is what lets ffmpeg finish and exit rather than block forever on
    a pipe that will never deliver another byte.
    """
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    stderr = _collect(drain)
    code = proc.wait()
    try:
        if proc.stderr is not None and not proc.stderr.closed:
            proc.stderr.close()
    except OSError:
        pass
    return code, stderr


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
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,pix_fmt",
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

    pix_fmt = stream.get("pix_fmt") or None

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        num_frames=num_frames,
        pix_fmt=pix_fmt,
    )


@contextmanager
def decode_frames(
    path: str | Path, pix_fmt: PixelFormat = RGB24, info: VideoInfo | None = None
) -> Iterator[Iterator[tuple[np.ndarray, ...]]]:
    """Yield an iterator of decoded frames, each a tuple of ``(h, w, channels)`` planes.

    Always a tuple, even for RGB24, whose single plane carries all three channels. A
    one-plane tuple rather than a bare array is what keeps every caller from branching on
    the pixel format.

    A context manager so the ffmpeg process is always reaped, even if the caller stops
    iterating early::

        with decode_frames("in.mp4") as frames:
            for planes in frames:
                ...
    """
    path = Path(path)
    info = info or probe(path)
    nbytes = pix_fmt.frame_nbytes(info.width, info.height)

    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin",
        "-i", str(path),
        "-map", "0:v:0",            # video only; audio is out of scope
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt.ffmpeg_name,
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    drain = _drain_stderr(proc)
    drained = False  # did we read all the way to end-of-stream?

    def read_frames() -> Iterator[tuple[np.ndarray, ...]]:
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
            # Zero-copy views, and each frame owns its own buffer, so callers may retain it.
            yield pixfmt.split(buf, pix_fmt, info.width, info.height)
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
        stderr = _collect(drain)
        code = proc.wait()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()
        if drained and code != 0:
            raise FFmpegError(
                f"ffmpeg failed decoding {path} (exit {code}):\n"
                f"{stderr.decode(errors='replace')}"
            )


def encode_frames(
    path: str | Path,
    frames: Iterable[tuple[np.ndarray, ...]],
    *,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    pix_fmt: PixelFormat = RGB24,
    lossless: bool = False,
) -> int:
    """Pipe frames into ffmpeg to produce a playable MP4. Returns the frame count.

    Each frame is a tuple of planes, as :func:`decode_frames` yields them.

    With ``lossless=False`` the output is a re-encode and is not byte-identical to what went
    in — it's for watching. Proving the round-trip is exact is ``verify``'s job, and it does
    that by comparing arrays in memory instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not lossless and (width % 2 or height % 2):
        # libx264 with yuv420p needs even dimensions in both axes. libx264rgb does not,
        # so the capability exists — say which flag reaches it instead of letting ffmpeg
        # emit ten lines of internals.
        raise FFmpegError(
            f"cannot export {width}x{height} as H.264: libx264 requires even "
            f"dimensions and these are odd. Re-run with --lossless, which uses "
            f"libx264rgb and accepts odd dimensions."
        )

    if lossless and pix_fmt is RGB24:
        # libx264rgb encodes RGB directly. Plain libx264 -qp 0 is only lossless *after* the
        # RGB->YUV conversion, and that conversion rounds — so it would quietly cost a pixel
        # value here and there, which defeats the point of asking for lossless.
        codec = ["-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24"]
    elif lossless:
        # The frames are already YUV, so the argument above inverts: libx264rgb would be
        # the one forcing a colour conversion. Keeping the output pix_fmt equal to the
        # input's is what makes -qp 0 actually lossless — x264 picks the profile that
        # carries it (high422, high444) by itself.
        codec = ["-c:v", "libx264", "-qp", "0", "-pix_fmt", pix_fmt.ffmpeg_name]
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
    drain = _drain_stderr(proc)

    written = 0
    broken = False
    try:
        for planes in frames:
            proc.stdin.write(pixfmt.join(planes))
            written += 1
    except BrokenPipeError:
        # ffmpeg died early. Its exit code and stderr say why, which is far more useful
        # than the BrokenPipeError traceback, so fall through and report those.
        broken = True
    except BaseException:
        # The frame source itself failed. Reap the child before the exception leaves, or
        # ffmpeg is left alive blocked on a stdin nobody will ever close again.
        _reap(proc, drain)
        raise

    code, stderr = _reap(proc, drain)
    if broken:
        raise FFmpegError(
            f"ffmpeg closed the pipe while writing {path} (exit {code}):\n"
            f"{stderr.decode(errors='replace')}"
        )
    if code != 0:
        raise FFmpegError(
            f"ffmpeg failed writing {path} (exit {code}):\n{stderr.decode(errors='replace')}"
        )
    return written

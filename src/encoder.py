"""MP4 -> .myv.

The encoder holds all the intelligence in this codec. It decides how each frame is
represented; the file records that decision; the decoder just follows instructions. That
asymmetry is why real-world decoders are cheap and encoders are expensive.

Deltas are taken against the *previous original frame*. Because the codec is lossless the
previous original and the previous reconstruction are identical, so the choice is invisible
in the output — but stating it explicitly matters, since the moment quantisation appears the
encoder must switch to predicting from the reconstruction. Predicting from originals while
the decoder predicts from reconstructions is the classic drift bug, and it is much easier to
avoid now than to diagnose later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.format import FRAME_HEADER_SIZE, HEADER_SIZE, MyvWriter
from src.models.frame import FrameType
from src.models.header import Header
from src.models.pixel_format import RGB24, PixelFormat
from src.models.stats import CodecStats
from src.modes import MODES, pixel_delta
from src.utils import entropy, ffmpeg


def encode(
    src: str | Path,
    dst: str | Path,
    *,
    use_entropy: bool = True,
    pix_fmt: PixelFormat = RGB24,
    progress=None,
) -> CodecStats:
    """Encode a video file into a .myv, returning the measurements taken along the way.

    The strategy here is fixed: frame 0 is FULL, every later frame is PIXEL_DELTA. M2
    replaces that with pricing every mode and keeping the cheapest; keeping this path
    available is what makes M2's improvement measurable rather than merely asserted.
    """
    src, dst = Path(src), Path(dst)
    info = ffmpeg.probe(src)

    header = Header(
        width=info.width,
        height=info.height,
        fps_num=info.fps_num,
        fps_den=info.fps_den,
        frame_count=0,  # back-patched on close
        pix_fmt=pix_fmt,
        entropy=use_entropy,
    )

    stats = CodecStats(
        width=info.width,
        height=info.height,
        fps=info.fps,
        source_bytes=src.stat().st_size,
        entropy_enabled=use_entropy,
        container_bytes=HEADER_SIZE,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file and rename only on success. MyvWriter back-patches the
    # frame count on close, so a failure partway through would otherwise leave a
    # structurally *valid* .myv that is silently missing its tail — and `info` would
    # report it as a complete archive. A half-written file is worse than no file.
    partial = dst.with_name(dst.name + ".partial")
    try:
        _write(src, partial, header, stats, info, pix_fmt, use_entropy, progress)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dst)

    return stats


def _write(
    src: Path,
    dst: Path,
    header: Header,
    stats: CodecStats,
    info,
    pix_fmt: PixelFormat,
    use_entropy: bool,
    progress,
) -> None:
    """The streaming encode itself. Separated so the caller owns the atomic rename."""
    with open(dst, "wb") as out, ffmpeg.decode_frames(src, pix_fmt, info) as frames:
        with MyvWriter(out, header) as writer:
            prev: np.ndarray | None = None
            for index, frame in enumerate(frames):
                frame_type = FrameType.FULL if prev is None else FrameType.PIXEL_DELTA
                payload = MODES[frame_type].encode(frame, prev)
                stored = entropy.compress(payload, enabled=use_entropy)
                writer.frame(frame_type, stored)

                changed = None
                if frame_type is FrameType.PIXEL_DELTA:
                    changed = pixel_delta.count(payload)
                stats.record(frame_type, len(payload), len(stored), changed)
                stats.container_bytes += FRAME_HEADER_SIZE

                if progress is not None:
                    progress(index, info.num_frames)

                prev = frame

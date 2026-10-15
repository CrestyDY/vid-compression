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
from src.models.options import DEFAULT_OPTIONS, ModeOptions
from src.models.pixel_format import RGB24, PixelFormat
from src.models.stats import CodecStats
from src.modes import pixel_delta
from src.selector import STRATEGIES, choose
from src.utils import entropy, ffmpeg


def encode(
    src: str | Path,
    dst: str | Path,
    *,
    use_entropy: bool = True,
    entropy_codec: str = "zstd",
    selection: str = "trial",
    enabled_modes: frozenset[FrameType] | None = None,
    opts: ModeOptions = DEFAULT_OPTIONS,
    pix_fmt: PixelFormat = RGB24,
    progress=None,
) -> CodecStats:
    """Encode a video file into a .myv, returning the measurements taken along the way.

    ``selection`` chooses how modes are ranked — see :mod:`src.selector`. ``fixed`` is M1's
    behaviour, kept so the improvement from choosing is measurable rather than asserted.
    ``enabled_modes`` restricts the candidate set, which is how a single mode gets measured
    in isolation.
    """
    src, dst = Path(src), Path(dst)
    if entropy_codec not in entropy.CODECS:
        raise ValueError(
            f"unknown entropy codec {entropy_codec!r}; pick one of "
            f"{entropy.CODECS}"
        )
    if selection not in STRATEGIES:
        raise ValueError(f"unknown selection strategy {selection!r}; pick one of {STRATEGIES}")
    info = ffmpeg.probe(src)

    header = Header(
        width=info.width,
        height=info.height,
        fps_num=info.fps_num,
        fps_den=info.fps_den,
        frame_count=0,  # back-patched on close
        pix_fmt=pix_fmt,
        entropy=use_entropy,
        entropy_codec=entropy_codec,
    )

    stats = CodecStats(
        width=info.width,
        height=info.height,
        fps=info.fps,
        source_bytes=src.stat().st_size,
        entropy_enabled=use_entropy,
        entropy_codec=entropy_codec,
        container_bytes=HEADER_SIZE,
        selection=selection,
        block_size=opts.block_size,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file and rename only on success. MyvWriter back-patches the
    # frame count on close, so a failure partway through would otherwise leave a
    # structurally *valid* .myv that is silently missing its tail — and `info` would
    # report it as a complete archive. A half-written file is worse than no file.
    partial = dst.with_name(dst.name + ".partial")
    try:
        _write(
            src, partial, header, stats, info, pix_fmt, use_entropy,
            entropy_codec, selection, enabled_modes, opts, progress,
        )
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
    entropy_codec: str,
    selection: str,
    enabled_modes: frozenset[FrameType] | None,
    opts: ModeOptions,
    progress,
) -> None:
    """The streaming encode itself. Separated so the caller owns the atomic rename."""
    with open(dst, "wb") as out, ffmpeg.decode_frames(src, pix_fmt, info) as frames:
        with MyvWriter(out, header) as writer:
            prev: np.ndarray | None = None
            for index, frame in enumerate(frames):
                pick = choose(
                    frame, prev,
                    strategy=selection,
                    enabled=enabled_modes,
                    opts=opts,
                    use_entropy=use_entropy,
                    entropy_codec=entropy_codec,
                )
                stored = entropy.compress(
                    pick.payload, enabled=use_entropy, codec=entropy_codec
                )
                writer.frame(pick.frame_type, stored)

                changed = None
                if pick.frame_type is FrameType.PIXEL_DELTA:
                    changed = pixel_delta.count(pick.payload)
                elif prev is not None:
                    # Report the changed-pixel count even when another mode won, so the
                    # figure stays comparable across strategies and milestones.
                    changed = int(np.count_nonzero(np.any(frame != prev, axis=2)))

                stats.record(
                    pick.frame_type, len(pick.payload), len(stored), changed,
                    candidate_raw=pick.candidate_raw,
                    candidate_stored=pick.candidate_stored,
                )
                stats.container_bytes += FRAME_HEADER_SIZE

                if progress is not None:
                    progress(index, info.num_frames)

                prev = frame

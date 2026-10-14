"""Proving the round-trip is exact.

This compares numpy arrays in memory, never exported files. Diffing two MP4s would fail on
re-encode noise even when the codec is perfect, so it would prove nothing either way — the
question "is my codec lossless" is about the decoded frames, and that's where it gets asked.
"""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path

import numpy as np

from src.decoder import open_myv, read_header
from src.models.pixel_format import PixelFormat
from src.models.verify import Mismatch, VerifyResult
from src.utils import ffmpeg


def verify(
    source: str | Path, myv: str | Path, pix_fmt: PixelFormat | None = None
) -> VerifyResult:
    """Compare every decoded .myv frame against the source video, frame by frame.

    Stops at the first mismatch — one is enough to disprove losslessness, and reporting
    where it happened is more useful than counting how many followed.
    """
    myv = Path(myv)

    # Compare dimensions before comparing pixels. Without this, a mismatch reaches
    # np.array_equal — which answers False for non-broadcastable shapes rather than
    # raising — and then dies inside _describe on the elementwise compare. Verifying
    # against the wrong source file is an ordinary slip and deserves a real answer.
    stored = read_header(myv)
    info = ffmpeg.probe(source)
    if (info.width, info.height) != (stored.width, stored.height):
        return VerifyResult(
            frames_checked=0,
            source_frames=info.num_frames or 0,
            decoded_frames=stored.frame_count,
            mismatch=Mismatch(
                frame=0, differing_pixels=0, first_at=None, expected=None, got=None,
                max_abs_error=0,
                note=(
                    f"dimension mismatch: the source is "
                    f"{info.width}x{info.height} but the .myv holds "
                    f"{stored.width}x{stored.height} — is this the right source file?"
                ),
            ),
        )

    with open_myv(myv) as (header, decoded):
        fmt = pix_fmt or header.pix_fmt
        with ffmpeg.decode_frames(source, fmt) as originals:
            checked = n_src = n_dec = 0
            for index, (original, reconstructed) in enumerate(
                zip_longest(originals, decoded)
            ):
                n_src += original is not None
                n_dec += reconstructed is not None
                if original is None or reconstructed is None:
                    # One side ran out. Drain the other so the reported totals are the real
                    # frame counts rather than just where the comparison stopped.
                    if original is None:
                        n_dec += sum(1 for _ in decoded)
                    else:
                        n_src += sum(1 for _ in originals)
                    return VerifyResult(
                        frames_checked=checked,
                        source_frames=n_src,
                        decoded_frames=n_dec,
                        mismatch=Mismatch(
                            frame=index, differing_pixels=0, first_at=None,
                            expected=None, got=None, max_abs_error=0,
                        ),
                    )

                if not np.array_equal(original, reconstructed):
                    return VerifyResult(
                        frames_checked=checked,
                        source_frames=n_src,
                        decoded_frames=n_dec,
                        mismatch=_describe(index, original, reconstructed),
                    )
                checked += 1

    return VerifyResult(
        frames_checked=checked, source_frames=n_src, decoded_frames=n_dec, mismatch=None
    )


def _describe(index: int, original: np.ndarray, got: np.ndarray) -> Mismatch:
    differing = np.any(original != got, axis=2)
    ys, xs = np.nonzero(differing)
    if ys.size == 0:
        # array_equal said they differ but no pixel does: the shapes broadcast against
        # each other. Report that rather than indexing into an empty array.
        return Mismatch(
            frame=index, differing_pixels=0, first_at=None, expected=None, got=None,
            max_abs_error=0,
            note=(
                f"frame {index}: shapes differ — source {original.shape} vs "
                f"reconstructed {got.shape}"
            ),
        )
    y, x = int(ys[0]), int(xs[0])
    error = np.abs(original.astype(np.int16) - got.astype(np.int16))
    return Mismatch(
        frame=index,
        differing_pixels=int(ys.size),
        first_at=(y, x),
        expected=tuple(int(v) for v in original[y, x]),
        got=tuple(int(v) for v in got[y, x]),
        max_abs_error=int(error.max()),
    )

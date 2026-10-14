""".myv -> frames.

The decoder is the simple half, and that asymmetry is the point: the encoder makes all the
decisions and the file records which ones it made, so the decoder only has to follow
instructions. That's true of real codecs too, and it's why H.264 decoders are cheap while
H.264 encoders are expensive.

Reconstruction is sequential — frame N needs frame N-1 — so there is no random access. Until
keyframes exist, seeking would mean decoding from the start anyway.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from src.format import MyvReader
from src.models.frame import FrameType
from src.models.header import Header
from src.modes import MODES
from src.utils import entropy
from src.utils.errors import BadMyvFile, CorruptPayload


@contextmanager
def open_myv(path: str | Path) -> Iterator[tuple[Header, Iterator[np.ndarray]]]:
    """Open a .myv, yielding its header and a lazy iterator of reconstructed frames.

        with open_myv("video.myv") as (header, frames):
            for frame in frames:
                ...
    """
    with open(Path(path), "rb") as fh:
        reader = MyvReader(fh)
        yield reader.header, _reconstruct(reader)


def read_header(path: str | Path) -> Header:
    with open(Path(path), "rb") as fh:
        return MyvReader(fh).header


def decode_to_list(path: str | Path) -> list[np.ndarray]:
    """Every frame at once. Convenient for tests; ruinous for a feature film."""
    with open_myv(path) as (_header, frames):
        return list(frames)


def _reconstruct(reader: MyvReader) -> Iterator[np.ndarray]:
    header = reader.header
    shape = header.frame_shape
    prev: np.ndarray | None = None

    for index, record in enumerate(reader.frames()):
        if index == 0 and record.type is not FrameType.FULL:
            raise BadMyvFile(
                f"first frame is {record.type.name}, but there is nothing for it to "
                f"predict from; the file must start with a FULL frame"
            )
        try:
            payload = entropy.decompress(record.payload, enabled=header.entropy)
        except CorruptPayload as exc:
            raise BadMyvFile(
                f"frame {index} ({record.type.name}): payload could not be "
                f"decompressed: {exc}"
            ) from None
        try:
            frame = MODES[record.type].decode(payload, prev, shape)
        except (ValueError, IndexError) as exc:
            # IndexError too: a corrupt coordinate in a delta record indexes outside the
            # frame, which is a broken file rather than a bug worth a traceback.
            raise BadMyvFile(f"frame {index} ({record.type.name}): {exc}") from None
        prev = frame
        yield frame

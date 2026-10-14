"""Exceptions shared across the codec."""

from __future__ import annotations


class FFmpegError(Exception):
    """ffmpeg or ffprobe failed, or isn't installed."""


class BadMyvFile(Exception):
    """The file isn't a .myv, or is corrupt, or came from a newer codec."""


class CorruptPayload(Exception):
    """A payload could not be decoded by the entropy stage.

    Raised by ``utils.entropy`` so that zstandard's own exception type never escapes into
    the codec, and the decoder can turn it into a ``BadMyvFile`` naming the frame.
    """

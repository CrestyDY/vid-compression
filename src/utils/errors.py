"""Exceptions shared across the codec."""

from __future__ import annotations


class FFmpegError(Exception):
    """ffmpeg or ffprobe failed, or isn't installed."""


class BadMyvFile(Exception):
    """The file isn't a .myv, or is corrupt, or came from a newer codec."""

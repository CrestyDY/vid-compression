"""The `vidc` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import stats as stats_report
from src.decoder import open_myv, read_header
from src.encoder import encode as encode_file
from src.models.frame import FrameType
from src.models.options import ModeOptions
from src.player import play as play_myv
from src.selector import STRATEGIES
from src.utils import ffmpeg
from src.utils.errors import BadMyvFile, FFmpegError
from src.utils.sizes import human_bytes
from src.verify import verify as verify_files


def _parse_modes(spec: str | None) -> frozenset[FrameType] | None:
    """``"FULL,PIXEL_DELTA"`` -> a candidate set. ``None`` means every mode."""
    if not spec:
        return None
    names = [part.strip().upper() for part in spec.split(",") if part.strip()]
    try:
        return frozenset(FrameType[name] for name in names)
    except KeyError as exc:
        known = ", ".join(ft.name for ft in FrameType)
        raise ValueError(f"unknown mode {exc.args[0]!r}; known modes are {known}") from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidc", description="An experimental lossless video codec."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="MP4 -> .myv, printing a size report")
    enc.add_argument("src", type=Path)
    enc.add_argument("dst", type=Path)
    enc.add_argument("--no-entropy", action="store_true", help="skip the final zstd stage")
    enc.add_argument(
        "--selection", choices=STRATEGIES, default="trial",
        help="how modes are ranked (default: trial)",
    )
    enc.add_argument(
        "--modes", default=None,
        help="restrict the candidate set, e.g. FULL,PIXEL_DELTA — to measure one mode alone",
    )
    enc.add_argument(
        "--block-size", type=int, default=16,
        help="grid cell side length for BLOCK_DELTA and MOTION (default: 16)",
    )
    enc.add_argument(
        "--search-range", type=int, default=7,
        help="how far MOTION looks for a match, in pixels (default: 7)",
    )

    dec = sub.add_parser("decode", help=".myv -> MP4, for sharing or archiving")
    dec.add_argument("src", type=Path)
    dec.add_argument("dst", type=Path)
    dec.add_argument("--lossless", action="store_true", help="export with libx264rgb -qp 0")

    pl = sub.add_parser("play", help="watch a .myv directly, with no export step")
    pl.add_argument("src", type=Path)
    pl.add_argument("--start", type=int, default=0, help="first frame to show")
    pl.add_argument("--count", type=int, default=0, help="how many frames (0 = all)")
    pl.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    pl.add_argument("--width", type=int, default=None, help="window width in pixels")
    pl.add_argument("--loop", action="store_true", help="repeat until the window closes")

    info = sub.add_parser("info", help="header and per-mode breakdown")
    info.add_argument("src", type=Path)

    ver = sub.add_parser("verify", help="prove the round-trip is pixel-exact")
    ver.add_argument("src", type=Path)
    ver.add_argument("myv", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "encode":
            result = encode_file(
                args.src, args.dst,
                use_entropy=not args.no_entropy,
                selection=args.selection,
                enabled_modes=_parse_modes(args.modes),
                opts=ModeOptions(
                    block_size=args.block_size,
                    search_range=args.search_range,
                ),
            )
            print(f"{args.src} -> {args.dst}")
            print(stats_report.render(result))
            return 0

        if args.command == "decode":
            with open_myv(args.src) as (header, frames):
                written = ffmpeg.encode_frames(
                    args.dst, frames,
                    width=header.width, height=header.height,
                    fps_num=header.fps_num, fps_den=header.fps_den,
                    pix_fmt=header.pix_fmt, lossless=args.lossless,
                )
            print(f"{args.src} -> {args.dst} ({written} frames)")
            return 0

        if args.command == "play":
            shown = play_myv(
                args.src, start=args.start, count=args.count,
                speed=args.speed, width=args.width, loop=args.loop,
            )
            print(f"  {shown} frames shown")
            return 0

        if args.command == "info":
            header = read_header(args.src)
            size = args.src.stat().st_size
            print(f"  {args.src}")
            print(f"  {header.width}x{header.height} @ {header.fps:.2f} fps")
            print(f"  {header.frame_count} frames, {header.pix_fmt.name}")
            print(f"  entropy coded: {header.entropy}")
            print(f"  on disk: {human_bytes(size)}")
            return 0

        if args.command == "verify":
            result = verify_files(args.src, args.myv)
            if result.ok:
                print(
                    f"  lossless: {result.frames_checked} frames are byte-identical "
                    f"to the source"
                )
                return 0
            print("  NOT lossless")
            if result.mismatch is not None:
                print(f"  {result.mismatch.describe()}")
            if result.source_frames != result.decoded_frames:
                print(
                    f"  frame counts differ: source {result.source_frames}, "
                    f"decoded {result.decoded_frames}"
                )
            return 1

    except (FFmpegError, BadMyvFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # A bad --modes or --selection value, caught before any work is done.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # A missing or unreadable file is the same user mistake whichever subcommand hit
        # it, so it gets the same exit code. `encode` reached this via probe()'s own
        # check; decode, info and verify open the .myv directly.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Rendering the size report.

Always shows both the pre-entropy and the final size, so you can tell what the codec's
*representation* achieved from what generic compression achieved. Those are different
questions, and a single number hides which one did the work.
"""

from __future__ import annotations

from src.models.stats import CodecStats
from src.utils.sizes import human_bytes


def render(stats: CodecStats) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"  {stats.width}x{stats.height} @ {stats.fps:.2f} fps, {stats.frames} frames")
    add(f"  source file        {human_bytes(stats.source_bytes):>12}")
    add(f"  raw frames         {human_bytes(stats.raw_frame_bytes):>12}")
    add(f"  payloads           {human_bytes(stats.raw_bytes):>12}   (before entropy)")
    stage = "zstd" if stats.entropy_enabled else "no entropy stage"
    add(f"  payloads stored    {human_bytes(stats.stored_bytes):>12}   ({stage})")
    add(f"  container overhead {human_bytes(stats.container_bytes):>12}")
    add(f"  total on disk      {human_bytes(stats.total_bytes):>12}")

    if stats.raw_frame_bytes and stats.total_bytes:
        ratio = stats.raw_frame_bytes / stats.total_bytes
        add(f"  vs raw frames      {ratio:>11.2f}x")

    if stats.frames:
        add(f"  selection          {stats.selection:>12}")
        add("  modes chosen:")
        for frame_type, count in sorted(stats.mode_counts.items()):
            add(f"    {frame_type.name:<14} {count:>5}")

        # What each mode would have cost if it had been used for every frame it was
        # priced on. This is what makes the choice's value visible rather than implied.
        priced = stats.locked_to_one_mode()
        if len(priced) > 1:
            label = "final" if stats.candidate_stored else "pre-entropy"
            add(f"  if locked to one mode ({label}, frame 0 forced to FULL):")
            best = min(priced.values())
            for frame_type, size in sorted(priced.items()):
                ratio = size / best if best else 1.0
                marker = "  <- best single" if size == best else f"  {ratio:.2f}x worse"
                add(f"    {frame_type.name:<14} {human_bytes(size):>12}{marker}")

            # The headline for mode selection: what choosing per frame actually shipped,
            # against the best any single mode could have managed.
            chosen = stats.stored_bytes if stats.candidate_stored else stats.raw_bytes
            gain = best / chosen if chosen else 1.0
            add(
                f"    {'(choosing)':<14} {human_bytes(chosen):>12}"
                f"  <- {gain:.2f}x better than the best single mode"
            )

        if stats.candidate_stored:
            add(
                f"  ranking disagreements {stats.ranking_disagreements:>8}"
                f" of {stats.frames} frames"
            )
        # Average over *predicted* frames only: frame 0 is FULL and has no reference, so
        # counting it would understate the figure.
        predicted = max(stats.frames - 1, 1)
        add(f"  changed pixels/frame {stats.changed_pixels // predicted:>10,}")

    return "\n".join(lines)

"""Choosing a mode per frame — the idea that makes this a codec rather than a scheme.

M1 had a fixed strategy: frame 0 FULL, everything else PIXEL_DELTA, no exceptions. That
loses badly at a scene cut, where describing every pixel individually costs 7 bytes each to
say what 3 bytes could have said outright — and on dense content it produced a file *larger*
than the raw frames it was describing. See docs/m1-results.md, Observation 1.

The fix is to stop guessing. For each frame, ask every available mode what it would cost and
keep the cheapest answer::

    mode = argmin(cost_full, cost_pixel_delta, ...)

Three strategies, because *what you compare* turns out to be a real question:

``fixed``
    M1's behaviour, kept so its cost stays measurable rather than remembered.

``actual``
    Compare the true pre-entropy payload sizes. For every mode here ``cost()`` is exact
    rather than estimated — a delta payload really is ``4 + 7n`` bytes — so this is optimal
    with respect to *the representation*, and cheap.

``trial`` (default)
    Encode *and compress* every candidate, then compare final on-disk sizes. Costs a zstd
    pass per candidate per frame.

**`trial` is the default even though, right now, it never disagrees with `actual`.** With
only FULL and PIXEL_DELTA available the two rank identically on every benchmark frame, so
today the extra zstd passes buy nothing measurable. It is still the right default, because
pre-entropy size is only a valid ranking signal while all candidates have similar
compressibility — and the moment a mode with a different statistical shape arrives, the
proxy silently stops tracking the thing it is a proxy for. BLOCK_DELTA at M3 is exactly
that mode: it emits a plane of raw delta *values*, which is low-entropy, while PIXEL_DELTA
emits coordinate-tagged records whose coordinates are all distinct and therefore nearly
incompressible. Two payloads of identical size can differ 2x after entropy coding.

Keeping `actual` is what lets that claim be measured at M3 instead of asserted.
"""

from __future__ import annotations

import numpy as np

from src.models.choice import Choice
from src.models.frame import FrameType
from src.models.options import DEFAULT_OPTIONS, ModeOptions
from src.modes import MODES, PLANEABLE, PREDICTED, TILEABLE, planes, tiled
from src.utils import entropy

#: Selection strategies, in the order they were introduced.
STRATEGIES = ("fixed", "actual", "trial")


def candidates(
    prev: np.ndarray | None, enabled: frozenset[FrameType] | None = None
) -> list[FrameType]:
    """Which modes may be used for a frame, given whether a reference frame exists."""
    allowed = set(MODES) if enabled is None else (set(MODES) & set(enabled))
    if prev is None:
        allowed -= PREDICTED  # nothing to predict from
    if not allowed:
        raise ValueError(
            "no usable modes for this frame; frame 0 requires FULL to be enabled"
        )
    return sorted(allowed)


def choose_frame(
    cur: tuple[np.ndarray, ...],
    prev: tuple[np.ndarray, ...] | None,
    *,
    strategy: str = "trial",
    enabled: frozenset[FrameType] | None = None,
    opts: ModeOptions = DEFAULT_OPTIONS,
    use_entropy: bool = True,
    entropy_codec: str = "zstd",
) -> Choice:
    """Pick a representation for a whole frame, which is a tuple of planes.

    A single-plane format delegates straight to :func:`choose` and produces exactly the
    payload it always did, so RGB24 files are unchanged by the existence of this function.

    A multi-plane format chooses per plane and wraps the results in a PLANES payload. Luma
    and chroma are different pictures at different resolutions — chroma is smoother and
    changes less often — so one answer for all three would lose on at least one of them,
    the same argument TILED makes about regions of a frame.
    """
    if len(cur) == 1:
        return choose(
            cur[0], None if prev is None else prev[0],
            strategy=strategy, enabled=enabled, opts=opts,
            use_entropy=use_entropy, entropy_codec=entropy_codec,
        )

    allowed = PLANEABLE if enabled is None else (PLANEABLE & enabled)
    picks = [
        choose(
            plane, None if prev is None else prev[index],
            strategy=strategy, enabled=allowed, opts=opts,
            use_entropy=use_entropy, entropy_codec=entropy_codec,
        )
        for index, plane in enumerate(cur)
    ]
    payload = planes.pack([(int(p.frame_type), p.payload) for p in picks])
    return Choice(
        frame_type=FrameType.PLANES,
        payload=payload,
        candidate_raw=_across_planes(picks, "candidate_raw"),
        candidate_stored=_across_planes(picks, "candidate_stored"),
        plane_modes=tuple(p.frame_type for p in picks),
    )


def _across_planes(picks: list[Choice], field: str) -> dict[FrameType, int]:
    """Sum each mode's candidate cost over the planes, keeping only modes priced on all.

    Dropping a mode that one plane could not use is what keeps the report's "if locked to
    one mode" comparison honest: a total covering two planes out of three is not a price
    for the frame.
    """
    per_plane = [getattr(pick, field) for pick in picks]
    if not all(per_plane):
        return {}
    common = set(per_plane[0])
    for sizes in per_plane[1:]:
        common &= set(sizes)
    return {
        mode: sum(sizes[mode] for sizes in per_plane)
        for mode in sorted(common)
    }


def choose(
    cur: np.ndarray,
    prev: np.ndarray | None,
    *,
    strategy: str = "trial",
    enabled: frozenset[FrameType] | None = None,
    opts: ModeOptions = DEFAULT_OPTIONS,
    use_entropy: bool = True,
    entropy_codec: str = "zstd",
) -> Choice:
    """Pick the cheapest representation of ``cur`` given ``prev``."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown selection strategy {strategy!r}; pick one of {STRATEGIES}"
        )

    available = candidates(prev, enabled)

    if strategy == "fixed":
        pick = FrameType.FULL if prev is None else FrameType.PIXEL_DELTA
        if pick not in available:
            raise ValueError(f"the 'fixed' strategy needs {pick.name} enabled")
        payload = MODES[pick].encode(cur, prev, opts)
        return Choice(frame_type=pick, payload=payload, candidate_raw={pick: len(payload)})

    payloads = {
        ft: (
            _build_tiled(cur, prev, strategy, opts, use_entropy, entropy_codec)
            if ft is FrameType.TILED
            else MODES[ft].encode(cur, prev, opts)
        )
        for ft in available
    }
    raw = {ft: len(p) for ft, p in payloads.items()}

    if strategy == "actual":
        # Ties go to the lower FrameType, which puts FULL first. Preferring FULL on a tie
        # is the right bias: it costs nothing here and is a clean resynchronisation point.
        winner = min(available, key=lambda ft: (raw[ft], int(ft)))
        return Choice(frame_type=winner, payload=payloads[winner], candidate_raw=raw)

    stored = {
        ft: len(entropy.compress(p, enabled=use_entropy, codec=entropy_codec))
        for ft, p in payloads.items()
    }
    winner = min(available, key=lambda ft: (stored[ft], int(ft)))
    return Choice(
        frame_type=winner,
        payload=payloads[winner],
        candidate_raw=raw,
        candidate_stored=stored,
    )


def _build_tiled(
    cur: np.ndarray,
    prev: np.ndarray,
    strategy: str,
    opts: ModeOptions,
    use_entropy: bool,
    entropy_codec: str = "zstd",
) -> bytes:
    """Choose a mode independently for each tile, using the frame-level criterion.

    Tiling lives here rather than in ``src.modes.tiled`` because picking well means ranking
    candidates *after* the entropy stage, and a mode module has no business reaching into
    it. ``tiled.encode`` keeps a simpler standalone path that ranks on payload size.

    Each tile is chosen by the same rule as a whole frame, one level down. There is no
    recursion: ``TILEABLE`` excludes TILED itself.
    """
    rows, cols = tiled.grid_dims(cur.shape, opts.grid)
    parts: list[tuple[int, bytes]] = []
    for y0, y1, x0, x1 in tiled.tiles(cur.shape, rows, cols):
        tile = choose(
            np.ascontiguousarray(cur[y0:y1, x0:x1]),
            np.ascontiguousarray(prev[y0:y1, x0:x1]),
            strategy=strategy,
            enabled=TILEABLE,
            opts=opts,
            use_entropy=use_entropy,
            entropy_codec=entropy_codec,
        )
        parts.append((tile.frame_type, tile.payload))
    return tiled.pack(rows, cols, parts)

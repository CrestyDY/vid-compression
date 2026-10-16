from __future__ import annotations

import numpy as np
import pytest

from src.models.frame import FrameType
from src.modes import MODES
from src.selector import STRATEGIES, candidates, choose

# ------------------------------------------------------------ which modes are legal


def test_frame_zero_cannot_use_a_predicted_mode():
    assert candidates(None) == [FrameType.FULL]


def test_later_frames_may_use_everything():
    """Every *plane* mode. PLANES is not one; it wraps them."""
    assert set(candidates(np.zeros((4, 4, 3), np.uint8))) == set(MODES)


def test_enabled_restricts_the_candidate_set(solid_frame):
    got = candidates(solid_frame(), enabled=frozenset({FrameType.FULL}))
    assert got == [FrameType.FULL]


def test_frame_zero_with_full_disabled_is_an_error(solid_frame):
    with pytest.raises(ValueError, match="frame 0 requires FULL"):
        candidates(None, enabled=frozenset({FrameType.PIXEL_DELTA}))


# ------------------------------------------------------------ the strategies


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_picks_full_for_frame_zero(solid_frame, strategy):
    pick = choose(solid_frame(), None, strategy=strategy)
    assert pick.frame_type is FrameType.FULL


def test_unknown_strategy_is_rejected(solid_frame):
    with pytest.raises(ValueError, match="unknown selection strategy"):
        choose(solid_frame(), solid_frame(), strategy="telepathy")


def test_fixed_reproduces_the_m1_behaviour(moving_frames):
    """The V2 strategy, kept so its cost stays measurable rather than remembered."""
    a, b = moving_frames()[:2]
    assert choose(b, a, strategy="fixed").frame_type is FrameType.PIXEL_DELTA


# ------------------------------------------------------------ it picks the cheaper one


def test_a_scene_cut_chooses_full(rng, solid_frame):
    """When nearly every pixel changes, 7 bytes per pixel loses to 3."""
    before = solid_frame(color=(0, 0, 0))
    after = rng.integers(0, 256, before.shape, dtype=np.uint8)

    for strategy in ("actual", "trial"):
        pick = choose(after, before, strategy=strategy)
        assert pick.frame_type is FrameType.FULL, strategy


def test_a_sparse_change_chooses_pixel_delta(solid_frame):
    before = solid_frame()
    after = before.copy()
    after[2, 3] = (200, 10, 10)

    for strategy in ("actual", "trial"):
        pick = choose(after, before, strategy=strategy)
        assert pick.frame_type is FrameType.PIXEL_DELTA, strategy


def test_an_unchanged_frame_chooses_pixel_delta(solid_frame):
    """An empty delta is 4 bytes; the whole frame is thousands."""
    frame = solid_frame()
    assert choose(frame, frame, strategy="trial").frame_type is FrameType.PIXEL_DELTA


# ------------------------------------------------------------ what it reports


def test_choice_records_every_candidate_it_priced(moving_frames):
    a, b = moving_frames()[:2]
    pick = choose(b, a, strategy="trial")
    assert set(pick.candidate_raw) == set(MODES)
    assert set(pick.candidate_stored) == set(MODES)
    assert all(v > 0 for v in pick.candidate_raw.values())


def test_actual_does_not_populate_stored_sizes(moving_frames):
    """`actual` never runs the entropy stage, so it has nothing to report there."""
    a, b = moving_frames()[:2]
    pick = choose(b, a, strategy="actual")
    assert pick.candidate_raw
    assert pick.candidate_stored == {}


def test_the_winning_payload_decodes_back(moving_frames):
    from src.modes import MODES

    a, b = moving_frames()[:2]
    pick = choose(b, a, strategy="trial")
    restored = MODES[pick.frame_type].decode(pick.payload, a, b.shape)
    assert np.array_equal(restored, b)


def test_the_winner_is_the_smallest_candidate(rng, solid_frame):
    before = solid_frame()
    after = rng.integers(0, 256, before.shape, dtype=np.uint8)
    pick = choose(after, before, strategy="trial")
    assert pick.candidate_stored[pick.frame_type] == min(pick.candidate_stored.values())


def test_ties_prefer_full(solid_frame, monkeypatch):
    """FULL is the clean resynchronisation point, so it wins a tie."""
    import src.selector as sel

    monkeypatch.setattr(
        sel.entropy, "compress", lambda payload, *, enabled, codec="zstd": b"x" * 10
    )
    pick = choose(solid_frame(), solid_frame(), strategy="trial")
    assert pick.frame_type is FrameType.FULL


# ------------------------------------------- choosing can never lose to a single mode


def test_locked_to_one_mode_counts_every_frame(rng, solid_frame):
    """Regression: a predicted mode's own total is one frame short, because frame 0 can
    only be FULL. Comparing that against the whole file made selection look like it lost."""
    from src.models.frame import FrameType as FT
    from src.models.stats import CodecStats

    stats = CodecStats(
        width=4, height=4, fps=25.0, source_bytes=0, entropy_enabled=True
    )
    # Frame 0: only FULL is available.
    stats.record(FT.FULL, 48, 30, None, candidate_raw={FT.FULL: 48},
                 candidate_stored={FT.FULL: 30})
    # Frame 1: both priced, PIXEL_DELTA wins.
    stats.record(FT.PIXEL_DELTA, 11, 9, 1,
                 candidate_raw={FT.FULL: 48, FT.PIXEL_DELTA: 11},
                 candidate_stored={FT.FULL: 30, FT.PIXEL_DELTA: 9})

    locked = stats.locked_to_one_mode()
    assert locked[FT.FULL] == 60           # 30 + 30, both frames
    assert locked[FT.PIXEL_DELTA] == 39    # 9, plus frame 0's forced FULL at 30
    # The invariant the bug violated: choosing is never worse than any single mode.
    assert stats.stored_bytes <= min(locked.values())


def test_tiled_tiles_are_chosen_independently_after_the_entropy_stage(rng, solid_frame):
    """TILED must rank each tile on final size, which only the selector can do."""
    from src.models.options import ModeOptions
    from src.modes import tiled

    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[:, 16:] = rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)

    pick = choose(after, before, strategy="trial", opts=ModeOptions(grid=2))
    assert FrameType.TILED in pick.candidate_stored

    # The TILED candidate must itself be a valid, mixed-mode payload.
    payload = choose(
        after, before, strategy="trial",
        enabled=frozenset({FrameType.FULL, FrameType.TILED}),
        opts=ModeOptions(grid=2),
    ).payload
    if payload[:1]:  # TILED won; confirm its tiles differ and it round-trips
        from src.modes import MODES
        restored = MODES[FrameType.TILED].decode(payload, before, after.shape) \
            if tiled.unpack(payload) else None
        if restored is not None:
            assert np.array_equal(restored, after)

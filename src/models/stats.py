"""Size accounting for one encode run.

Deliberately NOT frozen, unlike every other model here: this one accumulates as the encoder
runs. It still belongs in models/ because it is a data object — rendering it as a report is
src/stats.py's job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.frame import FrameType


class CodecStats(BaseModel):
    width: int
    height: int
    fps: float
    source_bytes: int
    entropy_enabled: bool
    entropy_codec: str = "zstd"
    #: Name of the pixel format the frames were coded in, and a note when it is not the
    #: source's own. See src.pixfmt.for_source.
    pix_fmt: str = "rgb24"
    pix_fmt_note: str | None = None
    #: Bytes in one raw frame, across every plane. Supplied by the encoder from the pixel
    #: format rather than computed here: a subsampled format is not width*height*channels,
    #: and this used to hardcode a 3 that only RGB24 ever justified.
    frame_nbytes: int = 0
    #: Header plus per-frame record overhead, so the report matches the file on disk.
    container_bytes: int = 0

    frames: int = 0
    #: Payload sizes before the entropy stage.
    raw_bytes: int = 0
    #: Payload sizes as stored.
    stored_bytes: int = 0
    changed_pixels: int = 0
    mode_counts: dict[FrameType, int] = Field(default_factory=dict)
    #: For a multi-plane format, the modes chosen *inside* each PLANES frame. Counted
    #: separately rather than folded into mode_counts, so neither number has to be read
    #: as "frames, except when it means planes".
    plane_mode_counts: dict[FrameType, int] = Field(default_factory=dict)

    #: How modes were ranked. See src.selector.STRATEGIES.
    selection: str = "trial"
    #: BLOCK_DELTA's grid cell side length, recorded so a sweep's rows are labelled.
    block_size: int = 16
    #: Summed pre-entropy size per mode across every frame where it was *priced*, won or
    #: lost. This is what makes "if locked to a single mode" answerable.
    candidate_raw: dict[FrameType, int] = Field(default_factory=dict)
    #: The same, after the entropy stage. Only the `trial` strategy fills it in.
    candidate_stored: dict[FrameType, int] = Field(default_factory=dict)
    #: How many frames each mode was *priced* on. Predicted modes are unavailable on
    #: frame 0, so their totals cover one frame fewer — which makes a naive comparison
    #: against the whole file apples-to-oranges.
    candidate_frames: dict[FrameType, int] = Field(default_factory=dict)
    #: Stored size of frame 0, which is always FULL. A file "locked to PIXEL_DELTA" is
    #: impossible — frame 0 has nothing to predict from — so this is the floor any
    #: predicted mode has to pay before it describes a single later frame.
    first_frame_stored: int = 0
    #: Frames where ranking on pre-entropy size would have picked a different mode than
    #: ranking on final size. The measurement that justifies the `trial` default.
    ranking_disagreements: int = 0

    @property
    def total_bytes(self) -> int:
        return self.stored_bytes + self.container_bytes

    def locked_to_one_mode(self) -> dict[FrameType, int]:
        """What the file would have cost locked to each single mode, over *all* frames.

        A predicted mode is unavailable on frame 0, so its own total is one frame short.
        Comparing that against the whole file would flatter it — and did, until the report
        showed selection apparently losing to a single mode, which cannot happen when the
        chosen payload is the per-frame minimum. Frame 0's forced FULL is added back, so
        every total covers the same frames and the comparison means something.
        """
        priced = self.candidate_stored or self.candidate_raw
        out: dict[FrameType, int] = {}
        for mode, size in priced.items():
            short = self.candidate_frames.get(mode, 0) < self.frames
            out[mode] = size + (self.first_frame_stored if short else 0)
        return out

    @property
    def raw_frame_bytes(self) -> int:
        """What the frames would cost uncompressed — the baseline worth beating."""
        return self.frame_nbytes * self.frames

    def record(
        self,
        frame_type: FrameType,
        raw: int,
        stored: int,
        changed: int | None = None,
        *,
        candidate_raw: dict[FrameType, int] | None = None,
        candidate_stored: dict[FrameType, int] | None = None,
        plane_modes: tuple[FrameType, ...] = (),
    ) -> None:
        if self.frames == 0:
            self.first_frame_stored = stored
        self.frames += 1
        self.raw_bytes += raw
        self.stored_bytes += stored
        if changed is not None:
            self.changed_pixels += changed
        self.mode_counts[frame_type] = self.mode_counts.get(frame_type, 0) + 1
        for mode in plane_modes:
            self.plane_mode_counts[mode] = self.plane_mode_counts.get(mode, 0) + 1

        for mode, size in (candidate_raw or {}).items():
            self.candidate_raw[mode] = self.candidate_raw.get(mode, 0) + size
            self.candidate_frames[mode] = self.candidate_frames.get(mode, 0) + 1
        for mode, size in (candidate_stored or {}).items():
            self.candidate_stored[mode] = self.candidate_stored.get(mode, 0) + size

        # Would ranking on pre-entropy size have chosen differently? Only answerable when
        # both sets are present, i.e. under `trial`.
        if candidate_raw and candidate_stored and len(candidate_raw) > 1:
            by_raw = min(candidate_raw, key=lambda m: (candidate_raw[m], int(m)))
            by_stored = min(candidate_stored, key=lambda m: (candidate_stored[m], int(m)))
            if by_raw is not by_stored:
                self.ranking_disagreements += 1

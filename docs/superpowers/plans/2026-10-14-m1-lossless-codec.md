# M0 + M1: Foundations and the V2 Lossless Codec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working lossless video codec that encodes an MP4 to `.myv` and back with byte-identical frames, using two competing representations (`FULL` and `PIXEL_DELTA`), driven by a CLI.

**Architecture:** Frames stream through ffmpeg subprocess pipes as numpy arrays, so memory stays at two frames regardless of video length. Each frame is encoded by one *mode* — a pure `encode`/`decode`/`cost` triple — and the chosen mode id is recorded in the frame container record, so the decoder follows instructions rather than re-deriving decisions. Pydantic frozen models in `src/models/` hold every value object; the byte-level packing that serialises them lives in `src/format.py`, keeping models free of format concerns.

**Tech Stack:** Python 3.12+, numpy (frame arrays and vectorised deltas), pydantic v2 (value objects), zstandard (entropy stage), ffmpeg/ffprobe via subprocess, pytest, ruff, uv.

**Spec:** `docs/ROADMAP.md` — milestones M0 and M1, including the reference targets this must hit.

## Global Constraints

- **Python** `>=3.12`. Every module starts with `from __future__ import annotations`.
- **Losslessness is the invariant.** Every decoded frame must be byte-identical to the ffmpeg-decoded source frame. A task that breaks this has failed, regardless of size wins.
- **Package root is `src/`.** All imports are absolute from it: `from src.models.video import VideoInfo`. Never relative imports.
- **Models are pydantic.** Value objects are `BaseModel` with `model_config = ConfigDict(frozen=True)`, and they live in `src/models/`. Logic never lives in `src/models/`.
- **Format constants are not model state.** Magic bytes, version numbers and struct layouts live in `src/format.py`, not on the models they serialise.
- **File-format numbers are permanent.** `FrameType` values, `PixelFormat.id` values and the header layout are written to disk. Never reassign an existing value; only append.
- **ffmpeg and ffprobe** must be resolved with `shutil.which` and produce a clear install message when missing — never a bare `FileNotFoundError` from `subprocess`.
- **No padding a short read.** A partial frame from a pipe is an error, never zero-filled; padding would corrupt the final frame and break losslessness invisibly.
- **Modular arithmetic for deltas.** uint8 subtraction wraps, and that is what makes delta modes lossless. Never clamp, never widen to int16.
- **Line length 96.** Ruff with `select = ["E", "F", "I"]` must pass clean.
- **Commit after every task** with a message naming the observable effect.

## File Structure

Created or modified by this plan. One responsibility each.

| file | responsibility |
| --- | --- |
| `pyproject.toml` | deps, console script, pytest and ruff config |
| `tests/conftest.py` | shared fixtures: synthetic frames, and a real MP4 built by ffmpeg directly |
| `src/models/pixel_format.py` | **modify** — `PixelFormat` gains `frame_shape` / `frame_nbytes`; id→format lookup |
| `src/models/video.py` | **modify** — `num_frames` is an optional hint; add `fps` |
| `src/models/header.py` | **modify** — drop `file_identifier`, rename `fps_count`→`frame_count`, hold a `PixelFormat` |
| `src/models/frame.py` | **modify** — add `PIXEL_DELTA` to `FrameType` |
| `src/models/stats.py` | **create** — accumulator for size accounting |
| `src/models/verify.py` | **create** — `Mismatch`, `VerifyResult` |
| `src/utils/errors.py` | **create** — `FFmpegError`, `BadMyvFile` |
| `src/utils/ffmpeg.py` | **modify** — fix `probe`; add `decode_frames`, `encode_frames` |
| `src/utils/entropy.py` | **create** — the zstd stage, isolated so sizes report with and without it |
| `src/utils/sizes.py` | **create** — `human_bytes` |
| `src/modes/full.py` | **implement** — store the whole frame |
| `src/modes/pixel_delta.py` | **create** — changed pixels as coordinates + modular deltas |
| `src/modes/__init__.py` | **implement** — the `MODES` registry and `PREDICTED` set |
| `src/format.py` | **implement** — header packing, `MyvWriter`, `MyvReader` |
| `src/encoder.py` | **implement** — MP4 → `.myv` |
| `src/decoder.py` | **implement** — `.myv` → frames |
| `src/verify.py` | **implement** — prove the round-trip is exact |
| `src/stats.py` | **implement** — render the size report |
| `src/cli.py` | **implement** — `encode` / `decode` / `info` / `verify` |

`src/selector.py` and `src/pixfmt.py` stay as docstring stubs until M2 and the YUV work respectively. Leave them in place.

`tests/` mirrors `src/`: create `tests/models/`, `tests/modes/` and `tests/utils/`. Do **not** add `__init__.py` to them — every test file basename in this plan is unique, so pytest's rootdir-relative collection handles them, and `pythonpath = ["tests"]` is what makes `from conftest import requires_ffmpeg` work from a subdirectory.

---

## M0 — Foundations

### Task 1: Project foundations — dependencies and test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/test_foundations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: fixtures `rng`, `solid_frame`, `moving_frames`, `write_clip` and the `requires_ffmpeg` marker, used by every later test; a `vidc` console script name.

- [ ] **Step 1: Replace the dependency block in `pyproject.toml`**

`tool>=0.8.0` is an inert PyPI package that does nothing here — drop it.

```toml
dependencies = [
    "numpy>=2.0",
    "pydantic>=2.13.5",
    "zstandard>=0.22",
]

[project.scripts]
vidc = "src.cli:main"
```

- [ ] **Step 2: Append pytest and ruff configuration to `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
markers = [
    "samples: round-trips real footage from samples/; skipped when the folder is empty",
]

[tool.ruff]
target-version = "py312"
line-length = 96

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.ruff.lint.isort]
known-first-party = ["src", "conftest"]
```

- [ ] **Step 3: Write `tests/conftest.py`**

`write_clip` shells out to ffmpeg directly rather than through `src.utils.ffmpeg`. Fixtures must not depend on the code under test, or a bug in the encoder silently becomes a bug in every fixture.

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

WIDTH, HEIGHT = 32, 24

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def solid_frame():
    def make(color=(10, 20, 30), height=HEIGHT, width=WIDTH) -> np.ndarray:
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :] = color
        return frame

    return make


@pytest.fixture
def moving_frames(solid_frame):
    """Four frames: a 4x4 white square stepping right across a dark background."""

    def make(count: int = 4) -> list[np.ndarray]:
        out = []
        for i in range(count):
            frame = solid_frame()
            frame[8:12, 2 + i * 4 : 6 + i * 4] = 255
            out.append(frame)
        return out

    return make


@pytest.fixture
def write_clip(tmp_path):
    """Write frames to a lossless RGB MP4 with ffmpeg and return its path.

    libx264rgb, not libx264 -qp 0: the latter converts RGB->YUV before encoding and that
    conversion rounds, so the clip would not decode back to the frames we wrote.
    """

    def make(frames: list[np.ndarray], fps: int = 25, name: str = "clip.mp4") -> Path:
        path = tmp_path / name
        height, width = frames[0].shape[:2]
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast",
            "-pix_fmt", "rgb24",
            str(path),
        ]
        payload = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
        result = subprocess.run(cmd, input=payload, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return path

    return make
```

- [ ] **Step 4: Write `tests/test_foundations.py`**

```python
from __future__ import annotations

import numpy as np
from conftest import requires_ffmpeg


def test_moving_frames_differ(moving_frames):
    frames = moving_frames()
    assert len(frames) == 4
    assert all(f.shape == (24, 32, 3) and f.dtype == np.uint8 for f in frames)
    assert not np.array_equal(frames[0], frames[1])


@requires_ffmpeg
def test_write_clip_produces_a_file(write_clip, moving_frames):
    path = write_clip(moving_frames())
    assert path.exists() and path.stat().st_size > 0
```

- [ ] **Step 5: Run the tests**

Run: `uv sync && uv run pytest -v`
Expected: both PASS. If ffmpeg is absent the second SKIPS rather than fails.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/test_foundations.py
git commit -m "Add numpy, zstandard and a test harness; drop the inert 'tool' dependency"
```

---

### Task 2: PixelFormat answers layout questions

**Files:**
- Modify: `src/models/pixel_format.py`
- Create: `tests/models/test_pixel_format.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PixelFormat.frame_shape(width, height) -> tuple[int, int, int]`, `PixelFormat.frame_nbytes(width, height) -> int`, `RGB24`, `by_id(fmt_id: int) -> PixelFormat`. Later tasks ask `PixelFormat` for layout instead of hardcoding `3`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest

from src.models.pixel_format import RGB24, by_id


def test_rgb24_layout():
    assert RGB24.frame_shape(320, 240) == (240, 320, 3)
    assert RGB24.frame_nbytes(320, 240) == 320 * 240 * 3


def test_by_id_round_trips():
    assert by_id(RGB24.id) is RGB24


def test_by_id_rejects_unknown():
    with pytest.raises(ValueError, match="unknown pixel format id 99"):
        by_id(99)


def test_pixel_format_is_frozen():
    with pytest.raises(Exception):
        RGB24.channels = 4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/models/test_pixel_format.py -v`
Expected: FAIL — `ImportError: cannot import name 'by_id'`

- [ ] **Step 3: Extend `src/models/pixel_format.py`**

Keep the existing class and `RGB24`; add the two methods and the lookup. Replace the trailing `# Eventually add YUV` comment with the note below, which records *why* this indirection exists.

```python
    def frame_shape(self, width: int, height: int) -> tuple[int, int, int]:
        """Numpy shape of one frame: rows, columns, channels."""
        return (height, width, self.channels)

    def frame_nbytes(self, width: int, height: int) -> int:
        return width * height * self.channels


RGB24 = PixelFormat(id=0, name="rgb24", ffmpeg_name="rgb24", channels=3)

# `id` is stored in the .myv header, so these values are part of the file format and must
# never be reassigned. Adding YUV later means adding an instance here and reading
# `channels` instead of grepping for the number 3.
_BY_ID = {fmt.id: fmt for fmt in (RGB24,)}


def by_id(fmt_id: int) -> PixelFormat:
    """Map a stored header byte back to a format. Raises rather than returning None."""
    try:
        return _BY_ID[fmt_id]
    except KeyError:
        raise ValueError(
            f"unknown pixel format id {fmt_id}; this file was probably written by a "
            f"newer version of the codec"
        ) from None
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/models/test_pixel_format.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/pixel_format.py tests/models/test_pixel_format.py
git commit -m "PixelFormat answers frame layout, so the RGB assumption lives in one place"
```

---

### Task 3: probe() survives real containers

**Files:**
- Create: `src/utils/errors.py`
- Modify: `src/utils/ffmpeg.py`
- Modify: `src/models/video.py`
- Create: `tests/utils/test_ffmpeg.py`

**Interfaces:**
- Consumes: `VideoInfo`.
- Produces: `FFmpegError`, `BadMyvFile`, `_require(tool) -> str`, `probe(path) -> VideoInfo`, `VideoInfo.fps -> float`.

The current `probe` has three crash paths: `int(stream.get("nb_frames"))` raises `TypeError` when the container omits `nb_frames` and `ValueError` when it reports `"N/A"` — both common — and `int(den)` raises on a rate like `"25"` with no slash. Frame rate stays a **rational**, never a float, so 30000/1001 survives exactly.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest
from conftest import requires_ffmpeg

from src.models.video import VideoInfo
from src.utils.errors import FFmpegError
from src.utils.ffmpeg import probe


@requires_ffmpeg
def test_probe_reads_dimensions_and_rate(write_clip, moving_frames):
    info = probe(write_clip(moving_frames(), fps=25))
    assert (info.width, info.height) == (32, 24)
    assert (info.fps_num, info.fps_den) == (25, 1)
    assert info.fps == 25.0


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(FFmpegError, match="no such file"):
        probe(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_probe_rejects_a_non_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(FFmpegError):
        probe(junk)


def test_video_info_frame_count_is_optional():
    info = VideoInfo(width=2, height=2, fps_num=30000, fps_den=1001)
    assert info.num_frames is None
    assert abs(info.fps - 29.97) < 0.01
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/utils/test_ffmpeg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.errors'`

- [ ] **Step 3: Create `src/utils/errors.py`**

```python
"""Exceptions shared across the codec."""

from __future__ import annotations


class FFmpegError(Exception):
    """ffmpeg or ffprobe failed, or isn't installed."""


class BadMyvFile(Exception):
    """The file isn't a .myv, or is corrupt, or came from a newer codec."""
```

- [ ] **Step 4: Add `fps` and document `num_frames` in `src/models/video.py`**

```python
    #: A hint from the container; often absent and sometimes wrong. Never a loop bound.
    num_frames: int | None = None

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den
```

- [ ] **Step 5: Rewrite `src/utils/ffmpeg.py`**

```python
"""The bridge to the outside world.

The codec never needs to understand MP4 — that's a container problem ffmpeg already solved.
This module is the only place that knows subprocesses exist; everything upstream of it deals
in numpy frames.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from src.models.video import VideoInfo
from src.utils.errors import FFmpegError


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegError(
            f"{tool} not found on PATH. This codec uses ffmpeg to read and write MP4; "
            f"install it from https://ffmpeg.org/download.html"
        )
    return path


def probe(path: str | Path) -> VideoInfo:
    """Read dimensions and frame rate from a media file's first video stream."""
    path = Path(path)
    if not path.exists():
        raise FFmpegError(f"no such file: {path}")

    cmd = [
        _require("ffprobe"), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed on {path}:\n{result.stderr.decode(errors='replace')}"
        )

    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise FFmpegError(f"{path} has no video stream")
    stream = streams[0]

    # A rational, not a float: 30000/1001 must survive the round-trip exactly.
    rate = stream.get("r_frame_rate") or "25/1"
    num, _, den = rate.partition("/")
    try:
        fps_num, fps_den = int(num), int(den or 1)
    except ValueError:
        fps_num, fps_den = 25, 1
    if fps_num <= 0 or fps_den <= 0:
        fps_num, fps_den = 25, 1  # the container lied; pick something sane and move on

    # nb_frames is absent on plenty of containers and "N/A" on plenty more.
    nb = stream.get("nb_frames")
    num_frames = int(nb) if nb not in (None, "N/A", "") else None

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        num_frames=num_frames,
    )
```

- [ ] **Step 6: Run it to verify it passes**

Run: `uv run pytest tests/utils/test_ffmpeg.py -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add src/utils/errors.py src/utils/ffmpeg.py src/models/video.py tests/utils/test_ffmpeg.py
git commit -m "Fix probe() crashing on containers that omit or fake nb_frames"
```

---

### Task 4: Stream frames in from a video file

**Files:**
- Modify: `src/utils/ffmpeg.py`
- Modify: `tests/utils/test_ffmpeg.py`

**Interfaces:**
- Consumes: `_require`, `probe`, `PixelFormat`, `RGB24`.
- Produces: `decode_frames(path, pix_fmt=RGB24, info=None)` — a `@contextmanager` yielding an iterator of `(h, w, c)` uint8 arrays.

Three details matter and all three are load-bearing. `np.frombuffer` is **zero-copy** and each frame is backed by its own `bytes` from `read()`, so callers may retain frames — the encoder depends on this, holding frame N−1 while working on N. A short-but-nonempty read is an **error**, never padded. And on early abandonment the process is **killed deliberately** rather than allowed to die of SIGPIPE, which would report failure for something we asked for.

- [ ] **Step 1: Write the failing test (append to `tests/utils/test_ffmpeg.py`)**

```python
@requires_ffmpeg
def test_decode_frames_round_trips_exactly(write_clip, moving_frames):
    from src.utils.ffmpeg import decode_frames

    original = moving_frames()
    with decode_frames(write_clip(original)) as frames:
        decoded = list(frames)

    assert len(decoded) == len(original)
    for want, got in zip(original, decoded):
        assert got.shape == (24, 32, 3)
        assert got.dtype == np.uint8
        assert np.array_equal(want, got)


@requires_ffmpeg
def test_decode_frames_survives_early_abandonment(write_clip, moving_frames):
    """Stopping after one frame must not raise: we asked ffmpeg to stop."""
    from src.utils.ffmpeg import decode_frames

    with decode_frames(write_clip(moving_frames())) as frames:
        first = next(iter(frames))
    assert first.shape == (24, 32, 3)


@requires_ffmpeg
def test_decoded_frames_are_independently_retainable(write_clip, moving_frames):
    """The encoder holds frame N-1 while reading N, so frames must not share a buffer."""
    from src.utils.ffmpeg import decode_frames

    with decode_frames(write_clip(moving_frames())) as frames:
        held = list(frames)
    assert not np.array_equal(held[0], held[-1])
```

Add `import numpy as np` to the test module imports.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/utils/test_ffmpeg.py -k decode -v`
Expected: FAIL — `ImportError: cannot import name 'decode_frames'`

- [ ] **Step 3: Add `decode_frames` to `src/utils/ffmpeg.py`**

Add `from contextlib import contextmanager`, `from typing import Iterator`, `import numpy as np`, and `from src.models.pixel_format import RGB24, PixelFormat` to the imports.

```python
@contextmanager
def decode_frames(
    path: str | Path, pix_fmt: PixelFormat = RGB24, info: VideoInfo | None = None
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of decoded frames as ``(h, w, channels)`` uint8 arrays.

    A context manager so the ffmpeg process is always reaped, even if the caller stops
    iterating early::

        with decode_frames("in.mp4") as frames:
            for frame in frames:
                ...
    """
    path = Path(path)
    info = info or probe(path)
    nbytes = pix_fmt.frame_nbytes(info.width, info.height)
    shape = pix_fmt.frame_shape(info.width, info.height)

    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin",
        "-i", str(path),
        "-map", "0:v:0",            # video only; audio is out of scope
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt.ffmpeg_name,
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    drained = False  # did we read all the way to end-of-stream?

    def read_frames() -> Iterator[np.ndarray]:
        nonlocal drained
        assert proc.stdout is not None
        index = 0
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                drained = True
                return
            if len(buf) != nbytes:
                # A partial frame means the stream was cut mid-frame. Never pad it out;
                # that would silently corrupt the last frame and break losslessness.
                raise FFmpegError(
                    f"{path}: frame {index} is short — got {len(buf)} of {nbytes} bytes"
                )
            # Zero-copy, and each frame owns its own buffer, so callers may retain it.
            yield np.frombuffer(buf, dtype=np.uint8).reshape(shape)
            index += 1

    try:
        yield read_frames()
    finally:
        if not drained:
            # The caller stopped early. Closing the pipe would make ffmpeg die of SIGPIPE
            # and report a nonzero exit, which isn't a failure worth reporting — we asked
            # it to stop. Kill it deliberately and don't check the exit code.
            proc.kill()
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.stderr:
            proc.stderr.close()
        code = proc.wait()
        if drained and code != 0:
            raise FFmpegError(
                f"ffmpeg failed decoding {path} (exit {code}):\n"
                f"{stderr.decode(errors='replace')}"
            )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/utils/test_ffmpeg.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/ffmpeg.py tests/utils/test_ffmpeg.py
git commit -m "Stream frames in from video over a pipe, two frames of memory at a time"
```

---

### Task 5: Stream frames out to a video file

**Files:**
- Modify: `src/utils/ffmpeg.py`
- Modify: `tests/utils/test_ffmpeg.py`

**Interfaces:**
- Consumes: `_require`, `PixelFormat`, `RGB24`.
- Produces: `encode_frames(path, frames, *, width, height, fps_num, fps_den, pix_fmt=RGB24, lossless=False) -> int` returning the frame count written.

- [ ] **Step 1: Write the failing test (append to `tests/utils/test_ffmpeg.py`)**

```python
@requires_ffmpeg
def test_encode_frames_lossless_round_trips_exactly(tmp_path, moving_frames):
    from src.utils.ffmpeg import decode_frames, encode_frames

    original = moving_frames()
    out = tmp_path / "out.mp4"
    written = encode_frames(
        out, original, width=32, height=24, fps_num=25, fps_den=1, lossless=True
    )
    assert written == len(original)

    with decode_frames(out) as frames:
        decoded = list(frames)
    for want, got in zip(original, decoded):
        assert np.array_equal(want, got)


@requires_ffmpeg
def test_encode_frames_reports_ffmpeg_stderr_on_failure(tmp_path, moving_frames):
    from src.utils.errors import FFmpegError
    from src.utils.ffmpeg import encode_frames

    with pytest.raises(FFmpegError):
        # A zero width is rejected by ffmpeg; the error must carry its stderr, not a
        # BrokenPipeError traceback.
        encode_frames(
            tmp_path / "bad.mp4", moving_frames(),
            width=0, height=0, fps_num=25, fps_den=1, lossless=True,
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/utils/test_ffmpeg.py -k encode -v`
Expected: FAIL — `ImportError: cannot import name 'encode_frames'`

- [ ] **Step 3: Add `encode_frames` to `src/utils/ffmpeg.py`**

Add `from typing import Iterable` to the imports.

```python
def encode_frames(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    pix_fmt: PixelFormat = RGB24,
    lossless: bool = False,
) -> int:
    """Pipe frames into ffmpeg to produce a playable MP4. Returns the frame count.

    With ``lossless=False`` the output is a re-encode and is not byte-identical to what went
    in — it's for watching. Proving the round-trip is exact is ``verify``'s job, and it does
    that by comparing arrays in memory instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if lossless:
        # libx264rgb encodes RGB directly. Plain libx264 -qp 0 is only lossless *after* the
        # RGB->YUV conversion, and that conversion rounds — so it would quietly cost a pixel
        # value here and there, which defeats the point of asking for lossless.
        codec = ["-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24"]
    else:
        # yuv420p so ordinary players and browsers can open the result.
        codec = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]

    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt.ffmpeg_name,
        "-s", f"{width}x{height}",
        "-r", f"{fps_num}/{fps_den}",
        "-i", "-",
        "-preset", "medium", *codec,
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    written = 0
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        # ffmpeg died early; its stderr says why, and that's far more useful than the
        # BrokenPipeError traceback.
        stderr = proc.stderr.read() if proc.stderr else b""
        proc.wait()
        raise FFmpegError(
            f"ffmpeg closed the pipe while writing {path}:\n"
            f"{stderr.decode(errors='replace')}"
        ) from None

    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.stderr:
        proc.stderr.close()
    code = proc.wait()
    if code != 0:
        raise FFmpegError(
            f"ffmpeg failed writing {path} (exit {code}):\n{stderr.decode(errors='replace')}"
        )
    return written
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v && uv run ruff check .`
Expected: 9 PASS, ruff clean. **M0 is complete: a working video I/O layer.**

- [ ] **Step 5: Commit**

```bash
git add src/utils/ffmpeg.py tests/utils/test_ffmpeg.py
git commit -m "Stream frames out to MP4, with a truly lossless RGB path"
```

---

## M1 — The V2 lossless codec

### Task 6: The entropy stage

**Files:**
- Create: `src/utils/entropy.py`
- Create: `tests/utils/test_entropy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `compress(payload: bytes, *, enabled: bool) -> bytes`, `decompress(payload: bytes, *, enabled: bool) -> bytes`, `LEVEL`.

Kept in its own module so reports can show pre-entropy and post-entropy sizes independently. Compression numbers only inform if you can tell what the *representation* achieved from what generic compression achieved — two different questions. This is the file that eventually becomes a hand-written range coder.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from src.utils import entropy


def test_round_trips_when_enabled():
    payload = b"a" * 1000
    stored = entropy.compress(payload, enabled=True)
    assert len(stored) < len(payload)
    assert entropy.decompress(stored, enabled=True) == payload


def test_is_identity_when_disabled():
    payload = b"abc"
    assert entropy.compress(payload, enabled=False) is payload
    assert entropy.decompress(payload, enabled=False) is payload


def test_round_trips_empty():
    stored = entropy.compress(b"", enabled=True)
    assert entropy.decompress(stored, enabled=True) == b""
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/utils/test_entropy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.entropy'`

- [ ] **Step 3: Write `src/utils/entropy.py`**

```python
"""The entropy-coding stage — the last box in the pipeline.

Separate from everything else so reports can show pre-entropy and post-entropy sizes
independently: what the codec's representation bought and what generic compression bought
are different questions, and conflating them hides which one is doing the work.
"""

from __future__ import annotations

import zstandard

#: Fixed, so size comparisons across runs and versions stay meaningful.
LEVEL = 10


def compress(payload: bytes, *, enabled: bool) -> bytes:
    """``enabled`` is keyword-only so a bare ``compress(data, True)`` can't happen."""
    if not enabled:
        return payload  # identity, not a copy
    return zstandard.ZstdCompressor(level=LEVEL).compress(payload)


def decompress(payload: bytes, *, enabled: bool) -> bytes:
    if not enabled:
        return payload
    return zstandard.ZstdDecompressor().decompress(payload)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/utils/test_entropy.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/entropy.py tests/utils/test_entropy.py
git commit -m "Add the zstd entropy stage, isolated so sizes report with and without it"
```

---

### Task 7: FULL — store the whole frame

**Files:**
- Implement: `src/modes/full.py`
- Create: `tests/modes/test_full.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `encode(cur, prev=None, opts=None) -> bytes`, `decode(payload, prev, shape) -> np.ndarray`, `cost(cur, prev=None, opts=None) -> int`.

Every mode module exposes the same three functions, which is what makes them interchangeable enough for the encoder to price several and keep the smallest. `decode` takes no options: a payload carries whatever parameters it needs in its own bytes, so a decoder is never configured to match an encoder.

At M1 the payload is the raw pixels. M8 adds intra prediction here, which changes the payload's *content* but not its length — so the closed-form `cost` survives that change.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pytest

from src.modes import full


def test_round_trips(moving_frames):
    frame = moving_frames()[0]
    payload = full.encode(frame)
    assert np.array_equal(full.decode(payload, None, frame.shape), frame)


def test_needs_no_reference_frame(moving_frames):
    frame = moving_frames()[0]
    assert full.encode(frame, None) == full.encode(frame, frame)


def test_cost_is_exact(moving_frames):
    frame = moving_frames()[0]
    assert full.cost(frame) == len(full.encode(frame))


def test_decode_rejects_a_wrong_length_payload(moving_frames):
    frame = moving_frames()[0]
    with pytest.raises(ValueError, match="expected"):
        full.decode(b"\x00\x01", None, frame.shape)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/modes/test_full.py -v`
Expected: FAIL — `AttributeError: module 'src.modes.full' has no attribute 'encode'`

- [ ] **Step 3: Implement `src/modes/full.py`**

```python
"""FULL — store the whole frame, predicting nothing from any other frame.

The baseline every other mode is measured against, and the only mode needing no reference
frame — so it is what frame 0 must be, what keyframes will be, and what wins whenever
prediction isn't worth it. On real footage that turns out to be very often.
"""

from __future__ import annotations

import numpy as np


def encode(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> bytes:
    """Serialise the frame. ``prev`` and ``opts`` exist for interface symmetry only."""
    return np.ascontiguousarray(cur).tobytes()


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    if len(payload) != expected:
        raise ValueError(
            f"FULL payload is {len(payload)} bytes, expected {expected} for shape {shape}"
        )
    return np.frombuffer(payload, dtype=np.uint8, count=expected).reshape(shape).copy()


def cost(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> int:
    """Encoded size in bytes, without doing the encoding. Exact, not an estimate."""
    return int(cur.size)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/modes/test_full.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/modes/full.py tests/modes/test_full.py
git commit -m "Add FULL mode: the baseline representation and what frame 0 must be"
```

---

### Task 8: PIXEL_DELTA — store only what changed

**Files:**
- Create: `src/modes/pixel_delta.py`
- Create: `tests/modes/test_pixel_delta.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `encode(cur, prev, opts=None) -> bytes`, `decode(payload, prev, shape) -> np.ndarray`, `cost(cur, prev, opts=None) -> int`, `count(payload) -> int`, `coordinates(payload) -> list[tuple[int, int]]`, `RECORD`, `MAX_COORD`.

This is the idea the project starts from: if a frame is mostly unchanged, describe the handful of pixels that moved rather than the millions that didn't.

Two things to understand. **The deltas are modular, and that is what makes it lossless** — a channel going 2 → 250 is −8 mod 256, not +248, so one byte covers every transition with no clamping and no escape hatch. numpy's uint8 arithmetic already wraps this way, so `cur - prev` and `prev + delta` are exact inverses for free. And **four of the seven bytes are coordinates** — that overhead is the mode's defining weakness, left in deliberately. It is what M3's `BLOCK_DELTA` exists to fix, and the size report is what will make the cost obvious.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pytest

from src.modes import pixel_delta


def test_record_is_seven_bytes():
    assert pixel_delta.RECORD.itemsize == 7


def test_round_trips(moving_frames):
    a, b = moving_frames()[:2]
    payload = pixel_delta.encode(b, a)
    assert np.array_equal(pixel_delta.decode(payload, a, b.shape), b)


def test_identical_frames_cost_only_the_header(solid_frame):
    frame = solid_frame()
    payload = pixel_delta.encode(frame, frame)
    assert pixel_delta.count(payload) == 0
    assert len(payload) == 4
    assert np.array_equal(pixel_delta.decode(payload, frame, frame.shape), frame)


def test_deltas_are_modular(solid_frame):
    """2 -> 250 is -8 mod 256. One byte must cover it without clamping."""
    before = solid_frame(color=(2, 2, 2))
    after = solid_frame(color=(250, 250, 250))
    payload = pixel_delta.encode(after, before)
    assert np.array_equal(pixel_delta.decode(payload, before, after.shape), after)


def test_counts_only_changed_pixels(solid_frame):
    before = solid_frame()
    after = before.copy()
    after[3, 4] = (99, 99, 99)
    after[5, 6, 1] = 7  # a single changed channel still makes the pixel changed
    payload = pixel_delta.encode(after, before)
    assert pixel_delta.count(payload) == 2
    assert set(pixel_delta.coordinates(payload)) == {(3, 4), (5, 6)}


def test_cost_is_exact(moving_frames):
    a, b = moving_frames()[:2]
    assert pixel_delta.cost(b, a) == len(pixel_delta.encode(b, a))


def test_encode_needs_a_reference_frame(solid_frame):
    with pytest.raises(ValueError, match="reference frame"):
        pixel_delta.encode(solid_frame(), None)


def test_encode_rejects_a_shape_mismatch(solid_frame):
    with pytest.raises(ValueError, match="shape mismatch"):
        pixel_delta.encode(solid_frame(), solid_frame(height=8))


def test_decode_rejects_a_truncated_payload(solid_frame):
    frame = solid_frame()
    payload = bytearray(pixel_delta.encode(frame, frame))
    payload[0] = 5  # claim five records that aren't there
    with pytest.raises(ValueError, match="claims 5 records"):
        pixel_delta.decode(bytes(payload), frame, frame.shape)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/modes/test_pixel_delta.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.modes.pixel_delta'`

- [ ] **Step 3: Write `src/modes/pixel_delta.py`**

```python
"""PIXEL_DELTA — store only the pixels that changed, as coordinates plus RGB offsets.

Payload layout::

    count : uint32
    count records of 7 bytes each:
        x  : uint16
        y  : uint16
        dR : uint8
        dG : uint8
        dB : uint8

The deltas are modular, which is what makes this lossless: uint8 subtraction wraps, so
``cur - prev`` and ``prev + delta`` are exact inverses and one byte covers every possible
transition. No clamping, no int16, no escape hatch for large jumps.

Four of the seven bytes are coordinates. That overhead is this mode's defining weakness and
is left in on purpose — a changed pixel costs 7 bytes no matter how small its colour change,
so a frame where 40% of pixels shifted by 1 costs nearly 2x what storing the whole frame
would. Fixing it by spatial grouping is what BLOCK_DELTA is for.
"""

from __future__ import annotations

import numpy as np

#: One changed pixel. Packed, not aligned, so it is exactly 7 bytes on disk.
RECORD = np.dtype([("x", "<u2"), ("y", "<u2"), ("d", "u1", (3,))])
assert RECORD.itemsize == 7, "record layout must stay 7 bytes; the format depends on it"

MAX_COORD = 0xFFFF
_HEADER = 4  # the uint32 count


def encode(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> bytes:
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame to diff against")
    if cur.shape != prev.shape:
        raise ValueError(f"shape mismatch: current {cur.shape} vs reference {prev.shape}")

    height, width = cur.shape[:2]
    if width - 1 > MAX_COORD or height - 1 > MAX_COORD:
        raise ValueError(
            f"frame is {width}x{height}; PIXEL_DELTA coordinates are uint16 and cannot "
            f"address beyond {MAX_COORD}"
        )

    ys, xs = np.nonzero(np.any(cur != prev, axis=2))
    # uint8 subtraction wraps, which is precisely the mod-256 delta we want.
    deltas = cur[ys, xs] - prev[ys, xs]

    records = np.empty(ys.size, dtype=RECORD)
    records["x"] = xs.astype(np.uint16, copy=False)
    records["y"] = ys.astype(np.uint16, copy=False)
    records["d"] = deltas

    return int(ys.size).to_bytes(_HEADER, "little") + records.tobytes()


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    n = count(payload)
    expected = _HEADER + n * RECORD.itemsize
    if len(payload) != expected:
        raise ValueError(
            f"PIXEL_DELTA claims {n} records ({expected} bytes) but payload is "
            f"{len(payload)} bytes"
        )
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame to reconstruct against")

    out = prev.copy()
    if n:
        records = np.frombuffer(payload, dtype=RECORD, count=n, offset=_HEADER)
        # Each pixel appears at most once, so this read-modify-write is unambiguous.
        out[records["y"], records["x"]] += records["d"]
    return out


def count(payload: bytes) -> int:
    """How many pixels this payload says changed."""
    if len(payload) < _HEADER:
        raise ValueError(f"PIXEL_DELTA payload is {len(payload)} bytes; needs at least 4")
    return int.from_bytes(payload[:_HEADER], "little")


def coordinates(payload: bytes) -> list[tuple[int, int]]:
    """The ``(y, x)`` of every changed pixel, in stored order. For tests and debugging."""
    n = count(payload)
    records = np.frombuffer(payload, dtype=RECORD, count=n, offset=_HEADER)
    return [(int(y), int(x)) for y, x in zip(records["y"], records["x"])]


def cost(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> int:
    """Encoded size in bytes, without building the payload. Exact, not an estimate."""
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame")
    changed = int(np.count_nonzero(np.any(cur != prev, axis=2)))
    return _HEADER + changed * RECORD.itemsize
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/modes/test_pixel_delta.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add src/modes/pixel_delta.py tests/modes/test_pixel_delta.py
git commit -m "Add PIXEL_DELTA mode: modular per-pixel deltas, lossless by construction"
```

---

### Task 9: The mode registry

**Files:**
- Modify: `src/models/frame.py`
- Implement: `src/modes/__init__.py`
- Create: `tests/modes/test_registry.py`

**Interfaces:**
- Consumes: `full`, `pixel_delta`.
- Produces: `FrameType.FULL = 0`, `FrameType.PIXEL_DELTA = 1`, `MODES: dict[FrameType, module]`, `PREDICTED: frozenset[FrameType]`.

`FrameType` values are stored as a byte in each frame record, so they are part of the file format. Leave gaps for the modes the roadmap plans, so their arrival never renumbers anything already written to disk.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from src.models.frame import FrameType
from src.modes import MODES, PREDICTED


def test_every_frame_type_has_a_mode():
    assert set(MODES) == set(FrameType)


def test_every_mode_exposes_the_same_interface():
    for frame_type, mode in MODES.items():
        for name in ("encode", "decode", "cost"):
            assert callable(getattr(mode, name)), f"{frame_type.name}.{name}"


def test_frame_type_values_are_stable():
    """These are written to disk. Changing one silently breaks every existing file."""
    assert FrameType.FULL == 0
    assert FrameType.PIXEL_DELTA == 1


def test_only_full_needs_no_reference_frame():
    assert PREDICTED == frozenset({FrameType.PIXEL_DELTA})
    assert FrameType.FULL not in PREDICTED
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/modes/test_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'MODES'`

- [ ] **Step 3: Add `PIXEL_DELTA` to `src/models/frame.py`**

```python
class FrameType(IntEnum):
    """Stored as a byte in each frame record — values are part of the file format.

    Gaps are reserved for the modes the roadmap plans, so their eventual arrival never
    renumbers anything already written to disk.
    """

    FULL = 0
    PIXEL_DELTA = 1
    # 2 reserved for BLOCK_DELTA (M3), 3 for MOTION (M4), 4 for TILED (M5)
```

- [ ] **Step 4: Implement `src/modes/__init__.py`**

```python
"""Encoding modes — the competing representations of "what changed".

No single representation is always best. Each mode here is one answer to "how do I describe
this frame most cheaply", and they are kept deliberately interchangeable so the encoder can
price several and keep the smallest.

Every mode module exposes the same three functions::

    encode(cur, prev, opts) -> bytes
    decode(payload, prev, shape) -> ndarray
    cost(cur, prev, opts) -> int          # encoded size, ideally without encoding

``prev`` is ``None`` for modes that don't predict from a reference frame.

``decode`` takes no options: every mode's payload carries whatever parameters it needs in
its own bytes. That keeps the container format stable as modes gain knobs, and means a
decoder never has to be configured to match an encoder.
"""

from __future__ import annotations

from src.models.frame import FrameType
from src.modes import full, pixel_delta

MODES = {
    FrameType.FULL: full,
    FrameType.PIXEL_DELTA: pixel_delta,
}

#: Modes that reconstruct from a reference frame, so are unavailable for frame 0.
PREDICTED = frozenset({FrameType.PIXEL_DELTA})

__all__ = ["MODES", "PREDICTED", "FrameType", "full", "pixel_delta"]
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/modes/ -v`
Expected: 17 PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/frame.py src/modes/__init__.py tests/modes/test_registry.py
git commit -m "Add the mode registry, so modes are interchangeable by construction"
```

---

### Task 10: The .myv container

**Files:**
- Modify: `src/models/header.py`
- Implement: `src/format.py`
- Create: `tests/test_format.py`

**Interfaces:**
- Consumes: `Header`, `PixelFormat`, `by_id`, `FrameType`, `Frame`, `BadMyvFile`.
- Produces: `MAGIC`, `VERSION`, `HEADER_SIZE`, `FRAME_HEADER_SIZE`, `pack_header(header) -> bytes`, `unpack_header(data) -> Header`, `MyvWriter(stream, header)` with `.frame(frame_type, payload)` / `.close()` / context-manager support, `MyvReader(stream)` with `.header` and `.frames() -> Iterator[Frame]`.

Three changes to the existing `Header` model, all cheap now and painful once files exist:

- **Drop `file_identifier`.** Magic bytes are a format constant, not model state; they belong in `format.py` beside the struct that writes them.
- **Rename `fps_count` → `frame_count`.** It is the frame total, back-patched when the writer closes — nothing to do with fps.
- **Hold a `PixelFormat`, not an id.** Pydantic nests models fine, and it lets `Header` answer `frame_shape` directly. `pack_header` writes `header.pix_fmt.id`.

The magic is `b"MYV1"` — four bytes with a trailing version digit, so a future incompatible format can be told apart at a glance rather than only by the version byte.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import io

import pytest

from src.format import (
    HEADER_SIZE,
    MyvReader,
    MyvWriter,
    pack_header,
    unpack_header,
)
from src.models.frame import FrameType
from src.models.header import Header
from src.utils.errors import BadMyvFile


def a_header(**kw) -> Header:
    base = dict(width=32, height=24, fps_num=25, fps_den=1, frame_count=0)
    return Header(**{**base, **kw})


def test_header_round_trips():
    header = a_header(frame_count=7)
    restored = unpack_header(pack_header(header))
    assert restored == header
    assert restored.fps == 25.0
    assert restored.frame_shape == (24, 32, 3)


def test_header_is_a_fixed_size():
    assert len(pack_header(a_header())) == HEADER_SIZE == 28


def test_unpack_rejects_bad_magic():
    with pytest.raises(BadMyvFile, match="not a .myv"):
        unpack_header(b"XXXX" + bytes(HEADER_SIZE - 4))


def test_unpack_rejects_a_short_header():
    with pytest.raises(BadMyvFile, match="too short"):
        unpack_header(b"MYV1")


def test_writer_back_patches_the_frame_count():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"abc")
        writer.frame(FrameType.PIXEL_DELTA, b"de")

    stream.seek(0)
    reader = MyvReader(stream)
    assert reader.header.frame_count == 2
    assert [(f.type, f.payload) for f in reader.frames()] == [
        (FrameType.FULL, b"abc"),
        (FrameType.PIXEL_DELTA, b"de"),
    ]


def test_reader_rejects_an_unknown_frame_type():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"x")
    data = bytearray(stream.getvalue())
    data[HEADER_SIZE] = 99  # the frame type byte
    with pytest.raises(BadMyvFile, match="unknown frame type 99"):
        list(MyvReader(io.BytesIO(bytes(data))).frames())


def test_reader_rejects_a_truncated_payload():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"abcdef")
    truncated = stream.getvalue()[:-3]
    with pytest.raises(BadMyvFile, match="truncated"):
        list(MyvReader(io.BytesIO(truncated)).frames())


def test_entropy_flag_survives_the_round_trip():
    assert unpack_header(pack_header(a_header(entropy=False))).entropy is False
    assert unpack_header(pack_header(a_header(entropy=True))).entropy is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_format.py -v`
Expected: FAIL — `ImportError: cannot import name 'pack_header'`

- [ ] **Step 3: Rewrite `src/models/header.py`**

```python
"""The .myv file header as a value object. Byte packing lives in src/format.py."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.pixel_format import RGB24, PixelFormat


class Header(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_num: int = Field(gt=0)
    fps_den: int = Field(gt=0)
    #: Back-patched when the writer closes; 0 while a file is still being written.
    frame_count: int = Field(default=0, ge=0)
    pix_fmt: PixelFormat = RGB24
    #: Whether frame payloads are entropy-coded. Stored as a header flag bit.
    entropy: bool = True

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.pix_fmt.frame_shape(self.width, self.height)

    @property
    def frame_nbytes(self) -> int:
        return self.pix_fmt.frame_nbytes(self.width, self.height)
```

- [ ] **Step 4: Implement `src/format.py`**

```python
"""The .myv container: a header, then one record per frame.

Deliberately minimal. There is no frame index yet, because without keyframes every seek
would still have to decode from frame 0 — an index would cost bytes and buy nothing. The
``version`` byte is what lets one be added later without breaking existing files.

::

    Header (28 bytes, little-endian)
      magic        4   b"MYV1"
      version      1
      pix_fmt      1   see src.models.pixel_format
      flags        1   bit0 = payloads are entropy-coded
      reserved     1
      width        4
      height       4
      fps_num      4
      fps_den      4   frame rate as a rational, so 30000/1001 stays exact
      frame_count  4   back-patched when the writer closes

    Frame record, repeated frame_count times
      frame_type   1   see src.models.frame.FrameType
      payload_len  4   length as stored, i.e. after entropy coding
      payload      payload_len bytes
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Iterator

from src.models.frame import Frame, FrameType
from src.models.header import Header
from src.models.pixel_format import by_id
from src.utils.errors import BadMyvFile

MAGIC = b"MYV1"
VERSION = 1

_HEADER = struct.Struct("<4sBBBBIIIII")
HEADER_SIZE = _HEADER.size

_FRAME_HEADER = struct.Struct("<BI")
FRAME_HEADER_SIZE = _FRAME_HEADER.size

_FLAG_ENTROPY = 0b1

#: Byte offset of frame_count within the header, for back-patching.
_FRAME_COUNT_OFFSET = HEADER_SIZE - 4


def pack_header(header: Header) -> bytes:
    return _HEADER.pack(
        MAGIC,
        VERSION,
        header.pix_fmt.id,
        _FLAG_ENTROPY if header.entropy else 0,
        0,  # reserved
        header.width,
        header.height,
        header.fps_num,
        header.fps_den,
        header.frame_count,
    )


def unpack_header(data: bytes) -> Header:
    if len(data) < HEADER_SIZE:
        raise BadMyvFile(f"header is too short: {len(data)} of {HEADER_SIZE} bytes")
    magic, version, fmt_id, flags, _res, w, h, num, den, count = _HEADER.unpack(
        data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise BadMyvFile(f"bad magic {magic!r}; this is not a .myv file")
    if version != VERSION:
        raise BadMyvFile(
            f"file format version {version}, but this codec only understands {VERSION}"
        )
    try:
        pix_fmt = by_id(fmt_id)
    except ValueError as exc:
        raise BadMyvFile(str(exc)) from None
    return Header(
        width=w, height=h, fps_num=num, fps_den=den, frame_count=count,
        pix_fmt=pix_fmt, entropy=bool(flags & _FLAG_ENTROPY),
    )


class MyvWriter:
    """Streams frame records out, then back-patches the frame count on close.

    Frames are written as they arrive rather than buffered, so encoding a long video needs
    memory for two frames, not for the whole file. That does mean the output stream has to
    be seekable.
    """

    def __init__(self, stream: BinaryIO, header: Header) -> None:
        self._stream = stream
        self._start = stream.tell()
        self.header = header
        self.frames_written = 0
        self.bytes_written = 0
        self._closed = False
        self._write(pack_header(header))

    def _write(self, data: bytes) -> None:
        self._stream.write(data)
        self.bytes_written += len(data)

    def frame(self, frame_type: FrameType, payload: bytes) -> None:
        self._write(_FRAME_HEADER.pack(int(frame_type), len(payload)))
        self._write(payload)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        end = self._stream.tell()
        self._stream.seek(self._start + _FRAME_COUNT_OFFSET)
        self._stream.write(struct.pack("<I", self.frames_written))
        self._stream.seek(end)

    def __enter__(self) -> "MyvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MyvReader:
    """Reads the header eagerly, then yields frame records lazily."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._start = stream.tell()
        self.header = unpack_header(stream.read(HEADER_SIZE))
        self._frames_start = stream.tell()

    def frames(self) -> Iterator[Frame]:
        self._stream.seek(self._frames_start)
        for index in range(self.header.frame_count):
            head = self._stream.read(FRAME_HEADER_SIZE)
            if len(head) < FRAME_HEADER_SIZE:
                raise BadMyvFile(
                    f"file truncated: header promised {self.header.frame_count} frames, "
                    f"stream ended at frame {index}"
                )
            type_id, length = _FRAME_HEADER.unpack(head)
            try:
                frame_type = FrameType(type_id)
            except ValueError:
                raise BadMyvFile(
                    f"frame {index} has unknown frame type {type_id}"
                ) from None
            payload = self._stream.read(length)
            if len(payload) != length:
                raise BadMyvFile(
                    f"frame {index} truncated: expected {length} bytes, "
                    f"got {len(payload)}"
                )
            yield Frame(type=frame_type, payload=payload)
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_format.py -v`
Expected: 8 PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/header.py src/format.py tests/test_format.py
git commit -m "Add the .myv container; rename fps_count to frame_count and widen the magic"
```

---

### Task 11: Encoder and decoder

**Files:**
- Implement: `src/encoder.py`
- Implement: `src/decoder.py`
- Create: `src/models/stats.py`
- Create: `tests/test_roundtrip.py`

**Interfaces:**
- Consumes: `ffmpeg.probe`, `ffmpeg.decode_frames`, `MyvWriter`, `MyvReader`, `MODES`, `entropy`.
- Produces: `encoder.encode(src, dst, *, use_entropy=True, pix_fmt=RGB24, progress=None) -> CodecStats`; `decoder.open_myv(path)` context manager yielding `(header, frames)`; `decoder.read_header(path) -> Header`; `decoder.decode_to_list(path) -> list[np.ndarray]`; `CodecStats` with `.record(...)`.

**On the delta reference.** Deltas are taken against the *previous original frame*. Because the codec is lossless the previous original and the previous reconstruction are identical, so the choice is invisible in the output — but state it explicitly, because the moment quantisation appears the encoder must switch to predicting from the reconstruction. Predicting from originals while the decoder predicts from reconstructions is the classic drift bug, and it is far easier to avoid now than to diagnose later.

**The strategy at M1 is fixed:** frame 0 is `FULL`, every other frame is `PIXEL_DELTA`. No selection. M2 replaces this with pricing every mode, and keeping the fixed strategy available is what makes M2's improvement measurable rather than merely asserted.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_ffmpeg

from src.decoder import decode_to_list, read_header
from src.encoder import encode
from src.models.frame import FrameType


@requires_ffmpeg
@pytest.mark.parametrize("use_entropy", [True, False])
def test_round_trip_is_lossless(tmp_path, write_clip, moving_frames, use_entropy):
    original = moving_frames()
    src = write_clip(original)
    dst = tmp_path / "out.myv"

    stats = encode(src, dst, use_entropy=use_entropy)
    assert stats.frames == len(original)

    decoded = decode_to_list(dst)
    assert len(decoded) == len(original)
    for want, got in zip(original, decoded):
        assert np.array_equal(want, got)


@requires_ffmpeg
def test_header_records_the_source_properties(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames(), fps=25)
    dst = tmp_path / "out.myv"
    encode(src, dst)

    header = read_header(dst)
    assert (header.width, header.height) == (32, 24)
    assert (header.fps_num, header.fps_den) == (25, 1)
    assert header.frame_count == 4


@requires_ffmpeg
def test_frame_zero_is_full_and_the_rest_are_deltas(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    stats = encode(src, dst)
    assert stats.mode_counts[FrameType.FULL] == 1
    assert stats.mode_counts[FrameType.PIXEL_DELTA] == 3


@requires_ffmpeg
def test_decoder_rejects_a_file_not_starting_with_full(tmp_path, write_clip, moving_frames):
    from src.format import HEADER_SIZE
    from src.utils.errors import BadMyvFile

    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    data = bytearray(dst.read_bytes())
    data[HEADER_SIZE] = int(FrameType.PIXEL_DELTA)
    broken = tmp_path / "broken.myv"
    broken.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile, match="nothing for it to predict from"):
        decode_to_list(broken)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_roundtrip.py -v`
Expected: FAIL — `ImportError: cannot import name 'encode' from 'src.encoder'`

- [ ] **Step 3: Create `src/models/stats.py`**

Not frozen: this one accumulates as the encoder runs. It stays in `models/` because it is still a data object — the rendering logic lives in `src/stats.py`.

```python
"""Size accounting for one encode run. Mutable, unlike every other model here."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.frame import FrameType


class CodecStats(BaseModel):
    # Deliberately NOT frozen: this one accumulates as the encoder runs.

    width: int
    height: int
    fps: float
    source_bytes: int
    entropy_enabled: bool
    #: Header plus per-frame record overhead, so the report matches the file on disk.
    container_bytes: int = 0

    frames: int = 0
    #: Payload sizes before the entropy stage.
    raw_bytes: int = 0
    #: Payload sizes as stored.
    stored_bytes: int = 0
    changed_pixels: int = 0
    mode_counts: dict[FrameType, int] = Field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.stored_bytes + self.container_bytes

    @property
    def raw_frame_bytes(self) -> int:
        """What the frames would cost uncompressed — the baseline worth beating."""
        return self.width * self.height * 3 * self.frames

    def record(
        self,
        frame_type: FrameType,
        raw: int,
        stored: int,
        changed: int | None = None,
    ) -> None:
        self.frames += 1
        self.raw_bytes += raw
        self.stored_bytes += stored
        if changed is not None:
            self.changed_pixels += changed
        self.mode_counts[frame_type] = self.mode_counts.get(frame_type, 0) + 1
```

- [ ] **Step 4: Implement `src/encoder.py`**

```python
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
from src.models.pixel_format import RGB24, PixelFormat
from src.models.stats import CodecStats
from src.modes import MODES, pixel_delta
from src.utils import entropy, ffmpeg


def encode(
    src: str | Path,
    dst: str | Path,
    *,
    use_entropy: bool = True,
    pix_fmt: PixelFormat = RGB24,
    progress=None,
) -> CodecStats:
    """Encode a video file into a .myv, returning the measurements taken along the way.

    The strategy here is fixed: frame 0 is FULL, every later frame is PIXEL_DELTA. M2
    replaces that with pricing every mode and keeping the cheapest; keeping this path
    available is what makes M2's improvement measurable rather than merely asserted.
    """
    src, dst = Path(src), Path(dst)
    info = ffmpeg.probe(src)

    header = Header(
        width=info.width,
        height=info.height,
        fps_num=info.fps_num,
        fps_den=info.fps_den,
        frame_count=0,  # back-patched on close
        pix_fmt=pix_fmt,
        entropy=use_entropy,
    )

    stats = CodecStats(
        width=info.width,
        height=info.height,
        fps=info.fps,
        source_bytes=src.stat().st_size,
        entropy_enabled=use_entropy,
        container_bytes=HEADER_SIZE,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as out, ffmpeg.decode_frames(src, pix_fmt, info) as frames:
        with MyvWriter(out, header) as writer:
            prev: np.ndarray | None = None
            for index, frame in enumerate(frames):
                frame_type = (
                    FrameType.FULL if prev is None else FrameType.PIXEL_DELTA
                )
                payload = MODES[frame_type].encode(frame, prev)
                stored = entropy.compress(payload, enabled=use_entropy)
                writer.frame(frame_type, stored)

                changed = None
                if frame_type is FrameType.PIXEL_DELTA:
                    changed = pixel_delta.count(payload)
                stats.record(frame_type, len(payload), len(stored), changed)
                stats.container_bytes += FRAME_HEADER_SIZE

                if progress is not None:
                    progress(index, info.num_frames)

                prev = frame

    return stats
```

- [ ] **Step 5: Implement `src/decoder.py`**

```python
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
from src.utils.errors import BadMyvFile


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
        payload = entropy.decompress(record.payload, enabled=header.entropy)
        try:
            frame = MODES[record.type].decode(payload, prev, shape)
        except ValueError as exc:
            raise BadMyvFile(f"frame {index} ({record.type.name}): {exc}") from None
        prev = frame
        yield frame
```

- [ ] **Step 6: Run it to verify it passes**

Run: `uv run pytest tests/test_roundtrip.py -v`
Expected: 5 PASS (the losslessness test runs twice, with and without entropy)

- [ ] **Step 7: Commit**

```bash
git add src/encoder.py src/decoder.py src/models/stats.py tests/test_roundtrip.py
git commit -m "Encode MP4 to .myv and back, losslessly, with a fixed FULL+delta strategy"
```

---

### Task 12: Prove the round-trip is exact

**Files:**
- Create: `src/models/verify.py`
- Implement: `src/verify.py`
- Create: `tests/test_verify.py`

**Interfaces:**
- Consumes: `open_myv`, `ffmpeg.decode_frames`, `PixelFormat`.
- Produces: `Mismatch` with `.describe() -> str`, `VerifyResult` with `.ok`, `verify(source, myv, pix_fmt=None) -> VerifyResult`.

This compares **numpy arrays in memory, never exported files**. Diffing two MP4s would fail on re-encode noise even when the codec is perfect, so it would prove nothing either way. The question "is my codec lossless" is about the decoded frames, and that is where it has to be asked.

It stops at the first mismatch: one is enough to disprove losslessness, and reporting *where* it happened is more useful than counting how many followed.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest
from conftest import requires_ffmpeg

from src.encoder import encode
from src.verify import verify


@requires_ffmpeg
def test_verify_passes_on_a_good_file(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    result = verify(src, dst)
    assert result.ok
    assert result.frames_checked == 4
    assert result.mismatch is None


@requires_ffmpeg
def test_verify_fails_on_a_corrupted_payload(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst, use_entropy=False)

    data = bytearray(dst.read_bytes())
    data[-1] ^= 0xFF  # flip the last payload byte
    corrupted = tmp_path / "bad.myv"
    corrupted.write_bytes(bytes(data))

    result = verify(src, corrupted)
    assert not result.ok
    assert result.mismatch is not None
    assert "pixels differ" in result.mismatch.describe()


@requires_ffmpeg
def test_verify_reports_a_frame_count_difference(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    longer = write_clip(moving_frames(count=6), name="longer.mp4")
    dst = tmp_path / "out.myv"
    encode(src, dst)

    result = verify(longer, dst)
    assert not result.ok
    assert result.source_frames != result.decoded_frames
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_verify.py -v`
Expected: FAIL — `ImportError: cannot import name 'verify' from 'src.verify'`

- [ ] **Step 3: Create `src/models/verify.py`**

```python
"""Results of a losslessness check."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Mismatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame: int
    differing_pixels: int
    first_at: tuple[int, int] | None
    expected: tuple[int, ...] | None
    got: tuple[int, ...] | None
    max_abs_error: int

    def describe(self) -> str:
        if self.first_at is None:
            return f"frame {self.frame}: frame counts differ"
        y, x = self.first_at
        return (
            f"frame {self.frame}: {self.differing_pixels:,} pixels differ "
            f"(max channel error {self.max_abs_error}); "
            f"first at (x={x}, y={y}) expected {self.expected}, got {self.got}"
        )


class VerifyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    frames_checked: int
    source_frames: int
    decoded_frames: int
    mismatch: Mismatch | None = None

    @property
    def ok(self) -> bool:
        return self.mismatch is None and self.source_frames == self.decoded_frames
```

- [ ] **Step 4: Implement `src/verify.py`**

```python
"""Proving the round-trip is exact.

This compares numpy arrays in memory, never exported files. Diffing two MP4s would fail on
re-encode noise even when the codec is perfect, so it would prove nothing either way — the
question "is my codec lossless" is about the decoded frames, and that's where it gets asked.
"""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path

import numpy as np

from src.decoder import open_myv
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
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_verify.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/verify.py src/verify.py tests/test_verify.py
git commit -m "Add verify: prove losslessness by comparing frames in memory, not files"
```

---

### Task 13: The CLI and the size report

**Files:**
- Create: `src/utils/sizes.py`
- Implement: `src/stats.py`
- Implement: `src/cli.py`
- Create: `tests/utils/test_sizes.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `encoder.encode`, `decoder.open_myv`, `decoder.read_header`, `verify.verify`, `CodecStats`, `ffmpeg.encode_frames`.
- Produces: `human_bytes(n) -> str`, `stats.render(stats) -> str`, `cli.main(argv=None) -> int`, and the `vidc` console script.

The report always shows **both** the pre-entropy and final sizes, so you can tell what the codec's representation bought from what generic compression bought. Those are different questions, and conflating them hides which one is doing the work.

- [ ] **Step 1: Write the failing tests**

`tests/utils/test_sizes.py`:

```python
from __future__ import annotations

from src.utils.sizes import human_bytes


def test_bytes_print_whole():
    assert human_bytes(7) == "7 B"
    assert human_bytes(0) == "0 B"


def test_larger_units_get_two_decimals():
    assert human_bytes(1536) == "1.50 KB"
    assert human_bytes(1024 * 1024) == "1.00 MB"


def test_spans_ten_orders_of_magnitude():
    assert human_bytes(2 * 1024**3) == "2.00 GB"
```

`tests/test_cli.py`:

```python
from __future__ import annotations

import pytest
from conftest import requires_ffmpeg

from src.cli import main


@requires_ffmpeg
def test_encode_then_info_then_verify(tmp_path, write_clip, moving_frames, capsys):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"

    assert main(["encode", str(src), str(dst)]) == 0
    assert dst.exists()
    out = capsys.readouterr().out
    assert "FULL" in out and "PIXEL_DELTA" in out

    assert main(["info", str(dst)]) == 0
    assert "32x24" in capsys.readouterr().out

    assert main(["verify", str(src), str(dst)]) == 0
    assert "lossless" in capsys.readouterr().out.lower()


@requires_ffmpeg
def test_decode_writes_a_playable_file(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    myv = tmp_path / "out.myv"
    out = tmp_path / "back.mp4"

    main(["encode", str(src), str(myv)])
    assert main(["decode", str(myv), str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/utils/test_sizes.py -v`
Expected: FAIL — `ImportError: cannot import name 'human_bytes'`

- [ ] **Step 3: Create `src/utils/sizes.py`**

```python
"""Human-readable byte counts.

A report can hold a 4-byte empty delta frame and a 2 GB raw baseline on adjacent lines. Ten
orders of magnitude in one table needs consistent rendering.
"""

from __future__ import annotations

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_bytes(count: float) -> str:
    """``human_bytes(1536)`` -> ``'1.50 KB'``. Whole bytes print without decimals."""
    value = float(count)
    for unit in _UNITS:
        if abs(value) < 1024 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Implement `src/stats.py`**

```python
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
        add("  modes:")
        for frame_type, count in sorted(stats.mode_counts.items()):
            add(f"    {frame_type.name:<14} {count:>5}")
        # Average over *predicted* frames only: frame 0 is FULL and has no reference, so
        # counting it would understate the figure.
        predicted = max(stats.frames - 1, 1)
        add(f"  changed pixels/frame {stats.changed_pixels // predicted:>10,}")

    return "\n".join(lines)
```

- [ ] **Step 5: Implement `src/cli.py`**

```python
"""The `vidc` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import stats as stats_report
from src.decoder import open_myv, read_header
from src.encoder import encode as encode_file
from src.utils import ffmpeg
from src.utils.errors import BadMyvFile, FFmpegError
from src.utils.sizes import human_bytes
from src.verify import verify as verify_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidc", description="An experimental lossless video codec."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="MP4 -> .myv, printing a size report")
    enc.add_argument("src", type=Path)
    enc.add_argument("dst", type=Path)
    enc.add_argument(
        "--no-entropy", action="store_true", help="skip the final zstd stage"
    )

    dec = sub.add_parser("decode", help=".myv -> MP4, for sharing or archiving")
    dec.add_argument("src", type=Path)
    dec.add_argument("dst", type=Path)
    dec.add_argument(
        "--lossless", action="store_true", help="export with libx264rgb -qp 0"
    )

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
            result = encode_file(args.src, args.dst, use_entropy=not args.no_entropy)
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v && uv run ruff check .`
Expected: all PASS, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/utils/sizes.py src/stats.py src/cli.py tests/utils/test_sizes.py tests/test_cli.py
git commit -m "Add the vidc CLI and a size report that separates representation from entropy"
```

---

### Task 14: Record the M1 result

**Files:**
- Create: `docs/m1-results.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the working CLI.
- Produces: a measured baseline every later milestone is compared against.

A milestone is not finished until its numbers are written down. M2 through M10 each justify themselves against these figures, so they have to exist before the next piece of work starts.

- [ ] **Step 1: Encode a real clip and capture the numbers**

```bash
uv run vidc encode <some.mp4> /tmp/out.myv
uv run vidc verify <some.mp4> /tmp/out.myv
uv run vidc encode <some.mp4> /tmp/out-raw.myv --no-entropy
```

- [ ] **Step 2: Write `docs/m1-results.md`**

Record, in a table: clip name, dimensions, frame count, raw frame bytes, payload bytes before entropy, final on-disk size, ratio vs raw, mode counts, and changed pixels per frame. State explicitly that every configuration verified lossless.

Then write the observations that motivate M2. At minimum these two, which the reference measured and this build should confirm:

1. **`PIXEL_DELTA` loses badly at a scene cut.** When nearly every pixel changes, describing each one costs 7 bytes to say what 3 bytes could say outright — so the payload can exceed the raw frame it describes. This is the whole argument for M2.
2. **Four of seven bytes are coordinates.** Quantify it: `changed_pixels * 4 / payload_bytes`. That fraction is what M3's `BLOCK_DELTA` exists to eliminate.

- [ ] **Step 3: Update `README.md`**

Replace the `### Structure` section with a current one: add a **Usage** section showing the four `vidc` commands, a line stating reconstruction is lossless and verified, and a link to `docs/ROADMAP.md` and `docs/m1-results.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/m1-results.md README.md
git commit -m "Record the M1 baseline: lossless round-trip, and what it costs"
```

---

## Definition of done

- [ ] `uv run pytest` is green, with no skips other than ffmpeg-absent ones.
- [ ] `uv run ruff check .` is clean.
- [ ] `vidc encode` → `vidc verify` reports lossless on at least one real MP4.
- [ ] `vidc decode` produces a file a normal player opens.
- [ ] `docs/m1-results.md` records the baseline, and names the weakness M2 will fix.
- [ ] Every task committed separately, each message naming an observable effect.

## What is deliberately not here

YAGNI, and each has a milestone of its own:

- **No mode selection.** The strategy is fixed; M2 adds pricing.
- **No `ModeOptions`.** Nothing has a knob yet. It arrives with `block_size` at M3.
- **No `src/selector.py` implementation.** It stays a docstring stub until M2.
- **No intra prediction.** `FULL` writes raw pixels; M8 makes it smart, and because intra prediction changes payload *content* but never its length, `full.cost` survives unchanged.
- **No keyframes or index.** M7. The `version` byte and the flags byte are what let the container grow into them without breaking M1 files.
- **No `av` dependency.** It arrives at M10 with H.264 motion-vector seeding.

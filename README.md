## Simple attempt at a video compression algorithm

An experimental lossless video codec built from scratch, to learn how video compression
actually works.

The guiding idea: **no single representation of change is always optimal.** Sometimes the
cheapest way to describe frame N is a handful of pixel deltas; sometimes it's "this block
moved 20 pixels right"; sometimes it's storing the whole frame again. A codec is best
understood as a collection of competing representations, with an encoder that prices them
and keeps the smallest.

Reconstruction is **lossless** at every milestone: every decoded frame is byte-identical to
the ffmpeg-decoded source frame. That is the invariant — a change that improves size but
breaks it has failed.

### Current status — M1

Two competing representations, a fixed strategy, and a proof that the round-trip is exact.

| mode | describes a frame as | wins when |
| --- | --- | --- |
| `FULL` | the raw pixels | almost everything changed, or nothing predicts well |
| `PIXEL_DELTA` | `(x, y, ΔR, ΔG, ΔB)` per changed pixel | changes are sparse and scattered |

On the five benchmark clips this ranges from **3206x smaller than raw** (`screen`, a static
UI) to **0.73x — larger than raw** (`noise`, where nearly every pixel changes every frame).
That spread is not a defect to hide; it is the measurement that motivates M2. The numbers,
and what they argue for, are in [docs/m1-results.md](docs/m1-results.md).

Where this is going next is [docs/ROADMAP.md](docs/ROADMAP.md) — eleven milestones, each a
response to a weakness the previous one's numbers exposed.

### Setup

Needs `ffmpeg` and `ffprobe` on `PATH`.

```bash
git clone https://github.com/CrestyDY/vid-compression.git
cd vid-compression
```

```bash
pip install uv
uv sync          # creates the venv and installs everything, including dev deps
uv run pytest    # run the tests
```

`uv sync` reads `uv.lock`, so everyone gets identical versions. Add a dependency with
`uv add <pkg>`, or a dev-only one with `uv add --dev <pkg>`.

### Use

```bash
uv run vidc encode input.mp4 video.myv    # MP4 -> .myv, prints a size report
uv run vidc play   video.myv              # watch it directly, no export step
uv run vidc decode video.myv output.mp4   # .myv -> MP4, for sharing or archiving
uv run vidc info   video.myv              # header and per-mode breakdown
uv run vidc verify input.mp4 video.myv    # prove the round-trip is pixel-exact
```

`play` renders the decoder's own frames as they are reconstructed, so what you see is the
codec's actual output rather than an H.264 re-encode of it — which is what you want when a
frame looks wrong. It takes `--start`, `--count`, `--speed`, `--width` and `--loop`.

| flag | effect |
| --- | --- |
| `encode --no-entropy` | skip the final zstd stage, to see what the representation alone costs |
| `decode --lossless` | export with `libx264rgb -qp 0` instead of a watchable re-encode |
| `play --speed 4 --width 480` | play at 4x in a 480px-wide window |

Needs `ffplay` for `play` — it ships with ffmpeg.

The size report always shows both the pre-entropy and the final size, because what the
codec's *representation* bought and what generic compression bought are different questions
— and on some clips the answer is that the entropy stage is doing nearly all the work.

Audio is dropped; this is a video codec experiment.

### Structure

```
src/
  cli.py            the `vidc` command
  encoder.py        mp4 -> .myv
  decoder.py        .myv -> frames
  format.py         container header + frame records, and their byte packing
  verify.py         the losslessness check
  stats.py          renders the size report
  player.py         direct playback, by piping decoded frames to ffplay
  selector.py       (M2) prices every candidate mode per frame
  models/           pydantic value objects: header, frame, pixel format, stats, verify
  modes/            one module per encoding mode, each a pure encode/decode/cost triple
  utils/            plumbing that knows nothing about video compression
    ffmpeg.py       MP4 <-> numpy frames, over subprocess pipes
    entropy.py      the zstd stage
    sizes.py        human-readable byte counts
    errors.py       shared exceptions
```

The dividing line that matters: **`src/` holds the codec's ideas, `src/utils/` holds the
plumbing.** Nothing under `utils/` knows anything about video compression, so changing how
the codec represents change never requires touching it.

`src/models/` holds every pydantic value object, separate from the logic that uses them —
so `Header` is a model and the struct that serialises it lives in `format.py`.

`tests/` mirrors `src/`.

### Measuring

```bash
uv run python bench/make_clips.py    # generate the five benchmark clips
```

The clips are not committed because they do not need to be — the generators are seeded and
the encode settings fixed, so this reproduces them byte-identically. Every figure in
`docs/` comes from them, so results stay comparable across milestones.

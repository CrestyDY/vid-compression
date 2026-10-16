# Documentation



- **[ROADMAP.md](ROADMAP.md)** — the eleven milestones, the measured target for each, and

  where the project goes after them.

- **[m1-results.md](m1-results.md)** — the V2 lossless codec. `FULL` + `PIXEL_DELTA`, a fixed

  strategy, and the baseline every later milestone is compared against.

- **[m12-results.md](m12-results.md)** — the source-native pixel format. 1.36x on real footage, and most of the FFV1 gap closed.
- **[next-lever.md](next-lever.md)** — the three candidate levers, each priced before any was built. Two came back too small to bother with; the third became M12.
- **[entropy-results.md](entropy-results.md)** — a context-modelled arithmetic coder. Mostly a negative result, and what came out of it instead.
- **[m8-results.md](m8-results.md)** — intra prediction. 1.53x on `cut`, and byte-exact agreement with the reference on all five `FULL` frames.
- **[m5-results.md](m5-results.md)** — `TILED`. Per-region mode choice: 1.16x on `mixed`, and `screen` overtakes the reference.
- **[m4-results.md](m4-results.md)** — `MOTION`. 4.27x on translation, and the measured proof that one block size cannot serve two modes.

- **[m3-results.md](m3-results.md)** — `BLOCK_DELTA`. Four of five clips improve, and the pre-entropy ranking proxy breaks exactly as M2 predicted.

- **[m2-results.md](m2-results.md)** — mode selection. 2.81x on scene cuts, and the fix for

  M1's file-larger-than-raw result.



Every results document is measured on the same five benchmark clips, so the numbers are

directly comparable. Regenerate the clips with:



```

uv run python bench/make_clips.py

```



The generators are seeded and the encode settings fixed, so this reproduces them

byte-identically.



## The story so far



Final on-disk size at each milestone's defaults. 320×240, 60 frames, raw RGB24 = 13.18 MB,

lossless throughout.



| clip | what it is | M1 | M3 | M4 | M5 | M8 | vs raw |

| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

| screen | static UI, sparse edits | 4.21 KB | 2.28 KB | 2.27 KB | 2.12 KB | **2.21 KB** | 5936x |

| cut | three scenes, two hard cuts | 973.75 KB | 347.01 KB | 347.01 KB | 347.01 KB | **226.20 KB** | 59.68x |

| slide | textured box translating | 2.76 MB | 1.56 MB | 373.54 KB | 373.54 KB | **373.59 KB** | 36.14x |

| noise | moving gradient + grain | 18.06 MB | 7.44 MB | 7.40 MB | 7.39 MB | **6.89 MB** | 1.91x |

| mixed | four quadrants, four behaviours | 760.69 KB | 400.90 KB | 288.47 KB | 247.82 KB | **200.99 KB** | 67.17x |



**M12 is absent from this table on purpose.** The five clips are `gbrp`, so the codec
already read their own samples and the source-native pixel format left them byte-identical.
Its 1.55x is on real footage, which these clips are not. That is also the limitation of the
table: five synthetic clips isolate five behaviours cleanly and between them failed to
expose the single largest deficiency in the codec for eight milestones.

Real footage, `samples/real12.mp4` (1920x1080, 12 frames, `yuv420p`), lossless:

| build | size | encode time | FFV1's lead | x265 lossless's lead |
| --- | ---: | ---: | ---: | ---: |
| M11 (`rgb24`) | 9,497,624 | 322.0s | 1.77x | 2.51x |
| **M12 (source-native)** | **6,114,014** | **143.3s** | **1.14x** | 1.62x |

Each milestone helps the clips whose weakness it addresses and leaves the others alone —

the mode registry working as designed. Nothing here is a general-purpose improvement; each

is a specific answer to a specific failure the previous milestone's numbers exposed.



> Results documents are **historical records**. Their figures were measured when the

> milestone landed and are not re-measured afterwards, so an older document may quote a size

> that newer code no longer produces. The table above is the current one.


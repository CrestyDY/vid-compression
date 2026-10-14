from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_ffmpeg
from src.encoder import encode
from src.models.header import Header
from src.player import build_command, select_frames


def a_header(**kw) -> Header:
    base = dict(width=320, height=240, fps_num=25, fps_den=1, frame_count=60)
    return Header(**{**base, **kw})


# ------------------------------------------------------------ command building


def test_command_describes_the_raw_stream():
    cmd = build_command(a_header(), ffplay="ffplay", title="t", width=None, speed=1.0)
    assert "-f" in cmd and "rawvideo" in cmd
    assert cmd[cmd.index("-pixel_format") + 1] == "rgb24"
    assert cmd[cmd.index("-video_size") + 1] == "320x240"
    assert cmd[-2:] == ["-i", "-"]


def test_command_closes_the_window_when_the_stream_ends():
    """Without -autoexit, `vidc play` hangs after the last frame waiting for a keypress."""
    assert "-autoexit" in build_command(
        a_header(), ffplay="ffplay", title="t", width=None, speed=1.0
    )


def test_speed_scales_the_frame_rate():
    cmd = build_command(a_header(), ffplay="ffplay", title="t", width=None, speed=2.0)
    assert float(cmd[cmd.index("-framerate") + 1]) == pytest.approx(50.0)


def test_width_scales_the_window_preserving_aspect():
    cmd = build_command(a_header(), ffplay="ffplay", title="t", width=640, speed=1.0)
    assert cmd[cmd.index("-x") + 1] == "640"
    assert cmd[cmd.index("-y") + 1] == "480"


def test_width_never_scales_height_to_zero():
    header = a_header(width=1000, height=3)
    cmd = build_command(header, ffplay="ffplay", title="t", width=10, speed=1.0)
    assert int(cmd[cmd.index("-y") + 1]) >= 1


# ------------------------------------------------------------ frame selection


def _frames(n: int):
    return (np.full((2, 2, 3), i, np.uint8) for i in range(n))


def test_select_frames_defaults_to_everything():
    assert len(list(select_frames(_frames(5), start=0, count=0))) == 5


def test_select_frames_skips_to_start():
    got = list(select_frames(_frames(5), start=2, count=0))
    assert [int(f[0, 0, 0]) for f in got] == [2, 3, 4]


def test_select_frames_honours_count():
    got = list(select_frames(_frames(10), start=1, count=3))
    assert [int(f[0, 0, 0]) for f in got] == [1, 2, 3]


def test_select_frames_stops_short_without_error():
    """Asking past the end yields what exists rather than raising."""
    assert len(list(select_frames(_frames(2), start=0, count=99))) == 2


def test_select_frames_past_the_end_is_empty():
    assert list(select_frames(_frames(3), start=99, count=0)) == []


def test_select_frames_rejects_a_negative_start():
    with pytest.raises(ValueError, match="start"):
        list(select_frames(_frames(3), start=-1, count=0))


# ------------------------------------------------------------ the CLI wiring


@requires_ffmpeg
def test_cli_play_pipes_decoded_frames_to_the_renderer(
    tmp_path, write_clip, moving_frames, monkeypatch
):
    """Verify the real decoded bytes reach the renderer, without opening a window."""
    from src import player as player_mod
    from src.cli import main

    src = write_clip(moving_frames())
    myv = tmp_path / "out.myv"
    encode(src, myv)

    written: list[bytes] = []

    class FakeRenderer:
        def __init__(self, cmd):
            self.cmd = cmd

        def write(self, data: bytes) -> None:
            written.append(data)

        def close(self) -> int:
            return 0

    monkeypatch.setattr(player_mod, "_open_renderer", lambda cmd: FakeRenderer(cmd))
    monkeypatch.setattr(player_mod, "_require_ffplay", lambda: "ffplay")

    assert main(["play", str(myv)]) == 0

    # Four frames of 32x24 RGB24.
    assert len(written) == 4
    assert all(len(chunk) == 32 * 24 * 3 for chunk in written)
    # And the pixels are the real reconstruction, not zeros.
    original = moving_frames()
    assert written[0] == np.ascontiguousarray(original[0]).tobytes()


@requires_ffmpeg
def test_cli_play_count_limits_what_is_sent(
    tmp_path, write_clip, moving_frames, monkeypatch
):
    from src import player as player_mod
    from src.cli import main

    src = write_clip(moving_frames())
    myv = tmp_path / "out.myv"
    encode(src, myv)

    written: list[bytes] = []

    class FakeRenderer:
        def __init__(self, cmd):
            self.cmd = cmd

        def write(self, data: bytes) -> None:
            written.append(data)

        def close(self) -> int:
            return 0

    monkeypatch.setattr(player_mod, "_open_renderer", lambda cmd: FakeRenderer(cmd))
    monkeypatch.setattr(player_mod, "_require_ffplay", lambda: "ffplay")

    assert main(["play", str(myv), "--start", "1", "--count", "2"]) == 0
    assert len(written) == 2


def test_cli_play_on_a_missing_file_exits_2(tmp_path, capsys):
    from src.cli import main

    assert main(["play", str(tmp_path / "nope.myv")]) == 2
    assert "error:" in capsys.readouterr().err.lower()

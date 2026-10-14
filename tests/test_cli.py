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

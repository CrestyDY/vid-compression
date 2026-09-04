"""ffmpeg wrapper."""

from pathlib import Path
import subprocess
import json

from src.models.video import VideoInfo

def probe(path: str | Path) -> VideoInfo:
    """Probe video file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path)
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode()}")

    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")

    stream = streams[0]
    
    frame_rate = stream.get("r_frame_rate")
    num, _, den = frame_rate.partition("/")

    fps_num, fps_den = int(num), int(den)
    if fps_den == 0:
        raise RuntimeError(f"Invalid frame rate: {frame_rate}")

    num_frames = int(stream.get("nb_frames"))
    return VideoInfo(
        width=int(stream.get("width")),
        height=int(stream.get("height")),
        fps_num=fps_num,
        fps_den=fps_den,
        num_frames=num_frames
    )


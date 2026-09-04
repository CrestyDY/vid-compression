"""PixelFormat: id, name, ffmpeg name, channel count."""

from pydantic import BaseModel, ConfigDict

class PixelFormat(BaseModel):
    """PixelFormat: id, name, ffmpeg name, channel count."""

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    ffmpeg_name: str
    channels: int
    
RGB24 = PixelFormat(id=0, name="rgb24", ffmpeg_name="rgb24", channels=3)

# Eventually add YUV


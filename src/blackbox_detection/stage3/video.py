from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")


def transcode_every_other_frame(
    src_hevc: str | Path,
    dst_mp4: str | Path,
    *,
    width: int = 512,
    height: int = 384,
    source_fps: int = 20,
    target_fps: int = 10,
    crf: int = 23,
    preset: str = "veryfast",
) -> Path:
    """Encode source frame indices 0,2,4,... as a constant-rate H.264 file."""
    require_ffmpeg()
    src_hevc, dst_mp4 = Path(src_hevc), Path(dst_mp4)
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        "select='not(mod(n,2))',"
        f"scale={width}:{height}:flags=lanczos,"
        f"setpts=N/({target_fps}*TB)"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-r", str(source_fps), "-i", str(src_hevc),
        "-vf", vf,
        "-an", "-c:v", "libx264", "-preset", preset,
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(target_fps), "-movflags", "+faststart",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)
    return dst_mp4

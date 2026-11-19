"""Screen capture utilities.

On macOS the system `screencapture` tool is used: the mss library calls the
deprecated CGDisplayCreateImage API, which blocks for ~30 s on macOS 26
before returning a frame. Elsewhere, mss is used as before. Either way we
hand back PNG bytes ready to be base64-encoded for the vision API.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import mss
import mss.tools
from PIL import Image

from quizai.logger import get_logger

log = get_logger(__name__)

# Hard cap to keep request payloads small. Claude Vision is happy at this size
# and it dramatically reduces base64 overhead vs. a 4K screenshot.
MAX_DIM = 1568


@dataclass
class Screenshot:
    png_bytes: bytes
    width: int
    height: int


def capture(monitor_index: int = 0) -> Screenshot:
    """Capture the screen.

    monitor_index = 0  -> all monitors stitched together (mss index 0)
    monitor_index = N  -> Nth monitor (mss indices 1..N)
    """
    if sys.platform == "darwin":
        raw = _capture_darwin(monitor_index)
        if raw is not None:
            return _finish(raw)

    with mss.mss() as sct:
        monitors = sct.monitors  # [0] is the "all" virtual monitor
        if not monitors:
            raise RuntimeError("No monitors detected by mss")
        if monitor_index < 0 or monitor_index >= len(monitors):
            log.warning(
                "Invalid monitor index %d; falling back to 0 (all monitors)",
                monitor_index,
            )
            monitor_index = 0
        mon = monitors[monitor_index]
        raw = sct.grab(mon)
        # mss returns BGRA; convert to a PIL Image (RGB).
        img = Image.frombytes("RGB", raw.size, raw.rgb)

    return _finish_img(img)


def _capture_darwin(monitor_index: int) -> bytes | None:
    """Grab the screen with /usr/sbin/screencapture (≈0.1 s).

    mss's CGDisplayCreateImage call blocks for ~30 s per capture on macOS 26,
    which made every answer feel like a 20 s wait. Returns None when the tool
    is unavailable or fails, so the caller can fall back to mss.
    """
    exe = shutil.which("screencapture")
    if not exe:
        return None
    fd, path = tempfile.mkstemp(prefix="quizai-capture-", suffix=".png")
    os.close(fd)
    # -x: silent, -t png: output format, -D: display number (1 = main).
    cmd = [exe, "-x", "-t", "png"]
    if monitor_index >= 1:
        cmd += ["-D", str(monitor_index)]
    cmd.append(path)
    try:
        subprocess.run(cmd, check=True, timeout=15, capture_output=True)
        with open(path, "rb") as fh:
            raw = fh.read()
        return raw or None
    except Exception:
        log.debug("screencapture failed; falling back to mss", exc_info=True)
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _finish(raw_png: bytes) -> Screenshot:
    try:
        img = Image.open(io.BytesIO(raw_png)).convert("RGB")
    except Exception:
        raise RuntimeError("Screen capture produced an unreadable image")
    return _finish_img(img)


def _finish_img(img: Image.Image) -> Screenshot:
    img = _downscale(img, MAX_DIM)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png = buf.getvalue()
    log.debug("Captured screen: %dx%d, %d bytes png", img.width, img.height, len(png))
    return Screenshot(png_bytes=png, width=img.width, height=img.height)


@dataclass
class MonitorInfo:
    index: int
    label: str
    width: int
    height: int


def list_monitors() -> list[MonitorInfo]:
    """Enumerate available monitors. Index 0 is the virtual "all monitors" stitch."""
    out: list[MonitorInfo] = []
    try:
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                w, h = int(mon.get("width", 0)), int(mon.get("height", 0))
                if i == 0:
                    label = f"All monitors ({w}×{h})"
                else:
                    label = f"Monitor {i} — {w}×{h} @ ({mon.get('left', 0)}, {mon.get('top', 0)})"
                out.append(MonitorInfo(index=i, label=label, width=w, height=h))
    except Exception as exc:
        log.warning("Could not enumerate monitors: %s", exc)
    return out


def _downscale(img: Image.Image, max_dim: int) -> Image.Image:
    """Downscale so the longest edge <= max_dim. Preserves aspect ratio."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

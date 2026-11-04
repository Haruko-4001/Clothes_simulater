"""Display: compositor, HUD, headless/window sinks."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import RGBImage
from .compositor import Compositor
from .hud import HUD


class HeadlessDisplay:
    """No GUI. Counts frames; optionally writes the first/last frame on request."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.frames_shown = 0
        self.last_image: RGBImage | None = None
        self.first_image: RGBImage | None = None

    def show(self, frame: RGBImage) -> None:
        self.frames_shown += 1
        if self.first_image is None:
            self.first_image = np.array(frame, copy=True)
        # Frames are freshly allocated each iteration; no copy is needed.
        self.last_image = frame


class WindowDisplay:
    """Optional GUI sink; importing PIL.ImageShow does not require a display."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.frames_shown = 0
        self.last_image: RGBImage | None = None

    def show(self, frame: RGBImage) -> None:  # pragma: no cover - GUI path
        from PIL import Image

        Image.fromarray(frame).show()
        self.frames_shown += 1
        self.last_image = np.array(frame, copy=True)


__all__ = ["Compositor", "HUD", "HeadlessDisplay", "WindowDisplay"]

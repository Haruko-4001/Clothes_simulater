"""Display: compositor, HUD, headless/window sinks."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..logging import log_event
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

    def close(self) -> None:
        return None


class WindowDisplay:
    """Realtime preview window (OpenCV HighGUI).

    Degrades to a no-op sink when OpenCV or a display server is unavailable,
    so ``--gui`` never turns a headless machine into a crash. ``Image.show()``
    is deliberately not used: it spawns an external viewer per call and would
    block the 30 fps loop.
    """

    WINDOW = "tryon"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.frames_shown = 0
        self.last_image: RGBImage | None = None
        self.closed = False
        self._cv2 = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # pragma: no cover - optional dependency
            log_event("E_ADAPTER_FALLBACK", f"WindowDisplay: {type(exc).__name__}: {exc}")

    def show(self, frame: RGBImage) -> None:  # pragma: no cover - GUI path
        self.frames_shown += 1
        self.last_image = frame
        cv2 = self._cv2
        if cv2 is None or self.closed:
            return
        try:
            cv2.imshow(self.WINDOW, cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.closed = True
        except Exception:
            # No display server / HighGUI backend: stop trying, keep running.
            self.closed = True

    def close(self) -> None:  # pragma: no cover - GUI path
        self.closed = True
        if self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass


__all__ = ["Compositor", "HUD", "HeadlessDisplay", "WindowDisplay"]

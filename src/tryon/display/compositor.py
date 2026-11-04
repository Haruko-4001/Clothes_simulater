"""Compositor: physics render + optional async keyframe, framing-safe clipping.

IMPL: src/tryon/display/compositor.py::Compositor
No generated content may overwrite contact-locked pixels (SPEC §2.1 / §17.3).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..keyframe.fuse import FusionBlender, blend
from ..types import Frame, FramingVerdict, KeyframeResult, RenderOutput, RGBImage, SystemStatus


class Compositor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fusion = FusionBlender(cfg)
        self.alpha = 0.0
        self._last_keyframe: KeyframeResult | None = None

    def begin_keyframe(self, kf_time_ms: float) -> None:
        self.fusion.begin(kf_time_ms)
        self.alpha = 0.0

    def advance_fusion(self, now_ms: float) -> tuple[float, bool]:
        alpha, active = self.fusion.step(now_ms)
        self.alpha = alpha
        return alpha, active

    def fusion_finished(self) -> bool:
        return not self.fusion.active and self.fusion.count >= self.fusion.n_frames

    def compose(
        self,
        frame: Frame,
        rendered: RenderOutput,
        status: SystemStatus,
        keyframe: KeyframeResult | None = None,
        protected_mask: np.ndarray | None = None,
    ) -> RGBImage:
        original = np.asarray(frame.image)
        h, w = original.shape[:2]

        if status.framing.verdict == FramingVerdict.REJECT:
            return np.array(original, copy=True)

        rimg = np.asarray(rendered.image)
        rmask = np.asarray(rendered.mask) > 0
        if rimg.shape[:2] != (h, w) or rmask.shape[:2] != (h, w):
            return np.array(original, copy=True)

        if status.framing.verdict == FramingVerdict.PARTIAL:
            allowed = np.zeros((h, w), dtype=bool)
            bbox = status.visible_bbox
            if bbox is None:
                allowed[:] = True
            else:
                x0, y0, x1, y1 = bbox
                x0, x1 = max(0, x0), min(w, x1)
                y0, y1 = max(0, y0), min(h, y1)
                if x1 > x0 and y1 > y0:
                    allowed[y0:y1, x0:x1] = True
            rmask = rmask & allowed
            base = np.array(original, copy=True)
            base[rmask] = rimg[rmask]
        else:
            base = rimg

        # Generated keyframes are only admissible for an OK framing and are
        # never allowed to cover physics-locked contact pixels.
        if (
            keyframe is not None
            and not keyframe.expired
            and status.framing.verdict == FramingVerdict.OK
            and self.alpha > 0.0
        ):
            base = blend(
                base,
                keyframe.image,
                self.alpha,
                keyframe.mask,
                protected_mask=protected_mask,
            )
            self._last_keyframe = keyframe
            return np.ascontiguousarray(base, dtype=np.uint8)

        return np.ascontiguousarray(base, dtype=np.uint8)


__all__ = ["Compositor"]

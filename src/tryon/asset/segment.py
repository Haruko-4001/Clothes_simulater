"""Real garment segmenter adapter (optional; falls back to Stub).

IMPL: src/tryon/asset/segment.py::Segmenter
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import GarmentClass, RGBImage, Segmentation


class Segmenter:
    """mask-free foreground segmentation via rembg when available.

    Missing dependency raises ImportError; ``factory.build`` then emits
    E_ADAPTER_FALLBACK and substitutes ``StubSegmenter`` (SPEC §8).
    """

    def __init__(self, cfg: Config | None = None) -> None:
        try:
            from rembg import remove  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("rembg is not installed") from exc
        self._remove = remove
        self.cfg = cfg

    def run(self, image: RGBImage) -> Segmentation:
        from PIL import Image

        rgb = np.asarray(image)
        out = self._remove(Image.fromarray(rgb))
        alpha = np.asarray(out.split()[-1])
        mask = (alpha > 127).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            raise ValueError("segmenter produced an empty mask")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return Segmentation(
            mask=mask,
            garment_class=GarmentClass.TOPS,
            bbox=bbox,
            confidence=0.85,
        )


__all__ = ["Segmenter"]

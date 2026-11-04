"""High-quality try-on generation — asynchronous, never per-frame (SPEC §8/§10.5).

IMPL: src/tryon/keyframe/tryon.py::KeyframeTryOn
"""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from ..config import Config
from ..types import KeyframeRequest, KeyframeResult, TryonError


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    border = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    return (dist > 18.0).astype(np.uint8)


class StubKeyframeTryOn:
    """Deterministic 2D composite; explicitly a stub, not a generative model."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg

    def generate(self, req: KeyframeRequest) -> KeyframeResult:
        person = np.asarray(req.person_image)
        garment = np.asarray(req.garment_image)
        h, w = person.shape[:2]
        mask = np.asarray(req.mask)
        if mask.ndim != 2 or mask.shape[:2] != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return KeyframeResult(
                req_id=req.req_id,
                image=person.copy(),
                mask=np.zeros((h, w), dtype=np.uint8),
                finished_at_ms=time.time() * 1000.0,
                expired=False,
            )
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        garment_img = Image.fromarray(garment).convert("RGB").resize((bw, bh), Image.BILINEAR)
        gmask = _foreground_mask(garment)
        gmask_img = Image.fromarray((gmask * 255).astype(np.uint8)).resize((bw, bh), Image.NEAREST)
        region = np.asarray(mask[y0:y1, x0:x1]) > 0
        alpha = (np.asarray(gmask_img) > 127) & region
        out = np.array(person, copy=True)
        garr = np.asarray(garment_img, dtype=np.uint8)
        sub = out[y0:y1, x0:x1]
        sub[alpha] = garr[alpha]
        out[y0:y1, x0:x1] = sub
        out_mask = np.zeros((h, w), dtype=np.uint8)
        out_mask[y0:y1, x0:x1] = alpha.astype(np.uint8)
        return KeyframeResult(
            req_id=req.req_id,
            image=out,
            mask=out_mask,
            finished_at_ms=time.time() * 1000.0,
            expired=False,
        )


# Default protocol implementation name used by SPEC §8 and the factory.
KeyframeTryOn = StubKeyframeTryOn


class FashnTryOn:
    """Default real backend (Apache-2.0, mask-free). Guarded integration point."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        try:
            import fashn_vton  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("fashn-vton is not installed") from exc
        raise ImportError(
            "FASHN VTON adapter is a guarded integration point (Phase 5); "
            "falling back to stub keyframe generation"
        )

    def generate(self, req: KeyframeRequest) -> KeyframeResult:  # pragma: no cover
        raise NotImplementedError("FASHN VTON adapter not implemented")


class KeyframeService:
    """Bounded async queue (max_queue=2) + worker thread. Never blocks B."""

    def __init__(
        self,
        cfg: Config,
        backend,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.max_queue = int(cfg.require("keyframe.max_queue"))
        self.delta_pose = float(cfg.require("keyframe.delta_pose"))
        self._queue: queue.Queue[KeyframeRequest | None] = queue.Queue(maxsize=self.max_queue)
        self._results: queue.Queue[KeyframeResult] = queue.Queue(maxsize=self.max_queue)
        self._pose_theta: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="keyframe-worker", daemon=True)
        self._thread.start()
        self.completed = 0
        self.dropped = 0
        self.expired = 0
        self._latencies: list[float] = []
        self._generation_time_s = 0.0
        self._lock = threading.Lock()

    # -- producer -------------------------------------------------------
    def set_pose(self, theta: np.ndarray | None) -> None:
        self._pose_theta = None if theta is None else np.asarray(theta, dtype=np.float64).reshape(-1)

    def submit(self, req: KeyframeRequest) -> bool:
        try:
            self._queue.put_nowait(req)
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                with self._lock:
                    self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(req)
            except queue.Full:
                pass
            return False

    def poll(self) -> KeyframeResult | None:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                break
        return latest

    # -- worker ---------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if req is None:
                break
            if self._pose_theta is not None:
                req_theta = np.asarray(req.pose.smpl_params.get("theta", []), dtype=np.float64).reshape(-1)
                if req_theta.size and float(np.linalg.norm(self._pose_theta - req_theta)) > self.delta_pose:
                    with self._lock:
                        self.expired += 1
                    result = KeyframeResult(
                        req_id=req.req_id,
                        image=np.array(req.person_image, copy=True),
                        mask=np.zeros(req.mask.shape[:2], dtype=np.uint8),
                        finished_at_ms=time.time() * 1000.0,
                        expired=True,
                    )
                    self._publish(result)
                    continue
            t0 = time.perf_counter()
            try:
                result = self.backend.generate(req)
            except Exception:
                with self._lock:
                    self.dropped += 1
                continue
            dt = time.perf_counter() - t0
            elapsed_ms = (time.time() * 1000.0) - req.enqueued_at_ms
            with self._lock:
                self.completed += 1
                self._latencies.append(float(elapsed_ms))
                self._generation_time_s += max(dt, 1e-9)
            self._publish(result)

    def _publish(self, result: KeyframeResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                _ = self._results.get_nowait()
            except queue.Empty:
                pass
            try:
                self._results.put_nowait(result)
            except queue.Full:
                pass

    # -- metrics --------------------------------------------------------
    def keyframe_latency_p95(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return float(np.percentile(np.asarray(self._latencies), 95.0))

    def generation_throughput_fps(self) -> float:
        with self._lock:
            if self._generation_time_s <= 0.0:
                return 0.0
            return float(self.completed / self._generation_time_s)

    def queue_size(self) -> int:
        return int(self._queue.qsize())

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


__all__ = [
    "KeyframeTryOn",
    "StubKeyframeTryOn",
    "FashnTryOn",
    "KeyframeService",
]

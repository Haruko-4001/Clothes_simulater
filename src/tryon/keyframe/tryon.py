"""High-quality try-on generation — asynchronous, never per-frame (SPEC §8/§10.5).

IMPL: src/tryon/keyframe/tryon.py::KeyframeTryOn
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import queue
import threading
import time
import urllib.request
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


def _fit_to(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize a generated image to the reference shape (nearest-safe)."""
    if image.shape[:2] == reference.shape[:2]:
        return image
    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    img = img.resize((reference.shape[1], reference.shape[0]), Image.BILINEAR)
    return np.asarray(img)


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
    """FASHN VTON v1.5 REST client (mask-free, Apache-2.0 backend).

    Enabling it requires all three of: ``mode: real|hybrid``,
    ``keyframe.backend: fashn``, and an API key from
    ``configs/local.yaml`` (git-ignored) or the ``FASHN_API_KEY`` env var.
    Without a key this raises ImportError so the factory falls back to the
    stub (E_ADAPTER_FALLBACK).

    Privacy: generate() uploads the person frame and the garment image to
    api.fashn.ai. There is no offline mode for this backend.

    ``_opener`` / ``_sleep`` are injectable for tests; no other network
    plumbing exists.
    """

    BASE_URL = "https://api.fashn.ai/v1"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        api_key = None
        if cfg is not None:
            api_key = cfg.get("keyframe.fashn_api_key")
        if not api_key:
            api_key = os.environ.get("FASHN_API_KEY")
        if not api_key:
            raise ImportError(
                "FASHN API key missing: set keyframe.fashn_api_key in "
                "configs/local.yaml (git-ignored) or export FASHN_API_KEY"
            )
        self._api_key = str(api_key)
        self._timeout_s = float(cfg.get("keyframe.fashn_timeout_s", 120.0)) if cfg else 120.0
        self._poll_s = float(cfg.get("keyframe.fashn_poll_s", 2.0)) if cfg else 2.0
        self._category = str(cfg.get("keyframe.fashn_category", "tops")) if cfg else "tops"
        self._opener = urllib.request.urlopen
        self._sleep = time.sleep

    # ------------------------------------------------------------------
    def _request(self, url: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="GET" if data is None else "POST",
        )
        with self._opener(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _png_data_url(image: np.ndarray) -> str:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8).copy()).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # ------------------------------------------------------------------
    def generate(self, req: KeyframeRequest) -> KeyframeResult:
        payload = {
            "model_image": self._png_data_url(req.person_image),
            "garment_image": self._png_data_url(req.garment_image),
            "category": self._category,
        }
        job = self._request(f"{self.BASE_URL}/run", payload)
        job_id = str(job.get("id", ""))
        if not job_id:
            raise TryonError("E_KF_TIMEOUT", "FASHN returned no job id")

        deadline = time.monotonic() + self._timeout_s
        output_url = None
        while time.monotonic() < deadline:
            self._sleep(self._poll_s)
            status = self._request(f"{self.BASE_URL}/run/{job_id}")
            state = str(status.get("status", ""))
            if state == "completed":
                outputs = status.get("output") or []
                if outputs:
                    output_url = str(outputs[0])
                break
            if state == "failed":
                raise TryonError(
                    "E_KF_TIMEOUT",
                    f"FASHN job {job_id} failed: {status.get('error', '')}",
                )
        if output_url is None:
            raise TryonError("E_KF_TIMEOUT", f"FASHN job {job_id} did not finish in time")

        with self._opener(urllib.request.Request(output_url), timeout=60) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        image = np.asarray(Image.open(io.BytesIO(arr.tobytes())).convert("RGB"))
        image = _fit_to(image, np.asarray(req.person_image))

        person = np.asarray(req.person_image)
        diff = np.abs(image.astype(np.int16) - person.astype(np.int16)).max(axis=2)
        mask = (diff > 12).astype(np.uint8)
        region = np.asarray(req.mask) > 0 if np.asarray(req.mask).shape == mask.shape else None
        if region is not None:
            mask &= region.astype(np.uint8)
        return KeyframeResult(
            req_id=req.req_id,
            image=image,
            mask=mask,
            finished_at_ms=time.time() * 1000.0,
            expired=False,
        )


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

"""Realtime mock main loop (SPEC §5 / §15 Phase 0–4).

Pipeline B is synchronous at 30fps; pipeline C runs in one background worker
and never blocks B; keyframes run in a bounded asynchronous service.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import dataclasses
import math
import multiprocessing
import queue as queue_mod
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Make ``python app/main.py`` work from a source checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tryon.asset.material_prior import FABRIC_PRESETS  # noqa: E402
from tryon.calibration.identifier import calibration_process_worker  # noqa: E402
from tryon.config import Config, load_config  # noqa: E402
from tryon.display import HeadlessDisplay, WindowDisplay  # noqa: E402
from tryon.factory import build  # noqa: E402
from tryon.interaction.state_machine import (  # noqa: E402
    AssetState,
    AssetStateMachine,
    InteractionStateMachine,
    KeyframeStateMachine,
)
from tryon.keyframe.tryon import KeyframeService  # noqa: E402
from tryon.logging import JsonlFrameLogger, dumps_json, setup_logging  # noqa: E402
from tryon.math import empirical_p50, empirical_p95, project  # noqa: E402
from tryon.physics.solver import PhysicsSolver  # noqa: E402
from tryon.types import (  # noqa: E402
    BodyState,
    CalibHistory,
    ContactEvent,
    FramingVerdict,
    Frame,
    GateVerdict,
    KeyframeRequest,
    KeyframeResult,
    MaterialParams,
    MaterialState,
    ObservationBatch,
    PhysicsState,
    SystemStatus,
    TimingBook,
    TryonError,
    overall_confidence,
)


# ---------------------------------------------------------------------------
# Offline asset input
# ---------------------------------------------------------------------------


def make_sample_garment(path: str | Path, width: int = 512, height: int = 640) -> Path:
    """Generate a deterministic sleeveless-top sample (no external assets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(img)
    cx = width // 2
    top, bottom = int(0.045 * height), int(0.985 * height)
    for y in range(top, bottom):
        t = (y - top) / max(1, bottom - top)
        half = int(0.395 * width * (1.0 - 0.03 * t))
        shade = int(12 * t)
        draw.line(
            [(cx - half, y), (cx + half, y)],
            fill=(72 + shade, 108 + shade, 150 + shade),
        )
    # Rounded hem/shoulder caps keep the silhouette close to the shell projection.
    r = int(0.06 * width)
    draw.ellipse([cx - int(0.395 * width), top - r, cx + int(0.395 * width), top + r], fill=(72, 108, 150))
    draw.ellipse(
        [cx - int(0.383 * width), bottom - r, cx + int(0.383 * width), bottom + r],
        fill=(84, 120, 162),
    )
    # Fabric texture, still inside the foreground mask.
    for k in range(14):
        y = top + int((k + 0.5) * (bottom - top) / 14)
        draw.line([(int(0.2 * width), y), (int(0.8 * width), y)], fill=(88, 122, 160), width=1)
    img.save(path)
    return path


def load_garment_image(path: str | Path, cfg: Config) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        make_sample_garment(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _bbox_rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1
    return m


def _contact_mask(
    state: PhysicsState,
    camera: Any,
    shape: tuple[int, int],
    radius_px: int = 7,
) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if not state.attached:
        return m
    ids = np.array(sorted(state.attached.keys()), dtype=np.int64)
    try:
        uv, _ = project(camera.K, camera.T_CW, state.positions[ids])
    except Exception:
        return m
    img = Image.fromarray(m)
    draw = ImageDraw.Draw(img)
    for x, y in uv:
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=1)
    return np.asarray(img, dtype=np.uint8)


def _trace_material(params: MaterialParams) -> float:
    return float(sum(v for k, v in params.variance.items() if math.isfinite(v)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MockRunner:
    def __init__(
        self,
        cfg: Config,
        *,
        garment_image: np.ndarray,
        log_path: Path,
        headless: bool = True,
        seed: int = 0,
        enable_keyframes: bool = True,
        enable_calibration: bool = True,
    ) -> None:
        self.cfg = cfg
        self.garment_image = garment_image
        self.log_path = Path(log_path)
        self.headless = headless
        self.seed = seed
        self.enable_keyframes = enable_keyframes
        self.enable_calibration = enable_calibration
        self.components = build(cfg)
        if not headless and isinstance(self.components["Display"], HeadlessDisplay):
            # --gui must actually change the sink, even in mock mode.
            self.components["Display"] = WindowDisplay(cfg)

        self.segmenter = self.components["Segmenter"]
        self.geometry = self.components["GeometryEstimator"]
        self.repairer = self.components["MeshRepairer"]
        self.drape = self.components["DrapeAligner"]
        self.prior = self.components["MaterialPrior"]
        self.asset_builder = self.components["AssetBuilder"]
        self.registry = self.components["AssetRegistry"]
        self.camera = self.components["Camera"]
        self.body_tracker = self.components["BodyTracker"]
        self.framing = self.components["FramingGate"]
        self.hand_tracker = self.components["HandTracker"]
        self.contact = self.components["ContactExtractor"]
        self.physics: PhysicsSolver = self.components["PhysicsSolver"]
        self.renderer = self.components["Renderer"]
        self.trigger = self.components["KeyframeTrigger"]
        self.compositor = self.components["Compositor"]
        self.hud = self.components["HUD"]
        self.display = self.components["Display"]
        self.collector = self.components["TrajectoryCollector"]
        self.identifier = self.components["SystemIdentifier"]
        self.calib_gate = self.components["CalibrationGate"]

        self.kf_service = KeyframeService(cfg, self.components["KeyframeTryOn"])

        self.asset = self.asset_builder.build(
            garment_image,
            timeout_s=float(cfg.require("asset.offline_timeout_s")),
        )
        # Synthetic ground truth for mock identification: observations are
        # generated by a hidden Θ*, while the asset only carries Θ₀.
        preset_name = cfg.get("mock.true_material_preset")
        if preset_name and str(preset_name) in FABRIC_PRESETS:
            true_material = MaterialParams(
                **dict(FABRIC_PRESETS[str(preset_name)]),
                state=MaterialState.AI_PRIOR,
            )
            self.world_asset = dataclasses.replace(self.asset, material=true_material)
            self.true_material_preset = str(preset_name)
        else:
            self.world_asset = self.asset
            self.true_material_preset = None
        self.body_history: deque[BodyState] = deque(maxlen=120)
        self.phys: PhysicsState = self.physics.initial_state(self.world_asset)

        self.asset_sm = AssetStateMachine(cfg)
        self.interaction_sm = InteractionStateMachine(cfg)
        self.keyframe_sm = KeyframeStateMachine(cfg)
        g1_pass = bool(self.asset.sewing_pattern.get("g1", {}).get("pass", False))
        self.asset_sm.g1(g1_pass)
        if self.asset_sm.state == AssetState.GEOMETRY_READY:
            self.asset_sm.confidence_review(self.asset.confidence.overall)

        self.calib_history = CalibHistory([], [], [], 0)
        # Pipeline C runs in a daemon PROCESS: CPU work can never contend with
        # the 30fps main loop, and a slow job is terminated at shutdown.
        self._mp_ctx = multiprocessing.get_context("spawn")
        self.calib_process: multiprocessing.Process | None = None
        self.calib_queue = None
        self.calib_theta = None
        self.calib_P = None
        self.last_calib_interaction = 0
        self.calib_submitted = 0
        self.calib_results = 0
        self.last_calib_loss: tuple[float, float] | None = None
        self.last_calib_cond: float | None = None
        self.commits = 0
        self.degrades = 0

        self.kf_active: KeyframeResult | None = None
        self.last_kf_enqueue_ms: float | None = None
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.bbox_crossed = False
        self.keyframes_completed = 0

        self.logger = JsonlFrameLogger(log_path)
        self.display_latencies: list[float] = []
        self.capture_times: list[float] = []
        self.display_times: list[float] = []
        self.timing = TimingBook()
        self.frame_statuses: list[str] = []
        self.verdict_counts: Counter[str] = Counter()
        self.material_counts: Counter[str] = Counter()
        self.last_status: SystemStatus | None = None
        self.start_wall = time.time()
        self.last_event_summary: list[dict[str, Any]] = []
        self.hud_texts: list[str] = []

    # ------------------------------------------------------------------
    def swap_garment(self, image: np.ndarray) -> dict[str, Any]:
        """Rebuild the asset from a new garment image and reset the simulation.

        This is the product's core user action ("try another garment"), so it
        lives on the runner instead of being re-implemented by each front end.
        Pipeline A runs synchronously here; callers that need to stay
        responsive should invoke it off the render thread.
        """
        asset = self.asset_builder.build(
            image, timeout_s=float(self.cfg.require("asset.offline_timeout_s"))
        )
        preset_name = self.cfg.get("mock.true_material_preset")
        if preset_name and str(preset_name) in FABRIC_PRESETS:
            world = dataclasses.replace(
                asset,
                material=MaterialParams(
                    **dict(FABRIC_PRESETS[str(preset_name)]),
                    state=MaterialState.AI_PRIOR,
                ),
            )
        else:
            world = asset
        self.asset = asset
        self.world_asset = world
        self.phys = self.physics.initial_state(world)
        self.body_history.clear()
        self.last_event_summary = []
        try:
            from tryon.render.texture import clear_cache

            clear_cache()
        except Exception:
            pass
        return {
            "asset_id": asset.asset_id,
            "version": asset.version,
            "garment_class": asset.garment_class.value,
            "mesh": {
                "n_vertices": asset.rest_mesh.n_vertices,
                "n_faces": asset.rest_mesh.n_faces,
            },
            "g1_pass": bool(asset.sewing_pattern.get("g1", {}).get("pass", False)),
            "confidence": asset.confidence.as_dict(),
            "needs_human_review": asset.needs_human_review,
        }

    # ------------------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        """JSON status for a UI: framing, material, physics, timing."""
        status = self.last_status
        if status is None:
            return {"ready": False}
        return {
            "ready": True,
            "frame_id": getattr(self, "frame_id", 0),
            "asset": {
                "id": self.asset.asset_id,
                "version": self.asset.version,
                "state": status.asset_state,
                "needs_human_review": self.asset.needs_human_review,
                "confidence": status.confidence.as_dict(),
            },
            "material_state": status.material_state.value,
            "framing": {
                "verdict": status.framing.verdict.value,
                "visible_keypoint_ratio": status.framing.visible_keypoint_ratio,
                "guide_text": status.framing.guide_text,
            },
            "physics": {
                "attached": len(self.phys.attached),
                "penetration": self.physics.last_penetration_count,
                "energy": self.phys.energy,
                "step": self.phys.step,
            },
            "timing": status.timing.as_dict(),
            "notice": status.notice,
        }

    # ------------------------------------------------------------------
    def _body_person_mask(self, body: BodyState, shape: tuple[int, int]) -> np.ndarray:
        return _bbox_rect_mask(shape, body.bbox)

    def _maybe_start_calibration(self) -> None:
        if not self.enable_calibration:
            return
        if self.calib_process is not None and self.calib_process.is_alive():
            return
        self._cleanup_calibration_process()
        if self.asset_sm.state not in (AssetState.MATERIAL_PRIOR, AssetState.CALIBRATING):
            return
        if self.collector.n_interactions <= self.last_calib_interaction:
            return
        if len(self.collector.samples) < 28:
            return
        if not self.collector.is_excited():
            return
        thumb = np.asarray(
            Image.fromarray(self.last_frame.image).resize((32, 32), Image.BILINEAR)
        )
        # Downsample the 60 Hz state stream to 30 Hz so H=0.4s is ≈12 frames
        # (SPEC §10.3) and finite-difference Jacobians stay tractable.
        recent = list(self.collector.samples)[-28:][::2]
        batch = ObservationBatch(
            contacts=[s.contact for s in recent],
            states=[s.state.copy() for s in recent],
            camera_views=[thumb.copy() for _ in recent],
            K=self.camera.K,
            T_CW=self.camera.T_CW,
            dt=1.0 / 30.0,
        )
        self.last_calib_interaction = self.collector.n_interactions
        payload = {
            "cfg": self.cfg,
            "asset": self.asset,
            "batch": batch,
            "theta": self.calib_theta,
            "P": self.calib_P,
        }
        q = self._mp_ctx.Queue(1)
        proc = self._mp_ctx.Process(
            target=calibration_process_worker,
            args=(payload, q),
            name="tryon-calibration",
            daemon=True,
        )
        proc.start()
        self.calib_process = proc
        self.calib_queue = q
        self.calib_submitted += 1

    def _cleanup_calibration_process(self) -> None:
        proc = self.calib_process
        if proc is not None:
            proc.join(timeout=0.05)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        q = self.calib_queue
        if q is not None:
            try:
                q.close()
            except Exception:
                pass
        self.calib_process = None
        self.calib_queue = None

    def _poll_calibration(self) -> None:
        proc = self.calib_process
        if proc is None:
            return
        payload = None
        try:
            payload = self.calib_queue.get_nowait()
        except queue_mod.Empty:
            if not proc.is_alive():
                self._notice_event(
                    "E_CALIB_UNIDENTIFIABLE",
                    "calibration process exited without a result",
                )
                self._cleanup_calibration_process()
            return
        self._cleanup_calibration_process()
        if payload is None:
            return
        if not payload.get("ok", False):
            self._notice_event(
                str(payload.get("code", "E_CALIB_UNIDENTIFIABLE")),
                str(payload.get("error", "calibration failed")),
            )
            return
        result = payload["result"]
        self.calib_theta = payload.get("theta")
        self.calib_P = payload.get("P")
        self.calib_results += 1
        self.last_calib_loss = (result.loss_before, result.loss_after)
        self.last_calib_cond = result.fisher_cond
        self._notice_event(
            "CALIB_RESULT",
            f"L {result.loss_before:.3g}->{result.loss_after:.3g} "
            f"cond={result.fisher_cond:.3g} predicts={result.predicts_next_frame}",
        )
        trP = float(sum(max(v, 0.0) for v in result.param_variance.values()))
        self.calib_history.append(result.loss_after, trP, result.fisher_cond)
        self.calib_history.n_interactions = self.collector.n_interactions

        kappa_max = float(self.cfg.require("calibration.kappa_max"))
        if self.asset_sm.state == AssetState.MATERIAL_PRIOR and result.fisher_cond < kappa_max:
            self.asset_sm.excitation(True, result.fisher_cond)
        if self.asset_sm.state != AssetState.CALIBRATING:
            return
        verdict = self.calib_gate.evaluate(result, self.calib_history)
        if verdict == GateVerdict.COMMIT:
            scene_tag = f"scene_{self.asset.asset_id}_{self.frame_id}"
            committed = self.registry.commit(
                self.asset,
                result.params,
                scene_tag=scene_tag,
                ttl_s=float(result.params.ttl_s),
            )
            self.asset = committed
            self.asset_sm.calibration(verdict, scene_tag=scene_tag, ttl_s=committed.material.ttl_s)
            self.commits += 1
            self._notice_event("COMMIT", f"material committed v{committed.version}")
        elif verdict == GateVerdict.DEGRADE:
            self.asset_sm.calibration(verdict)
            self.degrades += 1
            self._notice_event("DEGRADE", "calibration did not converge")
        else:
            self.asset_sm.interaction_update(
                self.collector.n_interactions,
                self.calib_gate.converged(self.calib_history),
            )

    def _notice_event(self, code: str, message: str) -> None:
        self.last_event_summary.append({"code": code, "message": message, "t_ms": time.time() * 1000.0})

    # ------------------------------------------------------------------
    def run(self, frames: int) -> dict[str, Any]:
        setup_logging()
        try:
            for _ in range(int(frames)):
                self._step_frame()
        finally:
            try:
                self._poll_calibration()
            except Exception:
                pass
            self._cleanup_calibration_process()
            self.kf_service.close()
            try:
                self.display.close()
            except Exception:
                pass
            self.logger.close()
        return self.summary()

    def _step_frame(self) -> None:
        t_frame0 = time.perf_counter()
        frame: Frame = self.camera.capture()
        self.capture_times.append(frame.pts_ms)
        self.last_frame = frame
        self.frame_id = frame.frame_id

        body = self.body_tracker.track(frame, self.body_history)
        self.body_history.append(body)
        framing = self.framing.check(frame, body)
        hands = self.hand_tracker.track(frame)
        if hasattr(self.asset.collision_body, "update_from_body"):
            self.asset.collision_body.update_from_body(body)

        contacts = self.contact.extract(hands, body, self.phys)
        for ev in contacts:
            self.interaction_sm.on_event(ev.kind)

        # Physics runs at 60 Hz: two fixed steps per 30 Hz display frame.
        substeps = 2
        dt = float(self.cfg.require("physics.dt"))
        for _ in range(substeps):
            try:
                self.phys = self.physics.step(self.phys, self.world_asset, contacts, dt)
            except TryonError as exc:
                if exc.code != "E_PHYS_DIVERGE" or exc.fallback is None:
                    raise
                self.phys = exc.fallback
                self._notice_event(exc.code, "reverted to last stable state")
                break
            for ev in contacts:
                self.collector.observe(ev, self.phys)
        self.interaction_sm.update_energy(self.phys.energy)

        # Keyframes: bounded async service, hard-gated by framing.
        self.kf_service.set_pose(body.smpl_params.get("theta"))
        now_ms = time.time() * 1000.0
        if self.last_kf_enqueue_ms is None:
            elapsed_kf = 1.0e9
        else:
            elapsed_kf = now_ms - self.last_kf_enqueue_ms
        if self.last_bbox is not None:
            prev_h = max(1, self.last_bbox[3] - self.last_bbox[1])
            cur_h = max(1, body.bbox[3] - body.bbox[1])
            if abs(cur_h - prev_h) / prev_h > 0.15:
                self.bbox_crossed = True
        from tryon.types import TriggerContext

        ctx = TriggerContext(
            pose_history=deque(self.body_history),
            framing=framing,
            garment_id=self.asset.asset_id,
            elapsed_since_kf_ms=elapsed_kf,
            bbox_crossed=self.bbox_crossed,
            state=self.trigger.state,
        )
        if self.enable_keyframes and self.trigger.should_fire(ctx):
            req = KeyframeRequest(
                req_id=f"kf_{frame.frame_id}",
                garment_asset_id=self.asset.asset_id,
                person_image=frame.image.copy(),
                garment_image=np.asarray(self.asset.appearance_mesh.source_image),
                mask=self._body_person_mask(body, frame.image.shape[:2]),
                pose=body,
                enqueued_at_ms=now_ms,
            )
            self.kf_service.submit(req)
            self.last_kf_enqueue_ms = now_ms
            self.bbox_crossed = False
        self.last_bbox = body.bbox

        result = self.kf_service.poll()
        if result is not None:
            self.keyframes_completed += 1
            if framework_ok := (framing.verdict == FramingVerdict.OK and not result.expired):
                self.kf_active = result
                self.compositor.begin_keyframe(result.finished_at_ms)
                self.trigger.notify_fusing()
                self.keyframe_sm.result_ready(False)
            else:
                self.trigger.notify_tracking()
                self.keyframe_sm.result_ready(True)
        if self.kf_active is not None:
            self.compositor.advance_fusion(now_ms)
            if self.compositor.fusion_finished():
                self.kf_active = None
                self.trigger.notify_tracking()
                self.keyframe_sm.fuse_done()

        rendered = self.renderer.render(frame, self.asset, self.phys, self.camera)
        material_state = self.asset.material.state
        notice = self._notice()
        status = SystemStatus(
            asset_state=self.asset_sm.state.value,
            material_state=material_state,
            confidence=self.asset.confidence,
            timing=self.timing,
            framing=framing,
            notice=notice,
            visible_bbox=body.bbox if framing.verdict == FramingVerdict.PARTIAL else None,
        )
        protected = _contact_mask(self.phys, self.camera, frame.image.shape[:2])
        composed = self.compositor.compose(
            frame,
            rendered,
            status,
            keyframe=self.kf_active,
            protected_mask=protected,
        )
        out = self.hud.draw(composed, status)
        self.hud_texts.append(self.hud.text)
        self.display.show(out)

        now_wall = time.time() * 1000.0
        latency = now_wall - frame.pts_ms
        self.display_latencies.append(float(latency))
        self.display_times.append(now_wall)
        self.frame_statuses.append(framing.verdict.value)
        self.verdict_counts[framing.verdict.value] += 1
        self.material_counts[material_state.value] += 1
        self.last_status = status

        self._update_timing(frame)
        self._maybe_start_calibration()
        self._poll_calibration()
        self._log_frame(frame, status, contacts=None)

    # ------------------------------------------------------------------
    def _notice(self) -> str:
        if self.asset.needs_human_review:
            return "资产需人工复核：背面为 AI 补全，画外区域不外推"
        if self.asset_sm.state == AssetState.DEGRADED:
            return "资产已降级：保持先验材质，禁止宣称物理已就绪"
        if self.asset_sm.state == AssetState.NEEDS_REVIEW:
            return "资产置信度偏低：建议补拍正面全身图"
        if self.asset_sm.state == AssetState.CALIBRATING:
            return "后台辨识中（不阻塞主循环）"
        return ""

    def _update_timing(self, frame: Frame) -> None:
        now = time.time()
        elapsed = max(now - self.start_wall, 1e-9)
        if len(self.capture_times) >= 2:
            capture_span = (self.capture_times[-1] - self.capture_times[0]) / 1000.0
            self.timing.camera_capture_fps = (
                (len(self.capture_times) - 1) / capture_span if capture_span > 0 else 0.0
            )
        self.timing.display_fps = len(self.display_times) / elapsed
        self.timing.physics_hz = self.physics.n_physics_steps / elapsed
        self.timing.keyframe_fps = self.kf_service.completed / elapsed
        self.timing.generation_throughput_fps = self.kf_service.generation_throughput_fps()
        self.timing.capture_to_display_ms_p50 = empirical_p50(self.display_latencies)
        self.timing.capture_to_display_ms_p95 = empirical_p95(self.display_latencies)
        self.timing.keyframe_latency_ms_p95 = self.kf_service.keyframe_latency_p95()

    def _log_frame(self, frame: Frame, status: SystemStatus, contacts=None) -> None:
        timing = status.timing.as_dict()
        lat = np.asarray(self.display_latencies, dtype=np.float64)
        timing["capture_to_display_ms_max"] = float(lat.max()) if lat.size else 0.0
        record = self.logger.make_record(
            frame_id=frame.frame_id,
            pts_ms=frame.pts_ms,
            asset_id=self.asset.asset_id,
            asset_version=self.asset.version,
            timing=timing,
            framing_verdict=status.framing.verdict.value,
            visible_keypoint_ratio=status.framing.visible_keypoint_ratio,
            guide_text=status.framing.guide_text,
            material_state=status.material_state.value,
            back_geometry_confidence=status.confidence.back_geometry,
            attached_count=len(self.phys.attached),
            penetration_count=self.physics.last_penetration_count,
            energy=self.phys.energy,
            events=list(self.last_event_summary),
            extra={
                "asset_state": status.asset_state,
                "keyframe_fps": status.timing.keyframe_fps,
                "keyframes_completed": self.kf_service.completed,
                "calibration": {
                    "n_interactions": self.collector.n_interactions,
                    "samples": len(self.collector.samples),
                    "excited": bool(self.collector.is_excited()),
                    "reason": self.collector.last_reason,
                    "pending": self.calib_process is not None,
                    "submitted": self.calib_submitted,
                    "commits": self.commits,
                    "degrades": self.degrades,
                },
            },
        )
        self.logger.write(record)
        self.last_event_summary = []

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.start_wall, 1e-9)
        return {
            "mode": self.cfg.get("mode"),
            "spec_version": "v2.0",
            "frames": len(self.display_times),
            "elapsed_s": elapsed,
            "asset": {
                "asset_id": self.asset.asset_id,
                "version": self.asset.version,
                "material_state": self.asset.material.state.value,
                "needs_human_review": self.asset.needs_human_review,
                "g1_pass": bool(self.asset.sewing_pattern.get("g1", {}).get("pass", False)),
                "confidence": self.asset.confidence.as_dict(),
            },
            "timing": self.timing.as_dict(),
            "framing_histogram": dict(self.verdict_counts),
            "material_histogram": dict(self.material_counts),
            "calibration_jobs_submitted": self.calib_submitted,
            "calibration_results": self.calib_results,
            "calibration_last_loss": self.last_calib_loss,
            "calibration_last_cond": self.last_calib_cond,
            "commits": self.commits,
            "degrades": self.degrades,
            "mock_ground_truth_preset": self.true_material_preset,
            "keyframes": {
                "completed": self.kf_service.completed,
                "expired": self.kf_service.expired,
                "dropped": self.kf_service.dropped,
                "queue_max": self.cfg.get("keyframe.max_queue"),
            },
            "log_path": str(self.log_path),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Any-garment realtime try-on (SPEC v2.0)")
    # None (not "mock") so an explicit --mode is distinguishable from "unset"
    # and configs/real.yaml / configs/hybrid.yaml are not silently overridden.
    parser.add_argument("--mode", default=None, choices=["mock", "real", "hybrid"])
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--frames", type=int, default=300)
    # Headless is the default (SPEC §15 ACCEPT_CMD passes --headless). The old
    # bug was that ``store_true, default=True`` made the flag impossible to turn
    # OFF; --gui is now the explicit, mutually exclusive opt-out.
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--headless", action="store_true", default=True,
                      help="no window (default)")
    view.add_argument("--gui", action="store_true", help="open a preview window")
    parser.add_argument("--garment", default=str(REPO_ROOT / "assets" / "samples" / "garment.png"))
    parser.add_argument("--log", default=str(REPO_ROOT / "var" / "logs" / "run.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-keyframes", action="store_true")
    parser.add_argument("--no-calibration", action="store_true",
                        help="disable pipeline C (pure engine benchmark; no spawn worker)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--config-dir", default=None)
    # Backend overrides. Choices mirror the documented config domains so a
    # value accepted here is always one the factory understands.
    parser.add_argument("--physics-solver", default=None,
                        choices=["stub_xpbd", "warp", "diffcloth"])
    parser.add_argument("--render-backend", default=None,
                        choices=["stub", "opencv", "3dgs", "gaussian"])
    parser.add_argument("--geometry", default=None,
                        choices=["stub", "silhouette", "pattern", "sewformer", "instantmesh", "garmentrec"])
    parser.add_argument("--keyframe-backend", default=None,
                        choices=["stub", "fashn", "fashn_vton_1_5", "catvton", "leffa", "omnitry"],
                        help="generation backend; requires --mode real|hybrid to take effect")
    parser.add_argument("--grad-backend", default=None, choices=["fd", "warp"],
                        help="pipeline C gradient source (warp needs the engine extra)")
    parser.add_argument("--self-collision", action="store_true",
                        help="enable vertex-level self-collision in the Warp solver")
    parser.add_argument("--mesh-resolution", nargs=2, type=int, default=None, metavar=("AROUND", "HEIGHT"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.mode, args.hardware, config_dir=args.config_dir)
    if args.width:
        cfg.set("camera.width", int(args.width))
    if args.height:
        cfg.set("camera.height", int(args.height))
    if args.physics_solver:
        cfg.set("physics.solver", args.physics_solver)
    if args.render_backend:
        cfg.set("render.backend", args.render_backend)
    if args.geometry:
        cfg.set("asset.geometry", args.geometry)
    if args.keyframe_backend:
        cfg.set("keyframe.backend", args.keyframe_backend)
    if args.grad_backend:
        cfg.set("calibration.grad_backend", args.grad_backend)
    if args.self_collision:
        cfg.set("physics.self_collision", True)
    if args.mesh_resolution:
        cfg.set("asset.mesh_resolution", [int(args.mesh_resolution[0]), int(args.mesh_resolution[1])])
    headless = not args.gui
    garment = load_garment_image(args.garment, cfg)
    runner = MockRunner(
        cfg,
        garment_image=garment,
        log_path=Path(args.log),
        headless=headless,
        seed=args.seed,
        enable_keyframes=not args.no_keyframes,
        enable_calibration=not args.no_calibration,
    )
    summary = runner.run(args.frames)
    print(dumps_json(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

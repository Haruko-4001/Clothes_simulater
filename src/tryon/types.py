"""Core data types and invariants for the any-garment realtime try-on system.

IMPL: src/tryon/types.py

This module is the single home of the enums / dataclasses defined in SPEC §9.
Units follow SPEC Appendix B exactly. No module may define a second copy of
these numeric domains or thresholds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields, replace
from enum import Enum
import math
from typing import Any, Iterable, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Errors (SPEC Appendix C)
# ---------------------------------------------------------------------------

ERROR_CODES = frozenset(
    {
        "E_MATH_QUAT",
        "E_MATH_DEPTH",
        "E_ASSET_GEOM_FAIL",
        "E_ASSET_TIMEOUT",
        "E_ASSET_LOW_CONF",
        "E_ADAPTER_FALLBACK",
        "E_FRAMING_REJECT",
        "E_FRAMING_PARTIAL",
        "E_PHYS_DIVERGE",
        "E_PHYS_PENETRATION",
        "E_KF_TIMEOUT",
        "E_KF_EXPIRED",
        "E_CALIB_UNIDENTIFIABLE",
        "E_CALIB_NO_EXCITATION",
        "E_CALIB_GATE_REJECT",
        "E_LICENSE_NC_BLOCKED",
    }
)


class TryonError(RuntimeError):
    """Typed error carrying a SPEC Appendix-C code.

    ``fallback`` is used by the physics layer: on E_PHYS_DIVERGE the caller
    restores the last stable state instead of propagating NaN.
    """

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        context: Mapping[str, Any] | None = None,
        fallback: Any = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.fallback = fallback


# ---------------------------------------------------------------------------
# Enumerations (SPEC §9.1)
# ---------------------------------------------------------------------------


class MaterialState(str, Enum):
    AI_PRIOR = "AI_PRIOR"
    CALIBRATING = "CALIBRATING"
    CALIBRATED = "CALIBRATED"
    DEGRADED = "DEGRADED"


class GeometryProvenance(str, Enum):
    OBSERVED = "OBSERVED"
    PRIOR_BACK = "PRIOR_BACK"
    PRIOR_PATTERN = "PRIOR_PATTERN"


class FramingVerdict(str, Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"
    GUIDE = "GUIDE"
    REJECT = "REJECT"


class ContactKind(str, Enum):
    GRASP = "GRASP"
    PULL = "PULL"
    LIFT = "LIFT"
    RELEASE = "RELEASE"
    SLIDE = "SLIDE"


class GateVerdict(str, Enum):
    KEEP_PRIOR = "KEEP_PRIOR"
    CALIBRATING = "CALIBRATING"
    COMMIT = "COMMIT"
    DEGRADE = "DEGRADE"


class GarmentClass(str, Enum):
    TOPS = "tops"
    BOTTOMS = "bottoms"
    ONE_PIECES = "one-pieces"
    ACCESSORY = "accessory"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Numeric domain (SPEC Appendix B — the sole authority)
# ---------------------------------------------------------------------------

MATERIAL_BOUNDS: dict[str, tuple[float, float]] = {
    "areal_density": (0.02, 1.20),       # kg/m^2
    "youngs_modulus": (1.0e2, 1.0e6),    # N/m
    "bending_stiffness": (1.0e-6, 1.0e-1),  # N·m
    "poisson_ratio": (0.0, 0.49),        # -
    "friction": (0.0, 1.5),              # -
    "damping": (0.0, 50.0),              # N·s/m
    "thickness": (1.0e-4, 5.0e-3),       # m
}

PARAM_NAMES: tuple[str, ...] = (
    "areal_density",
    "youngs_modulus",
    "bending_stiffness",
    "poisson_ratio",
    "friction",
    "damping",
    "thickness",
)

GRAVITY = np.array([0.0, -9.81, 0.0], dtype=np.float64)
DT_DISPLAY = 1.0 / 30.0
DT_PHYSICS = 1.0 / 60.0


def _check_unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be finite in [0,1], got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Asset types (SPEC §9.2)
# ---------------------------------------------------------------------------


@dataclass
class MaterialParams:
    """Cloth material vector Θ ∈ R^7 (SPEC §2.3) with Appendix-B domains."""

    areal_density: float = 0.25       # kg/m^2
    youngs_modulus: float = 2.0e3     # N/m (stretch-stiffness proxy)
    bending_stiffness: float = 1.0e-3  # N·m
    poisson_ratio: float = 0.30       # -
    friction: float = 0.40            # -
    damping: float = 2.0              # N·s/m
    thickness: float = 1.0e-3         # m
    state: MaterialState = MaterialState.AI_PRIOR
    variance: dict[str, float] = field(default_factory=dict)
    scene_tag: str = ""               # never reuse across scenes
    calibrated_at: float | None = None
    ttl_s: float = 1800.0

    def __post_init__(self) -> None:
        for name in PARAM_NAMES:
            value = float(getattr(self, name))
            lo, hi = MATERIAL_BOUNDS[name]
            if not math.isfinite(value) or value < lo or value > hi:
                raise ValueError(
                    f"material.{name}={value!r} outside feasible domain [{lo}, {hi}]"
                )
            setattr(self, name, value)
        if not isinstance(self.state, MaterialState):
            self.state = MaterialState(self.state)
        if self.state != MaterialState.CALIBRATED and self.calibrated_at is not None:
            raise ValueError(
                "INV violated: state != CALIBRATED ⇒ calibrated_at must be None"
            )
        self.ttl_s = float(self.ttl_s)
        if self.ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")

    # -- vectorisation -------------------------------------------------
    def to_vector(self) -> np.ndarray:
        return np.array([float(getattr(self, n)) for n in PARAM_NAMES], dtype=np.float64)

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray | Iterable[float],
        *,
        template: "MaterialParams | None" = None,
    ) -> "MaterialParams":
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.size != 7:
            raise ValueError(f"Θ must have 7 components, got {v.size}")
        base = template or cls()
        kwargs = {n: float(v[i]) for i, n in enumerate(PARAM_NAMES)}
        return cls(
            **kwargs,
            state=base.state,
            variance=dict(base.variance),
            scene_tag=base.scene_tag,
            calibrated_at=base.calibrated_at,
            ttl_s=base.ttl_s,
        )

    def copy(self) -> "MaterialParams":
        return replace(self, variance=dict(self.variance))

    def is_expired(self, now_s: float) -> bool:
        if self.calibrated_at is None:
            return True
        return (now_s - self.calibrated_at) > self.ttl_s

    def validate(self) -> None:
        __class__.__post_init__(self)


def material_bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([MATERIAL_BOUNDS[n][0] for n in PARAM_NAMES], dtype=np.float64)
    hi = np.array([MATERIAL_BOUNDS[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lo, hi


@dataclass
class Confidence:
    segmentation: float
    visible_geometry: float
    back_geometry: float
    topology: float
    material: float
    overall: float

    def __post_init__(self) -> None:
        for f in fields(self):
            setattr(self, f.name, _check_unit_interval(f.name, getattr(self, f.name)))

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


def weighted_geometric_mean(
    values: Mapping[str, float] | Iterable[float],
    weights: Mapping[str, float] | Iterable[float],
    eps: float = 1.0e-6,
) -> float:
    """Weighted geometric mean used by Confidence.overall (SPEC §9.2)."""
    if isinstance(values, Mapping):
        keys = list(values.keys())
        v = np.array([values[k] for k in keys], dtype=np.float64)
        w = np.array([weights[k] for k in keys], dtype=np.float64)  # type: ignore[index]
    else:
        v = np.asarray(list(values), dtype=np.float64)
        w = np.asarray(list(weights), dtype=np.float64)  # type: ignore[arg-type]
    if v.size == 0 or v.size != w.size:
        raise ValueError("values/weights must be non-empty and equal length")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)):
        raise ValueError("values/weights must be finite")
    if np.any(w < 0) or float(w.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with positive sum")
    v = np.clip(v, eps, 1.0)
    return float(np.exp(np.sum(w * np.log(v)) / float(w.sum())))


def overall_confidence(conf: Confidence, weights: Mapping[str, float]) -> float:
    parts = {
        "segmentation": conf.segmentation,
        "visible_geometry": conf.visible_geometry,
        "back_geometry": conf.back_geometry,
        "topology": conf.topology,
        "material": conf.material,
    }
    return weighted_geometric_mean(parts, weights)


@dataclass
class SimReadyMesh:
    """Physics shell mesh (SPEC §9.4). Appearance meshes are kept separate (D13)."""

    vertices: np.ndarray
    faces: np.ndarray
    thickness: float
    is_closed: bool
    has_isolated_edge: bool
    self_intersection_count: int
    collision_ready: bool

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("SimReadyMesh.vertices must be (N,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("SimReadyMesh.faces must be (M,3)")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("SimReadyMesh.faces contains out-of-range indices")
        self.thickness = float(self.thickness)
        if not (self.thickness > 0.0):
            raise ValueError("SimReadyMesh.thickness must be > 0")
        self.self_intersection_count = int(self.self_intersection_count)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


@dataclass
class DrapeReport:
    silhouette_iou: float
    chamfer_distance: float
    normal_error: float
    flyaway_faces: int

    def __post_init__(self) -> None:
        self.silhouette_iou = float(self.silhouette_iou)
        self.chamfer_distance = float(self.chamfer_distance)
        self.normal_error = float(self.normal_error)
        self.flyaway_faces = int(self.flyaway_faces)


@dataclass
class GarmentAsset:
    asset_id: str
    version: int
    garment_class: GarmentClass
    rest_mesh: SimReadyMesh
    appearance_mesh: object
    sewing_pattern: dict
    material: MaterialParams
    collision_body: object
    confidence: Confidence
    provenance: dict[str, GeometryProvenance]
    created_at: float
    needs_human_review: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.rest_mesh, SimReadyMesh):
            raise TypeError("rest_mesh must be a SimReadyMesh (physics shell)")
        if self.appearance_mesh is self.rest_mesh:
            raise ValueError(
                "D13 violated: appearance mesh must be separate from the physics shell"
            )
        if not isinstance(self.garment_class, GarmentClass):
            self.garment_class = GarmentClass(self.garment_class)
        if int(self.version) < 0:
            raise ValueError("version must be >= 0")
        self.version = int(self.version)
        self.provenance = {
            str(k): GeometryProvenance(v) for k, v in dict(self.provenance).items()
        }


# ---------------------------------------------------------------------------
# Runtime types (SPEC §9.3)
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    frame_id: int
    pts_ms: float
    image: np.ndarray      # uint8 HxWx3, RGB

    def __post_init__(self) -> None:
        self.frame_id = int(self.frame_id)
        self.pts_ms = float(self.pts_ms)
        if not np.isfinite(self.pts_ms):
            raise ValueError("pts_ms must be finite")
        img = np.asarray(self.image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Frame.image must be (H,W,3)")
        if img.dtype != np.uint8:
            raise ValueError("Frame.image must be uint8 RGB")
        self.image = img


@dataclass
class FramingCompliance:
    verdict: FramingVerdict
    visible_keypoint_ratio: float
    bbox_clipped: bool
    facing_estimate: str
    occlusion_ratio: float
    framing_confidence: float
    guide_text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, FramingVerdict):
            self.verdict = FramingVerdict(self.verdict)
        self.visible_keypoint_ratio = _check_unit_interval(
            "visible_keypoint_ratio", self.visible_keypoint_ratio
        )
        self.occlusion_ratio = _check_unit_interval("occlusion_ratio", self.occlusion_ratio)
        self.framing_confidence = _check_unit_interval(
            "framing_confidence", self.framing_confidence
        )
        if self.facing_estimate not in {"frontal", "side", "back", "unknown"}:
            raise ValueError(f"bad facing_estimate: {self.facing_estimate!r}")
        if self.verdict in (FramingVerdict.GUIDE, FramingVerdict.PARTIAL) and not self.guide_text:
            raise ValueError(f"INV violated: verdict={self.verdict.value} requires guide_text")


@dataclass
class ContactEvent:
    t_ms: float
    kind: ContactKind
    world_pos: np.ndarray   # W frame, metres
    vertex_id: int | None
    force_proxy: float      # visual proxy ONLY; never a physics force input
    hand_id: str

    def __post_init__(self) -> None:
        self.t_ms = float(self.t_ms)
        self.world_pos = np.asarray(self.world_pos, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(self.world_pos)):
            raise ValueError("ContactEvent.world_pos must be finite")
        if not isinstance(self.kind, ContactKind):
            self.kind = ContactKind(self.kind)
        self.force_proxy = float(self.force_proxy)
        self.vertex_id = None if self.vertex_id is None else int(self.vertex_id)
        self.hand_id = str(self.hand_id)


@dataclass
class PhysicsState:
    positions: np.ndarray    # (N,3) m
    velocities: np.ndarray   # (N,3) m/s
    attached: dict[int, str]
    lam: np.ndarray          # (M_c,)
    step: int
    energy: float            # J

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        self.lam = np.asarray(self.lam, dtype=np.float64).reshape(-1)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be (N,3)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must match positions shape")
        self.attached = {int(k): str(v) for k, v in dict(self.attached).items()}
        self.step = int(self.step)
        self.energy = float(self.energy)

    def copy(self) -> "PhysicsState":
        return PhysicsState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            attached=dict(self.attached),
            lam=self.lam.copy(),
            step=self.step,
            energy=self.energy,
        )


@dataclass
class TimingBook:
    """Five independent fps meters. NEVER derive one field from another."""

    camera_capture_fps: float = 0.0
    display_fps: float = 0.0
    physics_hz: float = 0.0
    keyframe_fps: float = 0.0
    generation_throughput_fps: float = 0.0
    capture_to_display_ms_p50: float = 0.0
    capture_to_display_ms_p95: float = 0.0
    keyframe_latency_ms_p95: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


@dataclass
class KeyframeRequest:
    req_id: str
    garment_asset_id: str
    person_image: np.ndarray
    garment_image: np.ndarray
    mask: np.ndarray
    pose: "BodyState"
    enqueued_at_ms: float

    def __post_init__(self) -> None:
        self.person_image = np.asarray(self.person_image)
        self.garment_image = np.asarray(self.garment_image)
        self.mask = np.asarray(self.mask)
        self.enqueued_at_ms = float(self.enqueued_at_ms)


@dataclass
class KeyframeResult:
    req_id: str
    image: np.ndarray
    mask: np.ndarray
    finished_at_ms: float
    expired: bool = False

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        self.mask = np.asarray(self.mask)
        self.finished_at_ms = float(self.finished_at_ms)
        self.expired = bool(self.expired)


@dataclass
class IdentificationResult:
    params: MaterialParams
    loss_before: float
    loss_after: float
    param_variance: dict[str, float]
    fisher_cond: float
    predicts_next_frame: bool

    def __post_init__(self) -> None:
        self.loss_before = float(self.loss_before)
        self.loss_after = float(self.loss_after)
        self.fisher_cond = float(self.fisher_cond)
        self.predicts_next_frame = bool(self.predicts_next_frame)
        if not math.isfinite(self.loss_before) or not math.isfinite(self.loss_after):
            raise ValueError("losses must be finite")
        if self.loss_before < 0.0 or self.loss_after < 0.0:
            raise ValueError("losses must be >= 0")


# ---------------------------------------------------------------------------
# Perception / geometry / calibration support types (SPEC §9.5)
# ---------------------------------------------------------------------------

RGBImage = np.ndarray  # uint8 (H,W,3), RGB order. BGR is forbidden.


@dataclass
class Segmentation:
    mask: np.ndarray                   # uint8 (H,W), 0/1
    garment_class: GarmentClass
    bbox: tuple[int, int, int, int]    # x0,y0,x1,y1 px
    confidence: float

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask)
        if self.mask.ndim != 2:
            raise ValueError("Segmentation.mask must be (H,W)")
        if not isinstance(self.garment_class, GarmentClass):
            self.garment_class = GarmentClass(self.garment_class)
        self.bbox = tuple(int(v) for v in self.bbox)  # type: ignore[assignment]
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (x0,y0,x1,y1)")
        self.confidence = _check_unit_interval("confidence", self.confidence)


@dataclass
class BodyState:
    smpl_params: dict                  # {"beta": (10,), "theta": (|θ|,), "transl": (3,)}
    pose_keypoints: np.ndarray         # (J,3) W frame m (or (J,2)+conf)
    bbox: tuple[int, int, int, int]
    confidence: float
    fitted_from_frames: int = 1        # D12: never claim high confidence at 1

    def __post_init__(self) -> None:
        self.pose_keypoints = np.asarray(self.pose_keypoints, dtype=np.float64)
        if self.pose_keypoints.ndim != 2 or self.pose_keypoints.shape[1] not in (2, 3):
            raise ValueError("pose_keypoints must be (J,2) or (J,3)")
        self.smpl_params = dict(self.smpl_params)
        self.bbox = tuple(int(v) for v in self.bbox)  # type: ignore[assignment]
        self.confidence = _check_unit_interval("confidence", self.confidence)
        self.fitted_from_frames = int(self.fitted_from_frames)
        if self.fitted_from_frames < 1:
            raise ValueError("fitted_from_frames must be >= 1")

    def keypoints_3d(self) -> np.ndarray:
        if self.pose_keypoints.shape[1] == 3:
            return self.pose_keypoints
        z = np.zeros((self.pose_keypoints.shape[0], 1), dtype=np.float64)
        return np.concatenate([self.pose_keypoints, z], axis=1)


@dataclass
class HandState:
    landmarks: np.ndarray              # (21,3) or MANO vertices
    mesh: object | None
    handedness: str                    # left|right
    occluded: bool

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float64)
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] not in (2, 3):
            raise ValueError("landmarks must be (K,2) or (K,3)")
        if self.handedness not in {"left", "right"}:
            raise ValueError("handedness must be left|right")
        self.occluded = bool(self.occluded)

    @property
    def palm(self) -> np.ndarray:
        if self.landmarks.shape[1] == 2:
            return np.array(
                [self.landmarks[:, 0].mean(), self.landmarks[:, 1].mean(), 0.0]
            )
        return self.landmarks.mean(axis=0)


@dataclass
class GeometryDraft:
    panels: list[object]
    stitches: list[tuple[int, int]]
    rest_mesh: object
    back_appearance: object | None     # visual only (D13)
    class_prior: GarmentClass
    provenance: dict[str, GeometryProvenance]

    def __post_init__(self) -> None:
        if not isinstance(self.class_prior, GarmentClass):
            self.class_prior = GarmentClass(self.class_prior)
        self.provenance = {
            str(k): GeometryProvenance(v) for k, v in dict(self.provenance).items()
        }


@dataclass
class ObservationBatch:
    contacts: list[ContactEvent]
    states: list[PhysicsState]
    camera_views: list[RGBImage]
    # Extensions required by the §10.3 reprojection residual: §9.5 omits the
    # camera, so the batch carries it explicitly (see STRUCTURED_REPORT).
    K: np.ndarray | None = None
    T_CW: np.ndarray | None = None
    seg_masks: list[np.ndarray] = field(default_factory=list)
    dt: float = DT_PHYSICS

    def __post_init__(self) -> None:
        n = len(self.states)
        if len(self.contacts) != n or len(self.camera_views) != n:
            raise ValueError(
                "INV violated: len(contacts)==len(states)==len(camera_views)"
            )
        if self.seg_masks and len(self.seg_masks) != n:
            raise ValueError("seg_masks, when provided, must match the window length")
        if self.K is not None:
            self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if self.T_CW is not None:
            self.T_CW = np.asarray(self.T_CW, dtype=np.float64).reshape(4, 4)
        self.dt = float(self.dt)
        if not (self.dt > 0.0) or not np.isfinite(self.dt):
            raise ValueError("batch.dt must be positive and finite")

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class LossWeights:
    w_r: float = 1.0
    w_v: float = 0.5
    w_s: float = 0.3
    w_p: float = 2.0
    lam: float = 1.0e-3
    sigma_inv: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("w_r", "w_v", "w_s", "w_p", "lam"):
            setattr(self, name, float(getattr(self, name)))
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.sigma_inv is not None:
            s = np.asarray(self.sigma_inv, dtype=np.float64)
            if s.ndim != 2 or s.shape[0] != s.shape[1]:
                raise ValueError("sigma_inv must be square")
            self.sigma_inv = s


@dataclass
class Residuals:
    reproj: np.ndarray       # (T,2N)
    velocity: np.ndarray     # (T,3N)
    silhouette: np.ndarray   # (T,)
    penetration: np.ndarray  # (T,)

    def __post_init__(self) -> None:
        self.reproj = np.asarray(self.reproj, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.silhouette = np.asarray(self.silhouette, dtype=np.float64).reshape(-1)
        self.penetration = np.asarray(self.penetration, dtype=np.float64).reshape(-1)

    def flatten(self) -> np.ndarray:
        return np.concatenate(
            [self.reproj.reshape(-1), self.velocity.reshape(-1), self.silhouette, self.penetration]
        )

    @property
    def dim(self) -> int:
        return int(self.flatten().size)


@dataclass
class CalibHistory:
    losses: list[float]
    variances: list[float]   # tr(P_k)
    conds: list[float]       # cond(I)
    n_interactions: int = 0

    def __post_init__(self) -> None:
        self.losses = [float(x) for x in self.losses]
        self.variances = [float(x) for x in self.variances]
        self.conds = [float(x) for x in self.conds]
        self.n_interactions = int(self.n_interactions)
        if not (len(self.losses) == len(self.variances) == len(self.conds)):
            raise ValueError("INV violated: losses/variances/conds must have equal length")

    def append(self, loss: float, variance: float, cond: float) -> None:
        self.losses.append(float(loss))
        self.variances.append(float(variance))
        self.conds.append(float(cond))

    def is_valid(self) -> bool:
        return len(self.losses) == len(self.variances) == len(self.conds)


@dataclass
class TriggerContext:
    pose_history: deque
    framing: FramingCompliance
    garment_id: str
    elapsed_since_kf_ms: float
    bbox_crossed: bool
    state: str                # TRACKING|STANDBY|GENERATING|FUSING

    def __post_init__(self) -> None:
        if not isinstance(self.pose_history, deque):
            self.pose_history = deque(self.pose_history)
        self.elapsed_since_kf_ms = float(self.elapsed_since_kf_ms)
        self.bbox_crossed = bool(self.bbox_crossed)


@dataclass
class SystemStatus:
    asset_state: str
    material_state: MaterialState
    confidence: Confidence
    timing: TimingBook
    framing: FramingCompliance
    notice: str = ""
    # Extension for PARTIAL clipping: the only region allowed to change pixels.
    visible_bbox: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.material_state, MaterialState):
            self.material_state = MaterialState(self.material_state)
        if not isinstance(self.framing, FramingCompliance):
            raise TypeError("framing must be a FramingCompliance")
        if self.visible_bbox is not None:
            if len(tuple(self.visible_bbox)) != 4:
                raise ValueError("visible_bbox must be (x0,y0,x1,y1)")
            self.visible_bbox = tuple(int(v) for v in self.visible_bbox)  # type: ignore[assignment]


@dataclass
class RenderOutput:
    """Result of the visible-layer renderer (physics-driven, not generative)."""

    image: RGBImage
    mask: np.ndarray            # uint8 (H,W), 1 where the garment was rendered
    coverage_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        self.mask = np.asarray(self.mask)
        self.coverage_bbox = tuple(int(v) for v in self.coverage_bbox)


__all__ = [
    "ERROR_CODES",
    "TryonError",
    "MaterialState",
    "GeometryProvenance",
    "FramingVerdict",
    "ContactKind",
    "GateVerdict",
    "GarmentClass",
    "MATERIAL_BOUNDS",
    "PARAM_NAMES",
    "GRAVITY",
    "DT_DISPLAY",
    "DT_PHYSICS",
    "MaterialParams",
    "material_bounds_arrays",
    "Confidence",
    "weighted_geometric_mean",
    "overall_confidence",
    "SimReadyMesh",
    "DrapeReport",
    "GarmentAsset",
    "Frame",
    "FramingCompliance",
    "ContactEvent",
    "PhysicsState",
    "TimingBook",
    "KeyframeRequest",
    "KeyframeResult",
    "IdentificationResult",
    "RGBImage",
    "Segmentation",
    "BodyState",
    "HandState",
    "GeometryDraft",
    "ObservationBatch",
    "LossWeights",
    "Residuals",
    "CalibHistory",
    "TriggerContext",
    "SystemStatus",
    "RenderOutput",
]

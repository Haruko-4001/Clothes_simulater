"""Config loading / merging. All thresholds come from ``configs/*.yaml``.

IMPL: src/tryon/config.py
Precedence: default.yaml < hardware/<hw>.yaml < <mode>.yaml < local.yaml < overrides.

``local.yaml`` is git-ignored on purpose: it is the supported place for
machine-local values and credentials (e.g. ``keyframe.fashn_api_key``), so
secrets never enter a tracked file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
VALID_MODES = {"mock", "real", "hybrid"}


class Config:
    """Lightweight nested config with attribute access (``cfg.camera.width``)."""

    def __init__(self, data: Mapping[str, Any], *, source: str = "<memory>") -> None:
        object.__setattr__(self, "_data", copy.deepcopy(dict(data)))
        object.__setattr__(self, "source", source)

    # -- read -----------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            return Config(value, source=self.source) if isinstance(value, dict) else value
        raise AttributeError(f"config has no key {name!r} (source={self.source})")

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_data")

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def keys(self) -> Iterable[str]:
        return object.__getattribute__(self, "_data").keys()

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = object.__getattribute__(self, "_data")
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        if isinstance(node, dict):
            return Config(node, source=self.source)
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {path}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(object.__getattribute__(self, "_data"))

    # -- write (tests / programmatic overrides) --------------------------
    def set(self, path: str, value: Any) -> "Config":
        parts = path.split(".")
        node: dict[str, Any] = object.__getattribute__(self, "_data")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = copy.deepcopy(value)
        _validate_config(self)
        return self

    def override(self, updates: Mapping[str, Any]) -> "Config":
        _merge(object.__getattribute__(self, "_data"), dict(updates))
        _validate_config(self)
        return self

    def clone(self) -> "Config":
        return Config(self.to_dict(), source=self.source)

    def __repr__(self) -> str:
        return f"Config(mode={self.get('mode')!r}, source={self.source!r})"


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    _merge(base, overlay)
    return base


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def load_config(
    mode: str | None = None,
    hardware: str | None = None,
    *,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR
    merged: dict[str, Any] = {}
    files = [root / "default.yaml"]
    if hardware:
        files.append(root / "hardware" / f"{hardware}.yaml")
    # Resolve the mode layer from default.yaml when the caller did not pin one,
    # so ``load_config(None)`` still picks up configs/mock.yaml (needed for the
    # hidden mock ground truth) while ``load_config("real")`` picks real.yaml.
    resolved_mode = mode
    if not resolved_mode:
        resolved_mode = str(_read_yaml(root / "default.yaml").get("mode", "mock"))
    files.append(root / f"{resolved_mode}.yaml")
    local = root / "local.yaml"
    if local.exists():
        # Git-ignored machine-local overlay: credentials / hardware tweaks.
        files.append(local)
    for path in files:
        _deep_merge(merged, _read_yaml(path))
    if mode:
        merged["mode"] = mode
    if overrides:
        _merge(merged, dict(overrides))
    cfg = Config(merged, source=str(root))
    if validate:
        _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config) -> None:
    mode = cfg.get("mode", "mock")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    width = int(cfg.require("camera.width"))
    height = int(cfg.require("camera.height"))
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")

    dt = float(cfg.require("math.dt"))
    if not (0.0 < dt <= 0.2):
        raise ValueError("math.dt must be in (0, 0.2]")
    if list(cfg.require("math.gravity")) != [0.0, -9.81, 0.0]:
        raise ValueError("math.gravity must equal the SPEC Appendix-B value [0,-9.81,0]")
    if str(cfg.get("math.world_up", "Y")) != "Y":
        raise ValueError("world_up must be Y (SPEC §1.2)")
    if str(cfg.get("math.camera_forward", "Z")) != "Z":
        raise ValueError("camera_forward must be Z (SPEC §1.2)")
    if str(cfg.get("math.rotation_repr", "unit_quaternion")) != "unit_quaternion":
        raise ValueError("rotation_repr must be unit_quaternion (SPEC §1.2)")

    phys_dt = float(cfg.require("physics.dt"))
    target_hz = float(cfg.require("physics.target_hz"))
    if abs(phys_dt * target_hz - 1.0) > 1e-3:
        raise ValueError("physics.dt must be 1/physics.target_hz")
    if int(cfg.require("physics.substeps")) < 1:
        raise ValueError("physics.substeps must be >= 1")
    if int(cfg.require("physics.xpbd_iters")) < 1:
        raise ValueError("physics.xpbd_iters must be >= 1")
    gamma = float(cfg.require("physics.energy_decay_gamma"))
    if not (0.0 < gamma < 1.0):
        raise ValueError("physics.energy_decay_gamma must be in (0,1) [s^-1]")

    eps_in = float(cfg.require("keyframe.epsilon_in"))
    eps_out = float(cfg.require("keyframe.epsilon_out"))
    if not (0.0 < eps_in < eps_out):
        raise ValueError("keyframe hysteresis requires 0 < epsilon_in < epsilon_out")
    if int(cfg.require("keyframe.max_queue")) < 1:
        raise ValueError("keyframe.max_queue must be >= 1")

    if int(cfg.require("calibration.window_frames")) < 2:
        raise ValueError("calibration.window_frames must be >= 2")
    if float(cfg.require("calibration.kappa_max")) <= 1.0:
        raise ValueError("calibration.kappa_max must be > 1")
    if float(cfg.require("calibration.delta_max")) <= 0.0:
        raise ValueError("calibration.delta_max must be > 0")

    tau_back = float(cfg.require("asset.back_confidence_threshold"))
    if not (0.0 < tau_back < 1.0):
        raise ValueError("asset.back_confidence_threshold must be in (0,1)")


def config_fingerprint(cfg: Config) -> dict[str, Any]:
    """Subset recorded in provenance / logs (thresholds, never duplicated in code)."""
    return {
        "mode": cfg.get("mode"),
        "hardware": cfg.get("hardware"),
        "tau_back": cfg.get("asset.back_confidence_threshold"),
        "tau_review": cfg.get("asset.review_threshold"),
        "tau_vis": cfg.get("framing.min_visible_keypoint_ratio"),
        "tau_occ": cfg.get("framing.max_occlusion_ratio"),
        "kappa_max": cfg.get("calibration.kappa_max"),
        "delta_max": cfg.get("calibration.delta_max"),
        "gamma": cfg.get("physics.energy_decay_gamma"),
    }


__all__ = ["Config", "load_config", "config_fingerprint", "REPO_ROOT", "CONFIG_DIR"]

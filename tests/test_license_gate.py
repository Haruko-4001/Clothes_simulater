import pytest

from tryon.config import load_config
from tryon.policy.license_gate import LicenseGate
from tryon.types import TryonError


def test_real_mode_blocks_nc_and_nd_weights():
    cfg = load_config("real")
    gate = LicenseGate(cfg)
    with pytest.raises(TryonError) as exc:
        gate.check("catvton", mode="real")
    assert exc.value.code == "E_LICENSE_NC_BLOCKED"
    with pytest.raises(TryonError):
        gate.check("wilor", mode="real", weights_license="cc-by-nc-nd")
    with pytest.raises(TryonError):
        gate.check("fashn_vton_1_5", mode="real", weights_license="CC-BY-NC-SA")


def test_allow_nc_weights_switch_is_effective():
    cfg = load_config("real")
    cfg.set("policy.allow_nc_weights", True)
    gate = LicenseGate(cfg)
    gate.check("catvton", mode="real")  # explicitly allowed by config
    cfg.set("policy.allow_nc_weights", False)
    gate2 = LicenseGate(cfg)
    with pytest.raises(TryonError):
        gate2.check("catvton", mode="real")


def test_mock_mode_does_not_block_experimental_weights():
    cfg = load_config("mock")
    LicenseGate(cfg).check("wilor", mode="mock", weights_license="nc")


def test_real_default_startup_is_allowed():
    cfg = load_config("real")
    LicenseGate(cfg).assert_startup_allowed(cfg)

# adapters_vendor

Optional third-party checkouts live here. Nothing in this directory is required
for `mode: mock`.

Rules (SPEC §13):

- Mount as git submodules / shallow clones; do not copy source into `src/`.
- Probe the dependency first; on failure fall back to the stub and log
  `E_ADAPTER_FALLBACK`.
- Repository MIT/Apache does **not** imply weights are commercially usable.
- Red-line licenses: CatVTON / IDM-VTON / OOTDiffusion / StableVITON / CausVid
  are CC BY-NC-SA 4.0; WiLoR is CC-BY-NC-ND; SMPL-X / MANO are research-only
  and require registration. `policy/license_gate.py` blocks these in `mode: real`.
- Dress-1-to-3 and TryOnCrafter have no verified public code. They are design
  references only and must not be listed as clonable dependencies.
- Never fabricate repository names, weight URLs, fps, or measured numbers.

Use `scripts/install_real.sh --only=<adapter>` for an optional checkout.

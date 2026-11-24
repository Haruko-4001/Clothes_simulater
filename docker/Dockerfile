FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
# SPEC §19 DoD: SPEC.md and CLAUDE.md must live at the repo root, and the
# STRUCTURED_REPORT must be present for a complete delivery check in-container.
COPY pyproject.toml README.md SPEC.md CLAUDE.md STRUCTURED_REPORT.json ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY tests ./tests
COPY assets ./assets
COPY scripts ./scripts
COPY adapters_vendor ./adapters_vendor
COPY Makefile ./

RUN pip install --no-cache-dir -e ".[test]"

# Optional engine backends (Warp XPBD + OpenCV rasteriser). Uncomment to build
# the engine image. NOTE: python:3.11-slim has no CUDA toolkit, so Warp would
# run its CPU path only — the README GPU benchmark is NOT reproducible here.
# RUN pip install --no-cache-dir -e ".[test,engine]"

# Mock mode needs no GPU / network / weights / GUI.
CMD ["python", "-m", "app.main", "--mode", "mock", "--frames", "300", "--headless"]

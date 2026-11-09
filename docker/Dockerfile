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

# Mock mode needs no GPU / network / weights / GUI.
CMD ["python", "-m", "app.main", "--mode", "mock", "--frames", "300", "--headless"]

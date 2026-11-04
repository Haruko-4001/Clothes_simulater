FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY tests ./tests
COPY assets ./assets
COPY Makefile ./

RUN pip install --no-cache-dir -e ".[test]"

# Mock mode needs no GPU / network / weights / GUI.
CMD ["python", "-m", "app.main", "--mode", "mock", "--frames", "300", "--headless"]

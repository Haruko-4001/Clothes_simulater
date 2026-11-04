PYTHON ?= python
IMG ?= assets/samples/garment.png

.PHONY: install test mock-run bootstrap-asset eval-report smoke clean

install:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest tests -q

mock-run:
	$(PYTHON) -m app.main --mode mock --frames 300 --headless

bootstrap-asset:
	$(PYTHON) -m app.cli bootstrap-asset --image $(IMG)

eval-report:
	$(PYTHON) -m app.cli eval-report --log var/logs/run.jsonl --out var/eval

smoke:
	bash scripts/smoke.sh

clean:
	rm -rf var/logs/*.jsonl var/eval

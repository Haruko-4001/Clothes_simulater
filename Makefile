PYTHON ?= python
IMG ?= assets/samples/garment.png

.PHONY: install install-engine install-perception test mock-run bootstrap-asset eval-report bench serve smoke clean

install:
	$(PYTHON) -m pip install -e ".[test]"

install-engine:
	$(PYTHON) -m pip install -e ".[test,engine]"

install-perception:
	$(PYTHON) -m pip install -e ".[perception]"

test:
	$(PYTHON) -m pytest tests -q

mock-run:
	$(PYTHON) -m app.main --mode mock --frames 300 --headless

bootstrap-asset:
	$(PYTHON) -m app.cli bootstrap-asset --image $(IMG)

eval-report:
	$(PYTHON) -m app.cli eval-report --log var/logs/run.jsonl --out var/eval

bench:
	$(PYTHON) scripts/bench_engines.py --out var/eval/bench.json

serve:
	$(PYTHON) -m app.serve --physics-solver warp --render-backend opencv --geometry pattern

smoke:
	bash scripts/smoke.sh

clean:
	rm -rf var/logs/*.jsonl var/eval

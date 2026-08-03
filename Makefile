.PHONY: install test check

install:
	pip install -e . --no-deps
	pip install -r requirements-base.txt

test:
	pytest -q

check:
	python -m compileall -q hullwake scripts tools tests
	python tools/check_release.py .

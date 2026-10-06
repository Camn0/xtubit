PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src

.PHONY: test check
check:
	$(PYTHON) scripts/env_check.py

test:
	$(PYTHON) -m pytest -q tests

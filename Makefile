.PHONY: help setup lint test data backtest train report serve docker loadtest all clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

lint:  ## static checks
	ruff check src tests

test:  ## unit tests
	pytest

data:  ## download and clean the PJM East series
	loadfc data

backtest:  ## rolling-origin evaluation against the baselines
	loadfc backtest

train:  ## fit the final model and export serving artefacts
	loadfc train

report:  ## collect metrics and figures into reports/
	loadfc report

serve:  ## run the inference service locally
	loadfc serve

docker:  ## build and run the service with Prometheus alongside
	docker compose up --build

loadtest:  ## hammer the running service and report latency percentiles
	$(PY) scripts/loadtest.py

all: data backtest train report  ## full reproduction

clean:
	rm -rf models reports/*.json reports/*.png reports/*.parquet reports/RESULTS.md

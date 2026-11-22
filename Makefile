.PHONY: install test test-artifacts test-gpu test-simulator lint format build

install:
	uv sync --locked --extra all

test:
	uv run --locked --extra all pytest -m "not gpu and not simulator and not artifacts"

test-artifacts:
	uv run --locked --extra all pytest -m artifacts

test-gpu:
	ROBOREPAIR_RUN_GPU=1 ROBOREPAIR_RUN_SIMULATOR=1 uv run --no-sync pytest -m gpu

test-simulator:
	ROBOREPAIR_RUN_SIMULATOR=1 uv run --no-sync pytest -m simulator

lint:
	uv run --locked --extra all ruff check .
	uv run --locked --extra all ruff format --check .

format:
	uv run --locked --extra all ruff format .

build:
	uv build

.PHONY: help run install dev clean upload visualize

help:
	@echo "Available targets:"
	@echo "  run        - Run the main script using uv"
	@echo "  install    - Install dependencies using uv sync"
	@echo "  dev        - Install dependencies and run in development mode"
	@echo "  clean      - Clean up cache and temporary files"
	@echo "  visualize  - Generate data visualizations"

# Run the main script
run:
	uv run python -m src.main

# Install dependencies
install:
	uv sync

# Development setup: install deps and run
dev: install run

# Clean up cache and temporary files
clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

upload:
	make dev
	make visualize
	hf auth whoami
	hf upload younissk/tool-calling-mix README.md --repo-type=dataset
	hf upload younissk/tool-calling-mix images --repo-type=dataset
	hf upload younissk/tool-calling-mix output/tool_sft_corpus --repo-type=dataset

# Generate data visualizations
visualize:
	uv pip install matplotlib seaborn pandas
	uv run python -m src.visualize
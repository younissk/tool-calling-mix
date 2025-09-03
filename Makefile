.PHONY: help run run-enhanced install dev clean upload visualize validate test-quality all git

help:
	@echo "Available targets:"
	@echo "  run           - Run the main script using uv"
	@echo "  run-enhanced  - Run enhanced script with quality controls"
	@echo "  install       - Install dependencies using uv sync"
	@echo "  dev           - Install dependencies and run enhanced mode"
	@echo "  clean         - Clean up cache and temporary files"
	@echo "  visualize     - Generate data visualizations"
	@echo "  validate      - Run quality validation on existing dataset"
	@echo "  test-quality  - Run unit tests for quality control"

# Run the main script (standard)
run:
	uv run python -m src.main

# Run enhanced script with quality controls
run-enhanced:
	uv run python -m src.enhanced_main --validate

# Run enhanced script in strict mode
run-strict:
	uv run python -m src.enhanced_main --validate --strict

# Install dependencies
install:
	uv sync

# Development setup: install deps and run enhanced mode
dev: install run-enhanced

# Run quality validation on existing dataset
validate:
	uv run python -m src.validate_quality

# Run unit tests for quality control
test-quality:
	uv run python -m src.unit_tests

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
	hf auth whoami
	hf upload younissk/tool-calling-mix README.md --repo-type=dataset
	hf upload younissk/tool-calling-mix images --repo-type=dataset
	hf upload younissk/tool-calling-mix output/tool_sft_corpus --repo-type=dataset

all:
	make dev
	make upload

git:
	git add .
	git commit -m "Update dataset"
	git push
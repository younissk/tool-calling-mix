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
	@echo "  generate-synthetic - Generate synthetic parallel tool call data"
	@echo "  test-synthetic     - Test synthetic data generation with small sample"

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

# Generate synthetic parallel tool call data
generate-synthetic:
	@echo "Generating synthetic parallel tool call data..."
	uv run python -c "from dotenv import load_dotenv; load_dotenv(); from src.synthetic_parallel_generator import SyntheticParallelGenerator; from src.utils import json_dumps; import json; generator = SyntheticParallelGenerator(); print(f'Using LLM endpoint: {generator.hf_url}'); examples = generator.generate_synthetic_examples(5000); validated = generator.validate_data(examples); print(f'Generated {len(validated)} validated examples'); import os; os.makedirs('data', exist_ok=True); json.dump(validated, open('data/synthetic_parallel_tool_calls.json', 'w'), indent=2); print('Saved to data/synthetic_parallel_tool_calls.json')"

# Test synthetic data generation with small sample
test-synthetic:
	@echo "Testing synthetic data generation with small sample..."
	uv run python -c "from dotenv import load_dotenv; load_dotenv(); from src.synthetic_parallel_generator import SyntheticParallelGenerator; generator = SyntheticParallelGenerator(); print(f'Using LLM endpoint: {generator.hf_url}'); examples = generator.generate_synthetic_examples(10); validated = generator.validate_data(examples); print(f'Test generated {len(validated)} examples'); [print(f'Example {i+1}: {ex[\"question\"][0][0][\"content\"][:50]}...') for i, ex in enumerate(validated[:3])]"

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
	make run-enhanced
	# Create with strict validation (fails on quality issues)  
	make run-strict
	# Validate existing dataset
	make validate
	# Run quality unit tests
	make test-quality
	make dev
	make upload

git:
	git add .
	git commit -m "Update dataset"
	git push
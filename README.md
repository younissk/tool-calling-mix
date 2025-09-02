---
title: Tool Calling SFT Mix
license: apache-2.0
tags:
- tool-calling
- function-calling
- instruction-tuning
- supervised-fine-tuning
- language-model
- artificial-intelligence
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- other
language:
- en
pretty_name: Tool Calling SFT Mix
dataset_info:
  features:
  - name: tools_json
    dtype: string
  - name: messages_json
    dtype: string
  - name: target_json
    dtype: string
  - name: meta_source
    dtype: string
  - name: n_calls
    dtype: int32
  - name: difficulty
    dtype: string
  - name: valid
    dtype: bool
  splits:
  - name: train
    num_bytes: null
    num_examples: null
  download_size: null
  dataset_size: null
---

This is a dataset for fine-tuning a language model to use tools. I combined sources from various other tool calling datasets and added some non-tool calling examples to prevent catastrophic forgetting.

## Quick Start

### Create and Save Dataset

```bash
# Create the mixed dataset with Hub-compatible schema
uv run python -m src.main

# Or specify custom output path
uv run python -m src.main --output-path output/my_dataset
```

### Upload to Hugging Face Hub

```bash
# Create dataset and upload to Hub
uv run python -m src.main --repo-id YOUR_USERNAME/tool-calling-sft-mix-v1

# Upload existing dataset only
uv run python -m src.main --upload-only --repo-id YOUR_USERNAME/tool-calling-sft-mix-v1

# Make dataset private
uv run python -m src.main --repo-id YOUR_USERNAME/tool-calling-sft-mix-v1 --private
```

### Features

- **Automatic Schema Fix**: Ensures Parquet-compatible data types for Hugging Face Hub
- **JSONL Backup**: Creates compressed JSONL files for future-proofing
- **Hub Integration**: Direct upload to Hugging Face Hub with proper error handling
- **No Internal Columns**: Automatically removes problematic `_format_kwargs` and similar columns

## Upstream Sources

I adapted and unified examples from the following sources. Please cite them if you use this dataset:

- Zhang, J. et al. (2024). xLAM: A Family of Large Action Models to Empower AI Agents. arXiv:2409.03215.
  Dataset: Salesforce/xlam-function-calling-60k (HF).
  (If using the parsed variant: minpeter/xlam-function-calling-60k-parsed, HF.)

- Patil, S. G., Zhang, T., Wang, X., Gonzalez, J. E. (2024). Gorilla: Large Language Model Connected with Massive APIs.
  NeurIPS 2024. Project: <https://gorilla.cs.berkeley.edu> (OpenFunctions).

- Databricks (2023). databricks-dolly-15k (HF). License: CC BY-SA 3.0.

- Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.
  Dataset: Salesforce/wikitext – subset `wikitext-103-raw-v1` (HF). License: CC BY-SA.

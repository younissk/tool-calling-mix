"""Utilities for uploading datasets to Hugging Face Hub."""

import gzip
import json
from pathlib import Path
from datasets import Dataset


def create_jsonl_backup(dataset: Dataset, output_dir: str) -> None:
    """Create JSONL.gz backup files for future-proofing."""
    output_path = Path(output_dir) / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL.gz
    jsonl_path = output_path / "train.jsonl.gz"
    print(f"Creating JSONL backup: {jsonl_path}")

    with gzip.open(jsonl_path, 'wt', encoding='utf-8') as f:
        for row in dataset:
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')

    print("JSONL backup created successfully!")


def save_dataset_with_backup(
    dataset: Dataset,
    output_path: str,
    create_jsonl: bool = True
) -> None:
    """
    Save dataset to disk with optional JSONL backup.

    Args:
        dataset: The dataset to save
        output_path: Path to save the dataset
        create_jsonl: Whether to create JSONL backup files
    """
    print(f"Saving dataset to: {output_path}")
    dataset.save_to_disk(output_path)
    print("Dataset saved successfully!")

    if create_jsonl:
        create_jsonl_backup(dataset, output_path)

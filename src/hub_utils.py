"""Utilities for uploading datasets to Hugging Face Hub."""

import gzip
import json
from pathlib import Path
from datasets.splits import NamedSplit
from datasets import Dataset, DatasetDict, Features, Value


# Define standard features schema
FEATURES = Features({
    "tools_json": Value("string"),
    "messages_json": Value("string"),
    "target_json": Value("string"),
    "meta_source": Value("string"),
    "n_calls": Value("int32"),
    "difficulty": Value("string"),
    "valid": Value("bool"),
})


def create_jsonl_backup(dataset: Dataset, output_dir: str, split: str | NamedSplit) -> None:
    """Create JSONL.gz backup files for future-proofing."""
    output_path = Path(output_dir) / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL.gz
    jsonl_path = output_path / f"{split}.jsonl.gz"
    print(f"Creating JSONL backup for {split}: {jsonl_path}")

    with gzip.open(jsonl_path, 'wt', encoding='utf-8') as f:
        for row in dataset:
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')

    print(f"JSONL backup for {split} created successfully!")


def save_dataset_with_backup(
    dataset: Dataset | DatasetDict,
    output_path: str,
    create_jsonl: bool = True
) -> None:
    """
    Save dataset to disk with JSONL files only (no Arrow format).

    Args:
        dataset: The dataset or DatasetDict to save
        output_path: Path to save the dataset
        create_jsonl: Whether to create JSONL backup files
    """
    print(f"Saving dataset to: {output_path}")
    
    if isinstance(dataset, DatasetDict):
        # Save each split as JSONL only
        for split_name, split_dataset in dataset.items():
            if create_jsonl:
                create_jsonl_backup(split_dataset, output_path, split_name)
    else:
        # Single dataset case
        if create_jsonl:
            create_jsonl_backup(dataset, output_path, "train")
            
    print("Dataset saved successfully!")


def prepare_for_hub(dataset: Dataset | DatasetDict) -> DatasetDict:
    """
    Prepare dataset for Hub upload by cleaning formatting and casting features.
    
    Args:
        dataset: Input dataset or DatasetDict
        
    Returns:
        Cleaned DatasetDict ready for upload
    """
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
        
    # Clean formatting and cast features
    for split in dataset.values():
        # Remove any internal formatting columns
        for col in list(split.features.keys()):
            if col.startswith('_'):
                split = split.remove_columns(col)
        split.reset_format()
        
        # Cast to standard features
        split = split.cast(FEATURES)
    
    return dataset

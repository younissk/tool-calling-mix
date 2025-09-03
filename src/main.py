
"""Main entry point for the tool-calling-sft-mix project."""

import random
from datasets import DatasetDict
from src.data_loaders import create_mixed_dataset
from src.hub_utils import save_dataset_with_backup, prepare_for_hub
from src.visualize import generate_visualizations
from dotenv import load_dotenv


def create_splits(dataset, train_size=0.8, val_size=0.1, seed=42):
    """Create train/validation/test splits."""
    # Shuffle and split the dataset
    shuffled = dataset.shuffle(seed=seed)
    n_samples = len(shuffled)

    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    # Create splits
    train_dataset = shuffled.select(range(train_end))
    val_dataset = shuffled.select(range(train_end, val_end))
    test_dataset = shuffled.select(range(val_end, n_samples))

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def main():
    """Main function to create and process the mixed dataset."""
    load_dotenv()
    print("Creating tool-calling-sft-mix dataset...")

    # Set random seed for reproducibility
    random.seed(42)

    # Create the mixed dataset
    mixed_dataset = create_mixed_dataset()

    # Create train/val/test splits
    print("\nCreating dataset splits...")
    dataset_dict = create_splits(mixed_dataset)

    # Print split sizes
    for split_name, split in dataset_dict.items():
        print(f"{split_name}: {len(split)} examples")

    # Prepare for Hub upload
    print("\nPreparing dataset for Hub...")
    clean_dataset = prepare_for_hub(dataset_dict)

    return clean_dataset


if __name__ == "__main__":
    # Create and save the dataset
    dataset = main()
    save_dataset_with_backup(dataset, "output/tool_sft_corpus")

    # Generate visualizations
    print("\nGenerating data visualizations...")
    try:
        stats = generate_visualizations()
        print("✓ Visualizations generated successfully!")
        print(
            f"✓ Dataset contains {stats['total_examples']} examples ({stats['valid_examples']} valid)")
        print(
            f"✓ Average tool calls per example: {stats['avg_tool_calls']:.2f}")
    except Exception as e:
        print(f"⚠ Warning: Visualization generation failed: {e}")
        print("Dataset creation completed successfully, but visualizations were skipped.")

    print("\nDataset is ready for Hub upload!")

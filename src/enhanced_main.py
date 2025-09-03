"""Enhanced main entry point with quality controls for the tool-calling-sft-mix project."""

import random
import argparse
from datasets import DatasetDict
from src.enhanced_data_loaders import create_enhanced_mixed_dataset
from src.hub_utils import save_dataset_with_backup, prepare_for_hub
from src.visualize import generate_visualizations
from src.validate_quality import validate_enhanced_dataset
from src.config import (
    ENABLE_QUALITY_CONTROLS, 
    MIN_NO_CALL_PERCENTAGE, 
    MAX_NO_CALL_PERCENTAGE,
    RANDOM_SEED
)
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


def check_dataset_balance(dataset) -> bool:
    """Check if dataset meets quality balance requirements."""
    print("\n🔍 Checking dataset balance...")
    
    # Count no-call examples
    no_call_count = 0
    total_count = len(dataset)
    
    # Sample check for large datasets
    sample_size = min(total_count, 5000)
    step = max(1, total_count // sample_size)
    
    for i in range(0, total_count, step):
        example = dataset[i]
        if example.get("n_calls", 0) == 0:
            no_call_count += 1
    
    # Scale up the count
    actual_no_call_count = no_call_count * step
    no_call_percentage = actual_no_call_count / total_count * 100
    
    print(f"📊 No-call percentage: {no_call_percentage:.1f}% (target: {MIN_NO_CALL_PERCENTAGE}-{MAX_NO_CALL_PERCENTAGE}%)")
    
    is_balanced = MIN_NO_CALL_PERCENTAGE <= no_call_percentage <= MAX_NO_CALL_PERCENTAGE
    
    if is_balanced:
        print("✅ Dataset balance is within target range!")
    elif no_call_percentage < MIN_NO_CALL_PERCENTAGE:
        print(f"⚠️  Need {(MIN_NO_CALL_PERCENTAGE - no_call_percentage) * total_count / 100:.0f} more no-call examples")
    else:
        print(f"⚠️  Need {(no_call_percentage - MAX_NO_CALL_PERCENTAGE) * total_count / 100:.0f} more tool-calling examples")
    
    return is_balanced


def main(args):
    """Enhanced main function with quality controls."""
    load_dotenv()
    
    if ENABLE_QUALITY_CONTROLS:
        print("🚀 Creating enhanced tool-calling-sft-mix dataset with quality controls...")
    else:
        print("📦 Creating standard tool-calling-sft-mix dataset...")

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Create the mixed dataset (enhanced or standard)
    if ENABLE_QUALITY_CONTROLS:
        mixed_dataset = create_enhanced_mixed_dataset()
    else:
        from src.data_loaders import create_mixed_dataset
        mixed_dataset = create_mixed_dataset()

    # Check dataset balance
    is_balanced = check_dataset_balance(mixed_dataset)
    if not is_balanced and args.strict:
        print("❌ Dataset balance check failed in strict mode. Exiting.")
        return None

    # Create train/val/test splits
    print("\n📊 Creating dataset splits...")
    dataset_dict = create_splits(mixed_dataset, seed=RANDOM_SEED)

    # Print split sizes
    for split_name, split in dataset_dict.items():
        print(f"  {split_name}: {len(split)} examples")

    # Prepare for Hub upload
    print("\n🔧 Preparing dataset for Hub...")
    clean_dataset = prepare_for_hub(dataset_dict)

    # Run quality validation if enabled
    if ENABLE_QUALITY_CONTROLS and args.validate:
        print("\n🔍 Running quality validation...")
        
        # Save temporarily for validation
        temp_path = "temp_validation_dataset"
        save_dataset_with_backup(clean_dataset, temp_path)
        
        # Validate
        validation_results = validate_enhanced_dataset(temp_path)
        
        if not validation_results.get("overall_success", False):
            print("❌ Quality validation failed!")
            if args.strict:
                print("Exiting due to validation failure in strict mode.")
                return None
            else:
                print("⚠️  Continuing despite validation warnings...")
        else:
            print("✅ Quality validation passed!")
        
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(temp_path)
        except:
            pass

    return clean_dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced tool-calling dataset creation with quality controls"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Run quality validation after dataset creation"
    )
    parser.add_argument(
        "--strict", 
        action="store_true", 
        help="Exit on quality validation failures"
    )
    parser.add_argument(
        "--skip-visualize", 
        action="store_true", 
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--output-dir", 
        default="output/tool_sft_corpus", 
        help="Output directory for the dataset"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("🔧 Enhanced Tool-Calling SFT Mix Dataset Creator")
    print("=" * 50)
    print(f"Quality controls: {'✅ Enabled' if ENABLE_QUALITY_CONTROLS else '❌ Disabled'}")
    print(f"Validation: {'✅ Enabled' if args.validate else '❌ Disabled'}")
    print(f"Strict mode: {'✅ Enabled' if args.strict else '❌ Disabled'}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Create and save the dataset
    dataset = main(args)
    
    if dataset is None:
        print("❌ Dataset creation failed or was aborted.")
        exit(1)
    
    # Save the dataset
    print(f"\n💾 Saving dataset to {args.output_dir}...")
    save_dataset_with_backup(dataset, args.output_dir)
    print("✅ Dataset saved successfully!")

    # Generate visualizations unless skipped
    if not args.skip_visualize:
        print("\n📊 Generating data visualizations...")
        try:
            stats = generate_visualizations()
            print("✅ Visualizations generated successfully!")
            print(f"📈 Dataset contains {stats['total_examples']} examples ({stats['valid_examples']} valid)")
            print(f"📊 Average tool calls per example: {stats['avg_tool_calls']:.2f}")
        except Exception as e:
            print(f"⚠️  Warning: Visualization generation failed: {e}")
            print("Dataset creation completed successfully, but visualizations were skipped.")

    print("\n🎉 Enhanced dataset is ready for Hub upload!")
    print(f"📁 Location: {args.output_dir}")
    
    if ENABLE_QUALITY_CONTROLS:
        print("\n🔧 Quality enhancements included:")
        print("  ✅ Schema-strict exemplars for failure patterns")
        print("  ✅ Negative clarification examples") 
        print("  ✅ ToolBench field normalization")
        print("  ✅ Balanced no-tool instructions")
        print("  ✅ Adversarial variants")
        print("  ✅ Programmatic validation")
    
    # Final validation summary
    if args.validate:
        print(f"\n📋 To re-run validation: python -m src.validate_quality {args.output_dir}")
    
    print("\n🚀 Next steps:")
    print("  1. Review generated visualizations in images/")
    print("  2. Run: make upload (to push to Hugging Face Hub)")
    print("  3. Test with your fine-tuning pipeline")

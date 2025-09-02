
"""Main entry point for the tool-calling-sft-mix project."""

from src.data_loaders import create_mixed_dataset
from dotenv import load_dotenv

def main():
    """Main function to create and process the mixed dataset."""
    load_dotenv()
    print("Hello from tool-calling-sft-mix!")
    
    # Create the mixed dataset
    mixed_dataset = create_mixed_dataset()
    
    # You can add additional processing here if needed
    # For example: save the dataset, perform analysis, etc.
    
    return mixed_dataset


if __name__ == "__main__":
    dataset = main()
    
    # Save the Dataset
    dataset.save_to_disk("output/tool_sft_corpus")
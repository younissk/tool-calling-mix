"""Data visualization script for analyzing the tool-calling dataset."""

import json
import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_dataset():
    """Load the dataset from JSONL.gz files."""
    data_dir = Path("output/tool_sft_corpus/raw")
    all_data = []
    
    # Load all splits
    for split_file in ["train.jsonl.gz", "validation.jsonl.gz", "test.jsonl.gz"]:
        file_path = data_dir / split_file
        if file_path.exists():
            print(f"Loading {split_file}...")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
    
    print(f"Loaded {len(all_data)} examples total")
    return all_data


def process_dataset(dataset):
    """Convert dataset to pandas DataFrame with extracted features."""
    records = []
    for item in dataset:
        record = {
            'meta_source': item.get('meta_source', 'unknown'),
            'n_calls': item.get('n_calls', 0),
            'difficulty': item.get('difficulty', 'unknown'),
            'valid': item.get('valid', False)
        }
        
        # Extract message features
        try:
            messages = json.loads(item.get('messages_json', '[]'))
            record['msg_count'] = len(messages)
            record['user_msg_len'] = len(messages[0]['content']) if messages else 0
        except:
            record['msg_count'] = 0
            record['user_msg_len'] = 0
            
        # Extract tool features
        try:
            tools = json.loads(item.get('tools_json', '[]'))
            record['tool_count'] = len(tools)
        except:
            record['tool_count'] = 0
            
        # Extract target features
        try:
            target = json.loads(item.get('target_json', '{}'))
            tool_calls = target.get('tool_calls', [])
            record['actual_tool_calls'] = len(tool_calls)
            record['tool_names'] = [call.get('name', 'unknown') for call in tool_calls]
        except:
            record['actual_tool_calls'] = 0
            record['tool_names'] = []
            
        records.append(record)
    
    return pd.DataFrame(records)


def plot_dataset_composition(df: pd.DataFrame, output_dir: str):
    """Create pie charts showing dataset composition by source and difficulty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Source distribution
    source_counts = df['meta_source'].value_counts()
    ax1.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Dataset Composition by Source')
    
    # Difficulty distribution
    diff_counts = df['difficulty'].value_counts()
    ax2.pie(diff_counts.values, labels=diff_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Dataset Composition by Difficulty')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_composition.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_tool_call_distribution(df: pd.DataFrame, output_dir: str):
    """Create histograms showing tool call distribution."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall distribution
    ax1.hist(df['n_calls'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution of Tool Calls')
    ax1.set_xlabel('Number of Tool Calls')
    ax1.set_ylabel('Frequency')
    
    # Distribution by source
    for source in df['meta_source'].unique():
        source_data = df[df['meta_source'] == source]['n_calls']
        ax2.hist(source_data, bins=20, alpha=0.7, label=source)
    ax2.set_title('Tool Calls by Source')
    ax2.set_xlabel('Number of Tool Calls')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Distribution by difficulty
    for diff in df['difficulty'].unique():
        diff_data = df[df['difficulty'] == diff]['n_calls']
        ax3.hist(diff_data, bins=20, alpha=0.7, label=diff)
    ax3.set_title('Tool Calls by Difficulty')
    ax3.set_xlabel('Number of Tool Calls')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Available vs Used tools
    ax4.scatter(df['tool_count'], df['actual_tool_calls'], alpha=0.5)
    ax4.set_title('Available vs Used Tools')
    ax4.set_xlabel('Available Tools')
    ax4.set_ylabel('Used Tools')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tool_call_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_message_analysis(df: pd.DataFrame, output_dir: str):
    """Create visualizations for message analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Message length distribution
    ax1.hist(df['user_msg_len'], bins=30, alpha=0.7, edgecolor='black')
    ax1.set_title('Message Length Distribution')
    ax1.set_xlabel('Message Length (characters)')
    ax1.set_ylabel('Frequency')
    
    # Message length vs tool calls
    ax2.scatter(df['user_msg_len'], df['n_calls'], alpha=0.5)
    ax2.set_title('Message Length vs Tool Calls')
    ax2.set_xlabel('Message Length (characters)')
    ax2.set_ylabel('Number of Tool Calls')
    
    # Message length by source
    source_data = [df[df['meta_source'] == source]['user_msg_len'] for source in df['meta_source'].unique()]
    ax3.boxplot(source_data, labels=df['meta_source'].unique())
    ax3.set_title('Message Length by Source')
    ax3.set_ylabel('Message Length (characters)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Message length by difficulty
    diff_data = [df[df['difficulty'] == diff]['user_msg_len'] for diff in df['difficulty'].unique()]
    ax4.boxplot(diff_data, labels=df['difficulty'].unique())
    ax4.set_title('Message Length by Difficulty')
    ax4.set_ylabel('Message Length (characters)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/message_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_tool_usage_patterns(df: pd.DataFrame, output_dir: str):
    """Analyze and visualize tool usage patterns."""
    # Extract all unique tool names and count their usage
    tool_counts = {}
    for tools in df['tool_names']:
        for tool in tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    if not tool_counts:
        print("No tool usage data found")
        return
    
    # Create DataFrame for visualization
    tool_df = pd.DataFrame([
        {'tool': tool, 'count': count}
        for tool, count in tool_counts.items()
    ]).sort_values('count', ascending=False)
    
    # Plot top 20 tools
    top_tools = tool_df.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_tools)), top_tools['count'])
    ax.set_yticks(range(len(top_tools)))
    ax.set_yticklabels(top_tools['tool'])
    ax.set_xlabel('Usage Count')
    ax.set_title('Top 20 Most Used Tools')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tool_usage_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations():
    """Generate all visualizations."""
    print("Loading dataset...")
    dataset = load_dataset()
    
    print("Processing dataset...")
    df = process_dataset(dataset)
    
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    plot_dataset_composition(df, output_dir)
    plot_tool_call_distribution(df, output_dir)
    plot_message_analysis(df, output_dir)
    plot_tool_usage_patterns(df, output_dir)
    
    print("Generating summary statistics...")
    stats = {
        'total_examples': len(df),
        'valid_examples': int(df['valid'].sum()),
        'avg_tool_calls': float(df['n_calls'].mean()),
        'max_tool_calls': int(df['n_calls'].max()),
        'difficulty_distribution': df['difficulty'].value_counts().to_dict(),
        'source_distribution': df['meta_source'].value_counts().to_dict(),
    }
    
    with open(f"{output_dir}/summary_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nVisualization generation complete!")
    print("- PNG versions of all plots are available in the /images directory")
    print("- Summary statistics saved to /images/summary_stats.json")
    return stats


if __name__ == "__main__":
    generate_visualizations()
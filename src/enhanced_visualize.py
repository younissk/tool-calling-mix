"""Enhanced visualization module for the tool-calling dataset."""

import os
import json
from typing import Dict, Any, List, cast, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.text import Text
import seaborn as sns
from datasets import load_dataset, Dataset

# Type aliases
PieTuple = Tuple[List[Wedge], List[Text], List[Text]]

# Set style for all plots
plt.style.use('seaborn-v0_8-white')
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 12,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold'
})

def process_dataset(dataset):
    """Process dataset into a pandas DataFrame with enhanced metrics."""
    records = []
    for example in dataset:
        # Parse JSON fields
        tools = json.loads(example['tools_json'])
        messages = json.loads(example['messages_json'])
        target = json.loads(example['target_json'])
        
        # Extract user message length
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        user_msg_len = len(user_messages[0]) if user_messages else 0
        
        # Extract tool information
        tool_names = [tool['name'] for tool in tools]
        tool_count = len(tools)
        actual_tool_calls = len(target.get('tool_calls', []))
        
        # Create record
        record = {
            'meta_source': example['meta_source'],
            'difficulty': example['difficulty'],
            'valid': example['valid'],
            'n_calls': example['n_calls'],
            'user_msg_len': user_msg_len,
            'tool_names': tool_names,
            'tool_count': tool_count,
            'actual_tool_calls': actual_tool_calls,
            'message_count': len(messages),
            'avg_message_len': np.mean([len(msg['content']) for msg in messages]),
            'complexity_score': actual_tool_calls * (user_msg_len / 1000)  # Normalized complexity metric
        }
        records.append(record)
    
    return pd.DataFrame(records)

def plot_dataset_composition(df: pd.DataFrame, output_dir: str) -> None:
    """Create enhanced pie charts showing dataset composition with better organization."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    
    # Color palettes - using colorblind-friendly palettes
    source_colors = sns.color_palette("husl", n_colors=len(df['meta_source'].unique()))
    difficulty_colors = sns.color_palette("Set2", n_colors=len(df['difficulty'].unique()))
    
    # Source distribution - Group minor sources
    source_counts = df['meta_source'].value_counts()
    threshold = int(len(df) * 0.05)  # 5% threshold
    major_sources = source_counts[source_counts >= threshold]
    minor_sources = pd.Series({'Others': source_counts[source_counts < threshold].sum()})
    source_data = pd.concat([major_sources, minor_sources])
    
    # Plot source distribution
    ax1 = fig.add_subplot(gs[0])
    wedges1, texts1, autotexts1 = cast(
        tuple[List[Wedge], List[Text], List[Text]],
        ax1.pie(
            source_data.values.astype(float),
            labels=[str(x) for x in source_data.index],
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*len(df)):,})',
            colors=source_colors[:len(source_data)],
            explode=[0.05] * len(source_data),
            startangle=90,
            pctdistance=0.85
        )
    )
    
    # Add center circle for donut effect
    centre_circle = Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    
    # Add title with total count
    ax1.set_title(f'Dataset Composition by Source\nTotal: {len(df):,} examples', 
                  pad=20, fontsize=14, fontweight='bold')
    
    # Difficulty distribution
    ax2 = fig.add_subplot(gs[1])
    diff_counts = df['difficulty'].value_counts()
    
    # Calculate percentages and counts
    total = diff_counts.sum()
    diff_labels = [f'{idx}\n{val:,} ({val/total*100:.1f}%)' 
                  for idx, val in diff_counts.items()]
    
    wedges2, texts2, autotexts2 = cast(
        tuple[List[Wedge], List[Text], List[Text]],
        ax2.pie(
            diff_counts.values.astype(float),
            labels=diff_labels,
            colors=difficulty_colors,
            explode=[0.05] * len(diff_counts),
            startangle=90,
            pctdistance=0.85
        )
    )
    
    # Add center circle for donut effect
    centre_circle = Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    
    ax2.set_title('Dataset Composition by Difficulty', pad=20, fontsize=14, fontweight='bold')
    
    # Style enhancements
    plt.setp(autotexts1 + autotexts2, size=10, weight="bold", color='white')
    plt.setp(texts1 + texts2, size=11)
    
    # Add legends with enhanced styling
    legend1 = ax1.legend(
        wedges1, 
        [f'{idx} ({val:,})' for idx, val in source_data.items()],
        title="Sources",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=True,
        edgecolor='black'
    )
    if legend1:
        legend1.get_frame().set_alpha(0.9)
    
    legend2 = ax2.legend(
        wedges2, 
        [f'{idx} ({val:,})' for idx, val in diff_counts.items()],
        title="Difficulty Levels",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=True,
        edgecolor='black'
    )
    if legend2:
        legend2.get_frame().set_alpha(0.9)
    
    plt.savefig(f"{output_dir}/dataset_composition.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_message_analysis(df: pd.DataFrame, output_dir: str):
    """Create enhanced visualizations for message analysis with better insights."""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Message length distribution with KDE
    ax1 = fig.add_subplot(gs[0, 0])
    # Use log scale for better visualization of long tail
    sns.histplot(data=df, x='user_msg_len', kde=True, ax=ax1, 
                bins=50, log_scale=(False, True))
    
    # Add mean and median lines
    mean_len = df['user_msg_len'].mean()
    median_len = df['user_msg_len'].median()
    ax1.axvline(mean_len, color='red', linestyle='--', alpha=0.8,
                label=f'Mean: {mean_len:,.0f}')
    ax1.axvline(median_len, color='green', linestyle='--', alpha=0.8,
                label=f'Median: {median_len:,.0f}')
    
    ax1.set_title('Message Length Distribution\n(Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Message Length (characters)', fontsize=12)
    ax1.set_ylabel('Count (log scale)', fontsize=12)
    ax1.legend()
    
    # Message length vs tool calls with trend line and density
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=df, x='user_msg_len', y='n_calls', 
                   alpha=0.3, size='complexity_score',
                   sizes=(20, 200), ax=ax2)
    
    # Add trend line
    sns.regplot(data=df, x='user_msg_len', y='n_calls',
                scatter=False, color='red', ax=ax2)
    
    # Calculate correlation
    corr = df['user_msg_len'].corr(df['n_calls'])
    ax2.set_title(f'Message Length vs Tool Calls\nCorrelation: {corr:.2f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Message Length (characters)', fontsize=12)
    ax2.set_ylabel('Number of Tool Calls', fontsize=12)
    
    # Message length by source - Enhanced violin plot
    ax3 = fig.add_subplot(gs[1, 0])
    # Sort sources by median message length
    source_medians = df.groupby('meta_source')['user_msg_len'].median().sort_values()
    source_order = source_medians.index
    
    sns.violinplot(data=df, x='meta_source', y='user_msg_len', ax=ax3,
                   order=source_order, inner='box', cut=0)
    
    # Add swarm plot for actual data points
    sns.swarmplot(data=df, x='meta_source', y='user_msg_len', ax=ax3,
                  order=source_order, size=3, color='red', alpha=0.3)
    
    ax3.set_title('Message Length Distribution by Source\nwith Data Points', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Source', fontsize=12)
    ax3.set_ylabel('Message Length (characters)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, ha='right')
    
    # Message complexity by difficulty with enhanced visualization
    ax4 = fig.add_subplot(gs[1, 1])
    # Create violin plot with embedded box plot
    sns.violinplot(data=df, x='difficulty', y='complexity_score', ax=ax4,
                   inner='box', cut=0)
    
    # Add individual points for better distribution visibility
    sns.stripplot(data=df, x='difficulty', y='complexity_score', ax=ax4,
                 size=3, color='red', alpha=0.3, jitter=0.2)
    
    # Add mean values as text
    for i, diff in enumerate(df['difficulty'].unique()):
        mean_val = df[df['difficulty'] == diff]['complexity_score'].mean()
        ax4.text(i, ax4.get_ylim()[1], f'Mean: {mean_val:.2f}',
                 ha='center', va='bottom')
    
    ax4.set_title('Message Complexity by Difficulty\nwith Distribution', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Difficulty Level', fontsize=12)
    ax4.set_ylabel('Complexity Score', fontsize=12)
    
    # Add overall figure title
    fig.suptitle('Message Analysis Dashboard', fontsize=16, fontweight='bold', y=0.95)
    
    plt.savefig(f"{output_dir}/message_analysis.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_tool_call_analysis(df: pd.DataFrame, output_dir: str):
    """Create enhanced visualizations for tool call patterns."""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Overall distribution with enhanced CDF
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate statistics
    mean_calls = df['n_calls'].mean()
    median_calls = df['n_calls'].median()
    mode_calls = df['n_calls'].mode().iloc[0]
    
    # Create histogram with KDE
    sns.histplot(data=df, x='n_calls', stat='density', alpha=0.7,
                color='skyblue', edgecolor='black', ax=ax1)
    
    # Add CDF on twin axis
    ax1_twin = ax1.twinx()
    sns.ecdfplot(data=df, x='n_calls', ax=ax1_twin, color='red', linewidth=2)
    
    # Add mean, median, mode lines
    ax1.axvline(mean_calls, color='red', linestyle='--', alpha=0.8,
                label=f'Mean: {mean_calls:.2f}')
    ax1.axvline(median_calls, color='green', linestyle='--', alpha=0.8,
                label=f'Median: {median_calls:.2f}')
    ax1.axvline(mode_calls, color='blue', linestyle='--', alpha=0.8,
                label=f'Mode: {mode_calls}')
    
    ax1.set_title('Distribution of Tool Calls\nwith Summary Statistics', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Tool Calls', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1_twin.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.legend(loc='upper right')
    
    # Tool calls by source - Enhanced violin plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Sort sources by median tool calls
    source_medians = df.groupby('meta_source')['n_calls'].median().sort_values()
    source_order = source_medians.index
    
    # Create violin plot with box plot inside
    sns.violinplot(data=df, x='meta_source', y='n_calls', ax=ax2,
                   order=source_order, inner='box', cut=0)
    
    # Add individual points
    sns.stripplot(data=df, x='meta_source', y='n_calls', ax=ax2,
                 order=source_order, size=3, color='red', alpha=0.3, jitter=0.2)
    
    # Add median values as text
    for i, source in enumerate(source_order):
        median_val = df[df['meta_source'] == source]['n_calls'].median()
        ax2.text(i, ax2.get_ylim()[1], f'Median: {median_val:.1f}',
                 ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Tool Calls Distribution by Source\nwith Individual Points', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Source', fontsize=12)
    ax2.set_ylabel('Number of Tool Calls', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, ha='right')
    
    # Tool efficiency with enhanced visualization
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create hexbin plot for density
    hb = ax3.hexbin(df['tool_count'], df['actual_tool_calls'],
                    gridsize=20, cmap='YlOrRd', mincnt=1)
    
    # Add diagonal line
    max_tools = max(df['tool_count'].max(), df['actual_tool_calls'].max())
    ax3.plot([0, max_tools], [0, max_tools], 'r--', 
             label='Perfect Efficiency (1:1)', alpha=0.8)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax3)
    cb.set_label('Number of Examples', fontsize=10)
    
    # Calculate and add efficiency metrics
    efficiency = (df['actual_tool_calls'] / df['tool_count']).mean()
    ax3.text(0.05, 0.95, f'Average Efficiency: {efficiency:.2%}',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_title('Tool Usage Efficiency\nAvailable vs Actually Used Tools', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Available Tools', fontsize=12)
    ax3.set_ylabel('Used Tools', fontsize=12)
    ax3.legend()
    
    # Tool calls by difficulty with enhanced visualization
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create violin plot
    sns.violinplot(data=df, x='difficulty', y='n_calls', ax=ax4,
                   inner='box', cut=0)
    
    # Add individual points
    sns.stripplot(data=df, x='difficulty', y='n_calls', ax=ax4,
                 size=3, color='red', alpha=0.3, jitter=0.2)
    
    # Add statistics
    for i, diff in enumerate(df['difficulty'].unique()):
        stats = df[df['difficulty'] == diff]['n_calls']
        mean_val = stats.mean()
        median_val = stats.median()
        ax4.text(i, ax4.get_ylim()[1], 
                f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    ax4.set_title('Tool Calls by Difficulty Level\nwith Distribution', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Difficulty Level', fontsize=12)
    ax4.set_ylabel('Number of Tool Calls', fontsize=12)
    
    # Add overall figure title
    fig.suptitle('Tool Call Analysis Dashboard', fontsize=16, fontweight='bold', y=0.95)
    
    plt.savefig(f"{output_dir}/tool_call_analysis.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_tool_usage_patterns(df: pd.DataFrame, output_dir: str):
    """Create enhanced visualization of tool usage patterns with categories."""
    # Extract and count tool usage
    tool_counts = {}
    tool_by_source = {}
    tool_by_difficulty = {}
    
    for _, row in df.iterrows():
        for tool in row['tool_names']:
            # Overall counts
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # Counts by source
            if tool not in tool_by_source:
                tool_by_source[tool] = {}
            source = row['meta_source']
            tool_by_source[tool][source] = tool_by_source[tool].get(source, 0) + 1
            
            # Counts by difficulty
            if tool not in tool_by_difficulty:
                tool_by_difficulty[tool] = {}
            diff = row['difficulty']
            tool_by_difficulty[tool][diff] = tool_by_difficulty[tool].get(diff, 0) + 1
    
    if not tool_counts:
        print("No tool usage data found")
        return
    
    # Create DataFrame with enhanced metrics
    tool_df = pd.DataFrame([
        {
            'tool': tool,
            'count': count,
            'percentage': (count/len(df))*100,
            'sources': len(tool_by_source[tool]),
            'difficulties': len(tool_by_difficulty[tool]),
            'primary_source': max(tool_by_source[tool].items(), key=lambda x: x[1])[0],
            'source_concentration': max(tool_by_source[tool].values()) / sum(tool_by_source[tool].values())
        }
        for tool, count in tool_counts.items()
    ]).sort_values('count', ascending=True)
    
    # Get top 20 tools
    top_tools = tool_df.tail(20)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Main usage plot
    ax1 = fig.add_subplot(gs[0])
    
    # Create horizontal bars with categorical colors
    colors = sns.color_palette("husl", n_colors=len(top_tools['primary_source'].unique()))
    color_map = dict(zip(top_tools['primary_source'].unique(), colors))
    bars = ax1.barh(range(len(top_tools)), top_tools['count'],
                    color=[color_map[src] for src in top_tools['primary_source']],
                    alpha=0.8)
    
    # Customize the main plot
    ax1.set_yticks(range(len(top_tools)))
    ax1.set_yticklabels([
        f"{tool}\n({sources} sources, {diffs} difficulties)"
        for tool, sources, diffs in zip(top_tools['tool'], 
                                      top_tools['sources'],
                                      top_tools['difficulties'])
    ], fontsize=10)
    
    ax1.set_xlabel('Usage Count', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Most Used Tools\nwith Source Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels with enhanced information
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = top_tools.iloc[i]['percentage']
        concentration = top_tools.iloc[i]['source_concentration']
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(width):,} ({percentage:.1f}%)\nSource Concentration: {concentration:.1%}',
                ha='left', va='center', fontsize=9)
    
    # Add grid and style
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add source distribution legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[src], alpha=0.8)
                      for src in color_map]
    ax1.legend(legend_elements, color_map.keys(),
              title='Primary Source',
              loc='center left',
              bbox_to_anchor=(1, 0.5))
    
    # Usage distribution plot
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate usage distribution statistics
    usage_stats = pd.DataFrame({
        'Total Usage': tool_df['count'].value_counts().sort_index(),
        'Cumulative Tools': np.arange(1, len(tool_df) + 1)
    })
    
    # Plot usage distribution
    sns.scatterplot(data=usage_stats, x=usage_stats.index, y='Total Usage',
                   ax=ax2, alpha=0.6, color='blue', label='Individual Tools')
    ax2.set_yscale('log')
    
    # Add trend line
    z = np.polyfit(np.log(usage_stats.index), np.log(usage_stats['Total Usage']), 1)
    p = np.poly1d(z)
    ax2.plot(usage_stats.index, np.exp(p(np.log(usage_stats.index))),
             'r--', label=f'Power Law (α={z[0]:.2f})')
    
    # Add cumulative line on twin axis
    ax2_twin = ax2.twinx()
    sns.lineplot(data=usage_stats, x=usage_stats.index, y='Cumulative Tools',
                ax=ax2_twin, color='green', label='Cumulative Tools')
    
    # Customize distribution plot
    ax2.set_xlabel('Tool Rank (by usage)', fontsize=12)
    ax2.set_ylabel('Usage Count (log scale)', fontsize=12)
    ax2_twin.set_ylabel('Cumulative Number of Tools', fontsize=12)
    ax2.set_title('Tool Usage Distribution\nPower Law Analysis', 
                  fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add summary statistics
    stats_text = (
        f"Total Unique Tools: {len(tool_df):,}\n"
        f"Mean Usage: {tool_df['count'].mean():.1f}\n"
        f"Median Usage: {tool_df['count'].median():.1f}\n"
        f"Top 20% Tools Account for: {(tool_df.nlargest(int(len(tool_df)*0.2), 'count')['count'].sum() / tool_df['count'].sum()*100):.1f}% of Usage"
    )
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f"{output_dir}/tool_usage_patterns.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def generate_enhanced_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate enhanced summary statistics with more insights."""
    
    # Calculate tool usage patterns
    tool_counts = {}
    tool_by_source = {}
    tool_by_difficulty = {}
    
    for _, row in df.iterrows():
        for tool in row['tool_names']:
            # Overall counts
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # Counts by source
            if tool not in tool_by_source:
                tool_by_source[tool] = {}
            source = row['meta_source']
            tool_by_source[tool][source] = tool_by_source[tool].get(source, 0) + 1
            
            # Counts by difficulty
            if tool not in tool_by_difficulty:
                tool_by_difficulty[tool] = {}
            diff = row['difficulty']
            tool_by_difficulty[tool][diff] = tool_by_difficulty[tool].get(diff, 0) + 1
    
    # Calculate advanced metrics
    tool_df = pd.DataFrame([
        {
            'tool': tool,
            'count': count,
            'percentage': (count/len(df))*100,
            'sources': len(tool_by_source[tool]),
            'difficulties': len(tool_by_difficulty[tool]),
            'primary_source': max(tool_by_source[tool].items(), key=lambda x: x[1])[0],
            'source_concentration': max(tool_by_source[tool].values()) / sum(tool_by_source[tool].values())
        }
        for tool, count in tool_counts.items()
    ]).sort_values('count', ascending=False)
    
    # Calculate message complexity by source
    source_complexity = df.groupby('meta_source').agg({
        'complexity_score': ['mean', 'median', 'std', 'max'],
        'user_msg_len': ['mean', 'median', 'std'],
        'n_calls': ['mean', 'median', 'count']
    }).round(2).to_dict()
    
    # Calculate correlations
    correlations = {
        'msg_len_vs_tool_calls': float(df['user_msg_len'].corr(df['n_calls'])),
        'msg_len_vs_complexity': float(df['user_msg_len'].corr(df['complexity_score'])),
        'tool_calls_vs_complexity': float(df['n_calls'].corr(df['complexity_score']))
    }
    
    stats = {
        'dataset_overview': {
            'total_examples': len(df),
            'valid_examples': int(df['valid'].sum()),
            'invalid_examples': len(df) - int(df['valid'].sum()),
            'quality_score': float((df['valid'].sum() / len(df)) * 100),
            'total_unique_tools': len(tool_counts),
            'avg_tools_per_example': float(df['tool_count'].mean()),
            'total_tool_calls': int(df['n_calls'].sum())
        },
        'tool_usage_metrics': {
            'avg_tool_calls': float(df['n_calls'].mean()),
            'median_tool_calls': float(df['n_calls'].median()),
            'std_tool_calls': float(df['n_calls'].std()),
            'max_tool_calls': int(df['n_calls'].max()),
            'min_tool_calls': int(df['n_calls'].min()),
            'tool_call_percentiles': {
                'p10': float(df['n_calls'].quantile(0.10)),
                'p25': float(df['n_calls'].quantile(0.25)),
                'p50': float(df['n_calls'].quantile(0.50)),
                'p75': float(df['n_calls'].quantile(0.75)),
                'p90': float(df['n_calls'].quantile(0.90)),
                'p95': float(df['n_calls'].quantile(0.95)),
                'p99': float(df['n_calls'].quantile(0.99))
            },
            'zero_tool_call_percentage': float((df['n_calls'] == 0).mean() * 100),
            'single_tool_call_percentage': float((df['n_calls'] == 1).mean() * 100),
            'multiple_tool_call_percentage': float((df['n_calls'] > 1).mean() * 100)
        },
        'message_metrics': {
            'avg_message_length': float(df['user_msg_len'].mean()),
            'median_message_length': float(df['user_msg_len'].median()),
            'std_message_length': float(df['user_msg_len'].std()),
            'max_message_length': int(df['user_msg_len'].max()),
            'min_message_length': int(df['user_msg_len'].min()),
            'message_length_percentiles': {
                'p10': float(df['user_msg_len'].quantile(0.10)),
                'p25': float(df['user_msg_len'].quantile(0.25)),
                'p50': float(df['user_msg_len'].quantile(0.50)),
                'p75': float(df['user_msg_len'].quantile(0.75)),
                'p90': float(df['user_msg_len'].quantile(0.90)),
                'p95': float(df['user_msg_len'].quantile(0.95)),
                'p99': float(df['user_msg_len'].quantile(0.99))
            }
        },
        'difficulty_distribution': {
            'counts': df['difficulty'].value_counts().to_dict(),
            'percentages': (df['difficulty'].value_counts(normalize=True) * 100).round(2).to_dict(),
            'avg_tool_calls_by_difficulty': df.groupby('difficulty')['n_calls'].mean().round(2).to_dict(),
            'avg_complexity_by_difficulty': df.groupby('difficulty')['complexity_score'].mean().round(2).to_dict()
        },
        'source_distribution': {
            'counts': df['meta_source'].value_counts().to_dict(),
            'percentages': (df['meta_source'].value_counts(normalize=True) * 100).round(2).to_dict(),
            'source_complexity_metrics': source_complexity
        },
        'tool_usage_patterns': {
            'top_10_tools': tool_df.head(10)[['tool', 'count', 'percentage', 'sources', 'primary_source']].to_dict('records'),
            'tool_source_diversity': {
                'avg_sources_per_tool': float(tool_df['sources'].mean()),
                'max_sources_per_tool': int(tool_df['sources'].max()),
                'tools_in_all_sources': int((tool_df['sources'] == len(df['meta_source'].unique())).sum())
            },
            'tool_concentration': {
                'top_10_percent_usage': float((tool_df.head(int(len(tool_df)*0.1))['count'].sum() / tool_df['count'].sum()) * 100),
                'top_20_percent_usage': float((tool_df.head(int(len(tool_df)*0.2))['count'].sum() / tool_df['count'].sum()) * 100)
            }
        },
        'complexity_metrics': {
            'avg_complexity_score': float(df['complexity_score'].mean()),
            'median_complexity_score': float(df['complexity_score'].median()),
            'std_complexity_score': float(df['complexity_score'].std()),
            'max_complexity_score': float(df['complexity_score'].max()),
            'min_complexity_score': float(df['complexity_score'].min()),
            'complexity_percentiles': {
                'p10': float(df['complexity_score'].quantile(0.10)),
                'p25': float(df['complexity_score'].quantile(0.25)),
                'p50': float(df['complexity_score'].quantile(0.50)),
                'p75': float(df['complexity_score'].quantile(0.75)),
                'p90': float(df['complexity_score'].quantile(0.90)),
                'p95': float(df['complexity_score'].quantile(0.95)),
                'p99': float(df['complexity_score'].quantile(0.99))
            }
        },
        'correlations': correlations,
        'metadata': {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'version': '2.0.0',
            'stats_schema_version': '2.0.0'
        }
    }
    return stats

def generate_visualizations(dataset_path: str = "younissk/tool-calling-mix") -> Dict[str, Any]:
    """Generate all enhanced visualizations.
    
    Args:
        dataset_path: Path to the dataset on the Hugging Face Hub.
        
    Returns:
        Dict containing the generated statistics.
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    if not dataset or not isinstance(dataset, Dataset):
        raise ValueError(f"Failed to load dataset from {dataset_path}")
    
    print("Processing dataset...")
    df = process_dataset(dataset)
    
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating enhanced visualizations...")
    plot_dataset_composition(df, output_dir)
    plot_message_analysis(df, output_dir)
    plot_tool_call_analysis(df, output_dir)
    plot_tool_usage_patterns(df, output_dir)
    
    print("Generating enhanced summary statistics...")
    stats = generate_enhanced_stats(df)
    
    with open(f"{output_dir}/summary_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nEnhanced visualization generation complete!")
    print("- PNG versions of all plots are available in the /images directory")
    print("- Enhanced summary statistics saved to /images/summary_stats.json")
    return stats

if __name__ == "__main__":
    generate_visualizations()

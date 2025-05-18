#!/usr/bin/env python3
import ast
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from analysis.utils import build_docker_image

# Constants
PROJECT_NAME = "block_ratio"
NUM_ITERATIONS = 500
TARGET_BLOCK_COUNT = 10
OUTPUT_DIR = Path("outputs")
PROJECT_DIR = OUTPUT_DIR / PROJECT_NAME
PLOTS_DIR = PROJECT_DIR / "plots"
RESULTS_DIR = PROJECT_DIR / "results"

# ============= DOMAIN LOGIC =============


def run_benchmark(block_count):
    """Run a benchmark with the specified block count."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR.absolute()}:/app/outputs",
        "metadrive",
        "--map", str(block_count),
        "--num-scenarios", "1",
        "--benchmark", str(NUM_ITERATIONS),
        "--project-name", PROJECT_NAME
    ]
    
    subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True
    )
    return True


def parse_csv_line(line):
    """
    Parse a single line from the benchmark CSV file and remove the 'I' starter block.
    """
    reader = csv.reader([line])
    fields = next(reader)
    
    try:
        block_count = int(fields[2])
        time_elapsed = float(fields[5])
        block_type_counts_str = fields[4]
        block_counts = ast.literal_eval(block_type_counts_str)
    except (ValueError, SyntaxError, IndexError) as e:
        return None
    
    # Remove 'I' block and count how many were removed
    i_count = 0
    if 'I' in block_counts:
        i_count = block_counts['I']
        del block_counts['I']
    
    data_entry = {
        'total_blocks': block_count - i_count,  # Subtract actual number of I blocks
        'time_elapsed': time_elapsed,
        'block_counts': block_counts
    }
    
    for block_type, count in block_counts.items():
        data_entry[f'count_{block_type}'] = count
    
    return data_entry


def collect_block_type_data():
    """
    Collect block type counts and generation times from all benchmarks.
    """
    all_data = []
    
    for block_dir in PROJECT_DIR.glob("blocks*"):
        if not block_dir.is_dir():
            continue
        
        csv_path = block_dir / "data" / "benchmark.csv"
        
        if not csv_path.exists():
            continue
        
        with open(csv_path, 'r') as file:
            # Skip header
            file.readline()
            
            for line in file:
                data_entry = parse_csv_line(line)
                if data_entry:
                    all_data.append(data_entry)
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    
    # Fill any missing block type counts with 0
    block_columns = [col for col in df.columns if col.startswith('count_')]
    for col in block_columns:
        df[col] = df[col].fillna(0)
    
    return df


def calculate_block_ratios(data_df):
    """
    Calculate the ratio of each block type relative to total blocks.
    """
    # Get all block types from the dataframe (excluding I)
    block_columns = [col for col in data_df.columns if col.startswith('count_')]
    block_types = [col.replace('count_', '') for col in block_columns]
    
    # Calculate ratio for each block type
    for block_type in block_types:
        ratio_col = f'ratio_{block_type}'
        count_col = f'count_{block_type}'
        
        # Calculate ratio: count / total_blocks
        data_df[ratio_col] = data_df[count_col] / data_df['total_blocks']
    
    return data_df


def analyze_ratio_correlation(data_df):
    """
    Analyze correlation between block type ratios and generation time.
    """
    # Get all ratio columns
    ratio_columns = [col for col in data_df.columns if col.startswith('ratio_')]
    block_types = [col.replace('ratio_', '') for col in ratio_columns]
    
    # Calculate Pearson correlation for each block type ratio
    correlations = {}
    for block_type, ratio_col in zip(block_types, ratio_columns):
        # Use try/except to handle zero variance case more elegantly
        try:
            corr, p_value = pearsonr(data_df[ratio_col], data_df['time_elapsed'])
            correlations[block_type] = {
                'pearson_r': corr,
                'p_value': p_value
            }
        except:
            correlations[block_type] = {
                'pearson_r': 0,
                'p_value': 1.0
            }
    
    # Sort block types by absolute correlation value
    sorted_blocks = sorted(
        correlations.keys(),
        key=lambda x: abs(correlations[x]['pearson_r']),
        reverse=True
    )
    
    # Create results dictionary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = {
        'metadata': {
            'timestamp': timestamp,
            'sample_size': len(data_df)
        },
        'correlations': correlations,
        'sorted_blocks': sorted_blocks,
        'raw_data': {
            'time_elapsed': data_df['time_elapsed'].tolist()
        }
    }
    
    # Add raw ratio data for each block type
    for block_type in block_types:
        ratio_col = f'ratio_{block_type}'
        results['raw_data'][block_type] = data_df[ratio_col].tolist()
    
    return results


def save_results(results):
    """Save all results and visualizations."""
    # Extract timestamp
    timestamp = results["metadata"]["timestamp"]
    
    # Save as JSON
    json_path = RESULTS_DIR / f"block_ratio_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    corr_data = []
    for block_type in results['sorted_blocks']:
        corr_data.append({
            'block_type': block_type,
            'pearson_r': results['correlations'][block_type]['pearson_r'],
            'p_value': results['correlations'][block_type]['p_value']
        })
    
    corr_df = pd.DataFrame(corr_data)
    csv_path = RESULTS_DIR / f"block_ratio_{timestamp}.csv"
    corr_df.to_csv(csv_path, index=False)
    
    # Create and save visualizations
    visualize_individual_blocks(results)


# ============= VISUALIZATION LOGIC =============

def visualize_individual_blocks(results):
    """Create separate scatter plots for each block type."""
    timestamp = results["metadata"]["timestamp"]
    time_elapsed = results['raw_data']['time_elapsed']
    
    # Process each block type separately (excluding I)
    for block_type in results['sorted_blocks']:
        # Get data for this block type
        ratio_data = results['raw_data'][block_type]
        

        if len(set(ratio_data)) <= 1:
            continue
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(ratio_data, time_elapsed, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(ratio_data, time_elapsed, 1)
        p = np.poly1d(z)
        x_sorted = sorted(ratio_data)
        plt.plot(x_sorted, p(x_sorted), "r--", alpha=0.8)
        
        # Get correlation info
        corr = results['correlations'][block_type]['pearson_r']
        p_value = results['correlations'][block_type]['p_value']
        
        # Format title based on correlation
        corr_direction = "increases" if corr > 0 else "decreases"
        title = f'Block {block_type}: Time {corr_direction} with proportion'
        
        # Create subtitle with correlation and p-value
        subtitle = f'r = {corr:.3f}, p = {p_value:.4f}'
        
        # Add significance marker with clear threshold indication
        if p_value < 0.05:
            plt.figtext(0.5, 0.01, 
                      "* p < 0.05: statistically significant correlation",
                      ha='center', fontsize=9)
            subtitle += " *"
            
        plt.title(f"{title}\n{subtitle}", fontsize=11)
        plt.xlabel(f'Proportion of Block {block_type} in Map')
        plt.ylabel('Generation Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save the plot
        plt_path = PLOTS_DIR / f"block_{block_type}_{timestamp}.png"
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return True


def main():
    """
    Run the block ratio analysis to determine how the relative proportion of 
    each block type affects generation time.
    """
    # Create all necessary directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    run_new_benchmarks = input("Run new benchmarks? (y/n, default: n): ").strip().lower() == 'y'
    
    if run_new_benchmarks:
        build_docker_image()
        run_benchmark(TARGET_BLOCK_COUNT)
    
    df = collect_block_type_data()
    
    if df is None or df.empty:
        return
    
    df_with_ratios = calculate_block_ratios(df)
    results = analyze_ratio_correlation(df_with_ratios)
    
    save_results(results)


if __name__ == "__main__":
    main() 
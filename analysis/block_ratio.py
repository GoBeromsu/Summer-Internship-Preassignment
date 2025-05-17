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

# Constants
PROJECT_NAME = "block_ratio"
DOCKER_IMAGE = "metadrive"
NUM_ITERATIONS = 500
TARGET_BLOCK_COUNT = 10
OUTPUT_DIR = Path("outputs")

# ============= DOMAIN LOGIC =============

def build_docker_image():
    """Build the Docker image with the name 'metadrive'."""
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "."],
        check=True,
        text=True,
        capture_output=True
    )


def run_benchmark(block_count):
    """Run a benchmark with the specified block count."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR.absolute()}:/app/outputs",
        DOCKER_IMAGE,
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


def parse_csv_line(line, block_dir):
    """
    Parse a single line from the benchmark CSV file and remove the 'I' starter block.
    """
    reader = csv.reader([line])
    fields = next(reader)
    
    if len(fields) < 6:
        return None
    
    block_count = int(fields[2])
    time_elapsed = float(fields[5])
    
    block_type_counts_str = fields[4]
    block_counts = ast.literal_eval(block_type_counts_str)
    
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
    project_dir = OUTPUT_DIR / PROJECT_NAME
    
    for block_dir in project_dir.glob("blocks*"):
        if not block_dir.is_dir():
            continue
        
        csv_path = block_dir / "data" / "benchmark.csv"
        
        if not csv_path.exists():
            continue
        
        with open(csv_path, 'r') as file:
            # Skip header
            file.readline()
            
            for line in file:
                data_entry = parse_csv_line(line, block_dir)
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
        
        # Ensure count column exists (with 0 if missing)
        if count_col not in data_df.columns:
            data_df[count_col] = 0
            
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
        # Skip if all values are 0 (no variance)
        if data_df[ratio_col].var() == 0:
            correlations[block_type] = {
                'pearson_r': 0,
                'p_value': 1.0
            }
            continue
            
        corr, p_value = pearsonr(data_df[ratio_col], data_df['time_elapsed'])
        correlations[block_type] = {
            'pearson_r': corr,
            'p_value': p_value
        }
    
    # Sort block types by absolute correlation value
    sorted_blocks = sorted(
        correlations.keys(),
        key=lambda x: abs(correlations[x]['pearson_r']),
        reverse=True
    )
    
    # Create results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
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
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    project_dir = OUTPUT_DIR / PROJECT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = project_dir / "plots"
    results_dir = project_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to files
    timestamp = results["metadata"]["timestamp"]
    
    # Save as JSON
    json_path = results_dir / f"block_ratio_{timestamp}.json"
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
    csv_path = results_dir / f"block_ratio_{timestamp}.csv"
    corr_df.to_csv(csv_path, index=False)
    
    # Create and save visualizations
    visualize_individual_blocks(results, plots_dir)


# ============= VISUALIZATION LOGIC =============

def visualize_individual_blocks(results, plots_dir):
    """Create separate scatter plots for each block type."""
    timestamp = results["metadata"]["timestamp"]
    time_elapsed = results['raw_data']['time_elapsed']
    
    # Process each block type separately (excluding I)
    for block_type in results['sorted_blocks']:
        # Get data for this block type
        ratio_data = results['raw_data'][block_type]
        
        # Skip if no variance in ratio values
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
        plt_path = plots_dir / f"block_{block_type}_{timestamp}.png"
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return True


def main():
    """
    Run the block ratio analysis to determine how the relative proportion of 
    each block type affects generation time.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
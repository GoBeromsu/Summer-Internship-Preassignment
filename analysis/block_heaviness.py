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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Constants
PROJECT_NAME = "block_heaviness"
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
    
    The 'I' block is removed because it's always present as the starting block
    in every map and would skew the analysis results.
    """
    reader = csv.reader([line])
    fields = next(reader)
    
    if len(fields) < 6:
        return None
    
    block_count = int(fields[2])
    time_elapsed = float(fields[5])
    
    block_type_counts_str = fields[4]
    block_counts = ast.literal_eval(block_type_counts_str)
    
    # Remove one "I" block (always present as starting block)
    if 'I' in block_counts and block_counts['I'] > 0:
        block_counts['I'] -= 1
        if block_counts['I'] == 0:
            del block_counts['I']
    
    data_entry = {
        'total_blocks': block_count,
        'time_elapsed': time_elapsed
    }
    
    for block_type, count in block_counts.items():
        data_entry[f'count_{block_type}'] = count
    
    return data_entry


def collect_block_type_data():
    """
    Collect block type counts and generation times from all benchmarks
    for linear regression analysis.
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


def analyze_block_heaviness(data_df):
    """
    Find which block types contribute most to map generation time.
    
    Uses linear regression to model: time ≈ a × count_A + b × count_B + ...
    where coefficients represent each block type's impact on generation time.
    """
    if data_df is None or len(data_df) < 10:
        return None
    
    # Extract time data and block counts
    y = data_df['time_elapsed']
    block_type_columns = [col for col in data_df.columns if col.startswith('count_')]
    
    if not block_type_columns:
        return None
    
    X = data_df[block_type_columns]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Extract coefficients as heaviness scores
    coefficients = model.coef_
    heaviness_scores = {}
    for idx, block_type_col in enumerate(block_type_columns):
        block_type = block_type_col.replace('count_', '')
        heaviness_scores[block_type] = coefficients[idx]
    
    # Calculate R-squared to evaluate the model
    y_pred = model.predict(X_scaled)
    r_squared = r2_score(y, y_pred)
    
    # Create results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'r_squared': r_squared,
            'sample_size': len(data_df)
        },
        'block_heaviness': heaviness_scores
    }
    
    return results


def create_output_directories():
    """Create necessary output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    project_dir = OUTPUT_DIR / PROJECT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = project_dir / "plots"
    results_dir = project_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return project_dir, plots_dir, results_dir


def save_results_to_files(results, plots_dir, results_dir):
    """Save analysis results to JSON and CSV files."""
    timestamp = results["metadata"]["timestamp"]
    
    # Save as JSON
    json_path = results_dir / f"block_heaviness_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    heaviness_df = pd.DataFrame({
        'block_type': list(results['block_heaviness'].keys()),
        'heaviness_score': list(results['block_heaviness'].values())
    })
    heaviness_df = heaviness_df.sort_values('heaviness_score', ascending=False)
    csv_path = results_dir / f"block_heaviness_{timestamp}.csv"
    heaviness_df.to_csv(csv_path, index=False)
    
    return json_path, csv_path


def visualize_heaviness(results):
    """Create a bar chart visualization of block heaviness."""
    if not results or 'block_heaviness' not in results:
        return None
    
    block_types = list(results['block_heaviness'].keys())
    heaviness_scores = list(results['block_heaviness'].values())
    
    # Sort by heaviness score (descending)
    sorted_indices = np.argsort(heaviness_scores)[::-1]
    sorted_block_types = [block_types[i] for i in sorted_indices]
    sorted_scores = [heaviness_scores[i] for i in sorted_indices]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    colors = ['red' if score > 0 else 'green' for score in sorted_scores]
    plt.bar(sorted_block_types, sorted_scores, color=colors)
    
    plt.title('Block Type Heaviness Analysis (Standardized β-Coefficients)')
    plt.xlabel('Block Type')
    plt.ylabel('Relative Time Contribution (β-Coefficient)')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add R-squared annotation
    r_squared = results['metadata']['r_squared']
    plt.annotate(f"R² = {r_squared:.3f}", xy=(0.95, 0.95), 
                 xycoords='axes fraction', ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    return plt


def save_plot(plt, plots_dir, timestamp):
    """Save the visualization plot to a file."""
    plt_path_png = plots_dir / f"block_heaviness_{timestamp}.png"
    plt.savefig(plt_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    return plt_path_png


def save_results(results):
    """Save all results and visualizations."""
    if not results:
        return
    
    timestamp = results["metadata"]["timestamp"]
    project_dir, plots_dir, results_dir = create_output_directories()
    save_results_to_files(results, plots_dir, results_dir)
    
    plt = visualize_heaviness(results)
    if plt:
        save_plot(plt, plots_dir, timestamp)


def main():
    """
    Run the block heaviness analysis to identify which blocks take more time to generate.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_new_benchmarks = input("Run new benchmarks? (y/n, default: n): ").strip().lower() == 'y'
    
    if run_new_benchmarks:
        build_docker_image()
        run_benchmark(TARGET_BLOCK_COUNT)
    
    df = collect_block_type_data()
    
    if df is None or df.empty:
        return
    
    results = analyze_block_heaviness(df)
    
    if not results:
        return
    
    save_results(results)


if __name__ == "__main__":
    main() 
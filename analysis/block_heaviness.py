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
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Constants
PROJECT_NAME = "block_heaviness"
DOCKER_IMAGE = "metadrive"
NUM_ITERATIONS = 1550
TARGET_BLOCK_COUNT = 15
OUTPUT_DIR = Path("outputs")
PROJECT_DIR = OUTPUT_DIR / PROJECT_NAME
PLOTS_DIR = PROJECT_DIR / "plots"
RESULTS_DIR = PROJECT_DIR / "results"

# ============= DOMAIN LOGIC =============

def build_docker_image():
    """Build the Docker image with the name 'metadrive'."""
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "."],
        check=True,
        text=True,
        capture_output=True
    )


def run_benchmark():
    """Run a benchmark with the specified block count."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR.absolute()}:/app/outputs",
        DOCKER_IMAGE,
        "--map", str(TARGET_BLOCK_COUNT),
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
    
    The 'I' block is removed because it's always present as the starting block
    in every map and would skew the analysis results.
    """
    reader = csv.reader([line])
    fields = next(reader)
    

    try:
        block_count = int(fields[2])
        time_elapsed = float(fields[5])
        block_type_counts_str = fields[4]   
        block_counts = ast.literal_eval(block_type_counts_str)
    except (ValueError, SyntaxError, IndexError) as e:
        raise ValueError(f"Malformed data line: {line.strip()} → {e}")
    
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


def collect_block_type_data(project_dir):
    """
    Collect block type counts and generation times from all benchmarks
    for linear regression analysis.
    """
    all_data = []

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
                all_data.append(parse_csv_line(line))
    
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
    
    Returns a results dictionary with analysis data and visualization.
    """
    # Extract time data and block counts
    y = data_df['time_elapsed']
    block_type_columns = [col for col in data_df.columns if col.startswith('count_')]
    
    if not block_type_columns:
        return None
    
    x = data_df[block_type_columns]
    
    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x_scaled, y)
    
    # Extract coefficients as heaviness scores
    coefficients = model.coef_
    heaviness_scores = {}
    
    # Calculate p-values for coefficients
    n = len(data_df)
    p = len(block_type_columns)
    y_pred = model.predict(x_scaled)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p - 1)
    
    # Calculate variance-covariance matrix
    x_scaled_with_const = np.hstack((np.ones((n, 1)), x_scaled))
    covariance = np.linalg.inv(np.dot(x_scaled_with_const.T, x_scaled_with_const))
    
    # Standard errors and t-statistics
    se = np.sqrt(np.diag(covariance)[1:] * mse)  # Skip intercept
    t_values = coefficients / se
    p_values = [2 * (1 - stats.t.cdf(abs(t), n - p - 1)) for t in t_values]
    
    # Store coefficients and p-values
    p_values_dict = {}
    for idx, block_type_col in enumerate(block_type_columns):
        block_type = block_type_col.replace('count_', '')
        heaviness_scores[block_type] = coefficients[idx]
        p_values_dict[block_type] = p_values[idx]
    
    # Calculate R-squared to evaluate the model
    r_squared = r2_score(y, y_pred)
    
    # Create results dictionary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = {
        'metadata': {
            'timestamp': timestamp,
            'r_squared': r_squared,
            'sample_size': len(data_df)
        },
        'block_heaviness': heaviness_scores,
        'p_values': p_values_dict
    }
    
    # Create visualization
    plot = create_heaviness_visualization(results)
    
    # Add plot to results
    results['plot'] = plot
    
    return results


def create_heaviness_visualization(results):
    """Create a bar chart visualization of block heaviness."""
    if not results or 'block_heaviness' not in results:
        return None
    
    block_types = list(results['block_heaviness'].keys())
    heaviness_scores = list(results['block_heaviness'].values())
    p_values = [results['p_values'].get(block_type, 1.0) for block_type in block_types]
    
    # Sort by heaviness score (descending)
    sorted_indices = np.argsort(heaviness_scores)[::-1]
    sorted_block_types = [block_types[i] for i in sorted_indices]
    sorted_scores = [heaviness_scores[i] for i in sorted_indices]
    sorted_p_values = [p_values[i] for i in sorted_indices]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Use colors to indicate significance
    colors = []
    for score, p_val in zip(sorted_scores, sorted_p_values):
        if p_val < 0.01:
            color = 'darkred' if score > 0 else 'darkgreen'  # Highly significant
        elif p_val < 0.05:
            color = 'red' if score > 0 else 'green'  # Significant
        else:
            color = 'lightcoral' if score > 0 else 'lightgreen'  # Not significant
        colors.append(color)
    
    bars = plt.bar(sorted_block_types, sorted_scores, color=colors)
    
    plt.title('Block Type Heaviness Analysis (Standardized β-Coefficients)')
    plt.xlabel('Block Type')
    plt.ylabel('Relative Time Contribution (β-Coefficient)')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add significance indicators
    for i, (p_val, bar) in enumerate(zip(sorted_p_values, bars)):
        significance = ''
        if p_val < 0.001:
            significance = '***'
        elif p_val < 0.01:
            significance = '**'
        elif p_val < 0.05:
            significance = '*'
        
        if significance:
            plt.text(bar.get_x() + bar.get_width()/2, 
                     bar.get_height() + (0.05 * max(abs(max(sorted_scores)), abs(min(sorted_scores)))), 
                     significance,
                     ha='center', va='bottom')
    
    # Add R-squared annotation
    r_squared = results['metadata']['r_squared']
    plt.annotate(f"R² = {r_squared:.3f}", xy=(0.95, 0.95), 
                 xycoords='axes fraction', ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add legend for significance
    plt.figtext(0.91, 0.05, "Significance:", ha="right")
    plt.figtext(0.91, 0.03, "* p<0.05, ** p<0.01, *** p<0.001", ha="right")
    
    return plt


def save_results(results):
    """Save analysis results to JSON, CSV, and visualization files."""
    timestamp = results['metadata']['timestamp']
    
    # Save as JSON
    json_path = RESULTS_DIR / f"block_heaviness_{timestamp}.json"
    
    # Create a copy of results without the plot for JSON serialization
    json_results = {k: v for k, v in results.items() if k != 'plot'}
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save as CSV
    heaviness_df = pd.DataFrame({
        'block_type': list(results['block_heaviness'].keys()),
        'heaviness_score': list(results['block_heaviness'].values()),
        'p_value': [results['p_values'].get(block_type, None) for block_type in results['block_heaviness'].keys()]
    })
    heaviness_df = heaviness_df.sort_values('heaviness_score', ascending=False)
    csv_path = RESULTS_DIR / f"block_heaviness_{timestamp}.csv"
    heaviness_df.to_csv(csv_path, index=False)
    
    plot_filename = f"block_heaviness_{timestamp}.png"
    plt_path_png = PLOTS_DIR / plot_filename
    plot = results['plot']
    plot.savefig(plt_path_png, dpi=300, bbox_inches='tight')
    plot.close()
    
    return json_path, csv_path, plt_path_png


def main():
    """
    Run the block heaviness analysis to identify which blocks take more time to generate.
    """
    # Create necessary directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    run_new_benchmarks = input("Run new benchmarks? (y/n, default: n): ").strip().lower() == 'y'
    
    if run_new_benchmarks:
        build_docker_image()
        run_benchmark()
    df = collect_block_type_data(PROJECT_DIR)
    
    if df is None or df.empty:
        return
    
    results = analyze_block_heaviness(df)
    save_results(results)


if __name__ == "__main__":
    main() 
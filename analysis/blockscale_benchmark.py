#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
PROJECT_NAME = "blockscale_benchmark"
DOCKER_IMAGE = "metadrive"
NUM_ITERATIONS = 3
BLOCK_RANGE = range(5, 70, 5)  # 5, 10, 15, ..., 100
OUTPUT_DIR = Path("outputs")
PROJECT_DIR = OUTPUT_DIR / PROJECT_NAME
PLOTS_DIR = PROJECT_DIR / "plots"
RESULTS_DIR = PROJECT_DIR / "results"
LINEAR_SCALE_DIR = PLOTS_DIR / "linear_scale"
LOG_SCALE_DIR = PLOTS_DIR / "log_scale"


def build_docker_image():
    """Build the Docker image with the name 'metadrive'."""
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "."],
        check=True,
        text=True,
        capture_output=True
    )
    print(f"Docker image '{DOCKER_IMAGE}' built successfully")


def run_benchmark(block_count):
    """Run a benchmark with the specified block count."""
    # Mount the base outputs directory so the container can create its own subdirs
    # Container will create: /app/outputs/{PROJECT_NAME}/blocks{block_count}/
    
    # Run docker with volume mount to get output
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{OUTPUT_DIR.absolute()}:/app/outputs",
        DOCKER_IMAGE,
        "--map", str(block_count),
        "--num-scenarios", "1",
        "--benchmark", str(NUM_ITERATIONS),
        "--project-name", PROJECT_NAME
    ]
    
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True
    )
    
    # Extract any useful output info
    if "Benchmark complete" in result.stdout:
        print(f"✓ Blocks {block_count} benchmark completed")
    return True


def read_csv_results():
    """Process all benchmark results from CSV files and create aggregated data."""
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "iterations_per_block": NUM_ITERATIONS,
            "project_name": PROJECT_NAME
        },
        "block_counts": [],
        "avg_times": [],
        "min_times": [],
        "max_times": [],
        "std_dev_times": []
    }
    
    for block_count in BLOCK_RANGE:
        block_dir = PROJECT_DIR / f"blocks{block_count}"
        # CSV files are stored in the data directory
        csv_path = block_dir / "data" / "benchmark.csv"
        
        if not csv_path.exists():
            print(f"No CSV data found for block count {block_count}")
            continue
        
        # Read the CSV file using pandas
        df = pd.read_csv(csv_path)
        
        # Add the block count to results
        results["block_counts"].append(block_count)
        
        # Calculate statistics for this block count
        times = df['time_elapsed'].values
        avg_time = times.mean()
        min_time = times.min()
        max_time = times.max()
        std_dev = times.std() if len(times) > 1 else 0
        
        results["avg_times"].append(round(avg_time, 3))
        results["min_times"].append(round(min_time, 3))
        results["max_times"].append(round(max_time, 3))
        results["std_dev_times"].append(round(std_dev, 3))
        
        print(f"Block {block_count}: Avg={avg_time:.3f}s, Min={min_time:.3f}s, Max={max_time:.3f}s")
    
    return results


def visualize_results(results):
    """Create linear visualization of generation time vs. block count."""
    block_counts = results["block_counts"]
    avg_times = results["avg_times"]
    min_times = results["min_times"]
    max_times = results["max_times"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines with distinct colors and markers
    plt.plot(block_counts, min_times, 'o-', color='green', label='Min Generation Time')
    plt.plot(block_counts, avg_times, 's-', color='blue', label='Avg Generation Time')
    plt.plot(block_counts, max_times, '^-', color='red', label='Max Generation Time')
    
    # Calculate trend line for average times
    if len(block_counts) >= 2:
        z = np.polyfit(block_counts, avg_times, 1)
        p = np.poly1d(z)
        plt.plot(block_counts, p(block_counts), "k--", label=f'Trend: {z[0]:.3f}x+{z[1]:.2f}')
    
    plt.title('MetaDrive Map Generation Performance Benchmark')
    plt.xlabel('Number of Blocks')
    plt.ylabel('Generation Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Set x-axis ticks to match block counts exactly
    plt.xticks(block_counts)
    
    return plt


def visualize_results_log_scale(results):
    """Create logarithmic visualization of generation time vs. block count."""
    block_counts = results["block_counts"]
    avg_times = results["avg_times"]
    min_times = results["min_times"]
    max_times = results["max_times"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines with distinct colors and markers
    plt.plot(block_counts, min_times, 'o-', color='green', label='Min Generation Time')
    plt.plot(block_counts, avg_times, 's-', color='blue', label='Avg Generation Time')
    plt.plot(block_counts, max_times, '^-', color='red', label='Max Generation Time')
    
    plt.title('MetaDrive Map Generation Performance Benchmark (Log Scale)')
    plt.xlabel('Number of Blocks')
    plt.ylabel('Generation Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.yscale('log')  # Set y-axis to logarithmic scale
    
    plt.xticks(block_counts)
    
    return plt


def save_results(results):
    """Save benchmark results to files."""
    # Extract timestamp for unique filenames
    timestamp = results["metadata"]["timestamp"]
    
    # Save raw data as JSON in results directory
    json_path = RESULTS_DIR / f"benchmark_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save linear scale plot
    plt_linear = visualize_results(results)
    plt_path_linear = LINEAR_SCALE_DIR / f"benchmark_linear_scale_{timestamp}.png"
    plt_linear.savefig(plt_path_linear, dpi=300, bbox_inches='tight')
    plt_linear.close()
    
    # Save log scale plot
    plt_log = visualize_results_log_scale(results)
    plt_path_log = LOG_SCALE_DIR / f"benchmark_log_scale_{timestamp}.png"
    plt_log.savefig(plt_path_log, dpi=300, bbox_inches='tight')
    plt_log.close()
    
    print(f"Linear plot saved to: {plt_path_linear}")
    print(f"Log-scale plot saved to: {plt_path_log}")
    print(f"JSON data saved to: {json_path}")


def main():
    """Main function to run the benchmark."""
    print(f"MetaDrive Block Scale Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create all necessary directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LINEAR_SCALE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_SCALE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ask if user wants to run benchmarks or just generate plots
    run_new_benchmarks = input("Run new benchmarks? (y/n, default: n): ").strip().lower() == 'y'
    
    if run_new_benchmarks:
        # Build Docker image
        build_docker_image()
        
        # Run benchmarks for each block count
        print(f"Running benchmarks for blocks {BLOCK_RANGE.start}-{BLOCK_RANGE.stop-1} (step {BLOCK_RANGE.step})...")
        for block_count in BLOCK_RANGE:
            print(f"Block count: {block_count}")
            run_benchmark(block_count)
    
    # Process results
    print("Reading CSV data and generating plots...")
    results = read_csv_results()
    
    # Save aggregated results
    save_results(results)
    
    print(f"Process complete. Results saved to {PROJECT_DIR}")

if __name__ == "__main__":
    main() 
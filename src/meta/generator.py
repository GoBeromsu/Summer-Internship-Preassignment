import time
import json
import statistics
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.meta.utils import make_env_config, save_map_outputs


def generate_single_map(
    args,
    output_dir="outputs",
    output_type="png",
):
    """Generate a single map based on provided arguments
    
    Args:
        args: Command line arguments
        output_dir: Base output directory
        output_type: Output format ("png" or "gif")
        
    Returns:
        Generated map metrics
    """
    # Create hierarchical directory structure
    output_dir = Path(output_dir) / "single" / f"blocks{args.map}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the next available index
    existing_files = list(output_dir.glob("metrics_*.json"))
    if existing_files:
        # Extract indices from filenames and find max
        indices = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
        file_idx = max(indices) + 1 if indices else 0
    else:
        file_idx = 0
    
    # Create environment with explicit seed
    config = make_env_config(args)
    env = MetaDriveEnv(config)
    
    metrics = generate_random_map(
        env=env,
        seed=args.seed,
        file_idx=file_idx,
        map_blocks=args.map,
        output_dir=output_dir,
        output_type=output_type
    )
    
    # Clean up environment
    env.close()
    
    return metrics


def generate_maps_benchmark(
    args,
    iterations=100,
    output_dir="outputs",
    output_type="png",
):
    """Run multiple iterations to benchmark MetaDrive map generation performance
    
    Args:
        args: Command line arguments
        iterations: Number of maps to generate
        output_dir: Base output directory
        output_type: Output format ("png" or "gif")
        
    Returns:
        Benchmark summary with statistics
    """
    # Create hierarchical directory structure (without seed in path)
    benchmark_dir = Path(output_dir) / "benchmark" / f"blocks{args.map}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original seed value
    base_seed = args.seed
    
    # Track metrics across all iterations
    all_metrics = []
    generation_times = []
    
    print(f"Running benchmark with {iterations} iterations...")
    print(f"Map blocks: {args.map}, Base seed: {base_seed}")
    
    for i in range(iterations):
        # Calculate seed for this iteration
        current_seed = base_seed + i
        file_idx = i
        
        # Create environment with specific seed (without modifying args)
        config = make_env_config(args, seed=current_seed)
        env = MetaDriveEnv(config)
        
        try:
            metrics = generate_random_map(
                env=env,
                seed=current_seed,
                file_idx=file_idx,
                map_blocks=args.map,
                output_dir=benchmark_dir,
                output_type=output_type
            )
            
            all_metrics.append(metrics)
            generation_times.append(metrics["time_elapsed"])
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{iterations} iterations completed")
        
        except Exception as e:
            print(f"Error in iteration {i} (seed {current_seed}): {e}")
        
        finally:
            env.close()
    
    # Calculate summary statistics
    if generation_times:
        avg_time = statistics.mean(generation_times)
        min_time = min(generation_times)
        max_time = max(generation_times)
    else:
        avg_time = min_time = max_time = 0
    
    # Create summary data
    summary = {
        "benchmark_config": {
            "map_blocks": args.map,
            "base_seed": base_seed,
            "iterations": iterations,
        },
        "performance": {
            "avg_generation_time": round(avg_time, 3),
            "min_generation_time": round(min_time, 3),
            "max_generation_time": round(max_time, 3),
        },
        "individual_metrics": all_metrics
    }
    
    # Save summary
    summary_path = benchmark_dir / "benchmark.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBenchmark complete. Results saved to {benchmark_dir}")
    print(f"Average generation time: {avg_time:.3f}s")
    print(f"Min/Max times: {min_time:.3f}s / {max_time:.3f}s")
    
    return summary


def generate_random_map(
    env,
    seed,
    file_idx,
    map_blocks,
    output_dir="outputs",
    output_type="png"
):
    """Generate a random map with specific parameters
    
    Args:
        env: MetaDrive environment
        seed: Random seed for generation
        file_idx: Index for output filename
        map_blocks: Number of map blocks
        output_dir: Output directory
        output_type: Output format ("png" or "gif")
        
    Returns:
        Map generation metrics
    """
    output_dir = Path(output_dir)
    
    # Start timing
    t0 = time.time()
    
    # Reset the environment with the explicit seed
    env.reset(seed=seed)
    
    # Initialize metrics
    metrics = {
        "index": file_idx,
        "seed": seed,
        "map_blocks": map_blocks
    }
    
    # Save outputs using helper function
    try:
        save_map_outputs(
            env=env,
            output_dir=output_dir,
            file_idx=file_idx,
            output_type=output_type
        )
    except Exception as e:
        print(f"Warning: Failed to save map outputs (seed {seed}): {e}")
    
    # End timing
    t1 = time.time()
    
    # Add summary metrics
    metrics.update({
        "time_elapsed": round(t1 - t0, 3)
    })

    # Save JSON metrics
    metrics_path = output_dir / f"metrics_{file_idx:04d}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[✓] Generated map {file_idx}: seed={seed}, blocks={map_blocks}, time={metrics['time_elapsed']}s")
    
    return metrics

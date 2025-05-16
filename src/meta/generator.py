import time
import statistics
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.meta.utils import make_env_config, save_map, save_json, find_next_index


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
    
    file_idx = find_next_index(output_dir)
    
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
    save_json(output_dir / f"metrics_{file_idx:04d}", metrics)

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

        
        env.close()

    avg_time = statistics.mean(generation_times)
    min_time = min(generation_times)
    max_time = max(generation_times)

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
    
    # Save summary using the utility function
    save_json(benchmark_dir / "benchmark", summary)
    
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

    output_dir = Path(output_dir)
    
    t0 = time.time()
    
    # Reset the environment with the explicit seed
    env.reset(seed=seed)
    
    # Initialize metrics
    metrics = {
        "index": file_idx,
        "seed": seed,
        "map_blocks": map_blocks
    }
    

    save_map(
        env=env,
        output_dir=output_dir,
        file_idx=file_idx,
        output_type=output_type
    )
    
    t1 = time.time()
    metrics.update({"time_elapsed": round(t1 - t0, 3)})
    print(f"[✓] Generated map {file_idx}: seed={seed}, blocks={map_blocks}, time={metrics['time_elapsed']}s")
    
    return metrics

import time
import statistics
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from meta.utils import make_env_config, save_map, save_json, save_metrics_to_csv, create_iso_timestamp
from collections import Counter

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
    # Use output directory directly without adding blocks subdirectory
    # Create environment with explicit seed
    config = make_env_config(args)
    env = MetaDriveEnv(config)
    
    metrics = generate_random_map(
        env=env,
        seed=args.seed,
        map_blocks=args.map,
        output_dir=output_dir,
        output_type=output_type
    )
    
    # Clean up environment
    env.close()
    save_json(output_dir / f"metrics_{create_iso_timestamp(False)}", metrics)

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
        
        # Create environment with specific seed (without modifying args)
        config = make_env_config(args, seed=current_seed)
        env = MetaDriveEnv(config)
        
        metrics = generate_random_map(
            env=env,
            seed=current_seed,
            map_blocks=args.map,
            output_dir=output_dir,
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
    
    # Save summary using the utility function with a timestamp for uniqueness
    timestamp = create_iso_timestamp(False)
    save_json(output_dir / f"benchmark_{timestamp}", summary)
    
    print(f"\nBenchmark complete. Results saved to {output_dir}")
    print(f"Average generation time: {avg_time:.3f}s")
    print(f"Min/Max times: {min_time:.3f}s / {max_time:.3f}s")
    
    return summary


def generate_random_map(
        env,
        seed,
        map_blocks,
        output_dir="outputs",
        output_type="png"
    ):

    output_dir = Path(output_dir)
    
    t0 = time.time()
    env.reset(seed=seed)
    t1 = time.time()
    
    block_sequence = [block.ID for block in env.current_map.blocks]
    block_type_counts = dict(Counter(block_sequence))
    
    # Save the map and get the timestamp information
    idx, ts = save_map(
        env=env,
        output_dir=output_dir,
        output_type=output_type
    )
    
    metrics = {
        "filename": f"map_{idx}_{ts}.png",
        "seed": seed,
        "map_blocks": map_blocks,
        "block_sequence": block_sequence,
        "block_type_counts": block_type_counts,
        "time_elapsed": round(t1 - t0, 3),
        "idx": idx,
        "timestamp": ts
    }
    
    # Save metrics to CSV
    save_metrics_to_csv(output_dir, metrics)
    
    print(f"[✓] Generated map: seed={seed}, blocks={map_blocks}, time={metrics['time_elapsed']}s")
    
    return metrics

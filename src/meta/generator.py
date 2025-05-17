import time
import statistics
from pathlib import Path
import concurrent.futures
from metadrive.envs.metadrive_env import MetaDriveEnv
from meta.utils import make_env_config, save_map, save_json, save_metrics_to_csv, create_iso_timestamp, ensure_dir_exists
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
    
    # Save full metrics data
    data_dir = Path(output_dir) / "data"
    save_json(data_dir / f"metrics_{create_iso_timestamp(False)}", metrics)

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
    success_count = 0
    failure_count = 0
    
    print(f"Running benchmark with {iterations} iterations...")
    print(f"Map blocks: {args.map}, Base seed: {base_seed}")
    
    for i in range(iterations):
        # Calculate seed for this iteration
        current_seed = base_seed + i
        
        # Create environment with specific seed
        config = make_env_config(args, seed=current_seed)
        env = MetaDriveEnv(config)
        
        # Print progress at the start of each iteration
        print(f"[{i+1:3d}/{iterations}] Trying seed={current_seed}, blocks={args.map} ...")
        
        metrics = generate_random_map(
            env=env,
            seed=current_seed,
            map_blocks=args.map,
            output_dir=output_dir,
            output_type=output_type
        )
            
        all_metrics.append(metrics)
        
        # Only include successful generations in time statistics
        if "error" not in metrics:
            generation_times.append(metrics["time_elapsed"])
            success_count += 1
        else:
            failure_count += 1
            
        env.close()

    # Calculate statistics only if we have successful generations
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
            "success_count": success_count,
            "failure_count": failure_count
        },
        "individual_metrics": all_metrics
    }
    
    # Save summary using the utility function with a timestamp for uniqueness
    data_dir = Path(output_dir) / "data"
    ensure_dir_exists(data_dir)
    timestamp = create_iso_timestamp(False)
    save_json(data_dir / f"benchmark_{timestamp}", summary)
    
    print(f"\nBenchmark complete.  ✔ {success_count}/{iterations} successful,  ❌ {failure_count} failed")
    if generation_times:
        print(f"Average generation time: {avg_time:.3f}s  (on {success_count} successful runs)")
        print(f"Min/Max times: {min_time:.3f}s / {max_time:.3f}s")
    else:
        print("No successful maps generated.")
    
    return summary


def safe_reset(env, seed, timeout=120):
    """Reset environment with timeout protection
    
    Retry added because BIG discards invalid blocks up to T times, then backtracks if all fail.
    This may cause performance issues, appearing as if the process is stuck or stopped (Li et al., 2021, p. 7).
    Li, Q., et al. (2021). MetaDrive: Composing diverse driving scenarios for generalizable reinforcement learning. *arXiv:2109.12674*

    Args:
        env: The MetaDrive environment
        seed: Random seed for reproducibility
        timeout: Maximum time in seconds to wait for reset
        
    Raises:
        TimeoutError: If the reset operation times out or fails
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(env.reset, seed=seed)
            future.result(timeout=timeout)
    except Exception as e:
        raise TimeoutError(f"[✗] env.reset(seed={seed}) failed: {type(e).__name__}") from e


def generate_random_map(
        env,
        seed,
        map_blocks,
        output_dir="outputs",
        images_dir=None,
        data_dir=None,
        output_type="png"
    ):

    output_dir = Path(output_dir)
    
    # Set up directory structure consistently 
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    ensure_dir_exists(images_dir)
    ensure_dir_exists(data_dir)
    
    t0 = time.time()
    
    try:
        safe_reset(env, seed)
        t1 = time.time()
    except TimeoutError as e:
        # Return metrics with error information
        return {
            "seed": seed,
            "map_blocks": map_blocks,
            "error": str(e),
            "time_elapsed": None,
            "idx": None,
            "timestamp": create_iso_timestamp(False)
        }
    
    # Continue with successful map generation
    block_sequence = [block.ID for block in env.current_map.blocks]
    block_type_counts = dict(Counter(block_sequence))
    
    # Save the map and get the timestamp information
    idx, ts = save_map(
        env=env,
        output_dir=images_dir,
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
    
    # Save metrics to CSV in data directory
    save_metrics_to_csv(data_dir, metrics)
    
    print(f"[✓] Generated: seed={seed}, time={metrics['time_elapsed']}s")
    
    return metrics

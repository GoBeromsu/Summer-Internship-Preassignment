import time
import json
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt
import statistics


def make_env_config(args):
    """Create MetaDrive configuration using MetaDrive's built-in config system"""
    config = {
        # Map generation parameters
        "map": args.map,  # Number of blocks in the map
        "num_scenarios": args.num_scenarios,
        "start_seed": args.seed,  # Starting seed for random generation
        
        # Rendering options
        "use_render": False,  # Disable screen rendering in headless environment

        # Logging options
        "log_level": 50,  # Minimal logging
        
    }
    return config


def generate_single_map(
    args,
    output_dir="outputs",
    output_type="png",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the next available index
    existing_files = list(output_dir.glob("metrics_*.json"))
    if existing_files:
        # Extract indices from filenames and find max
        indices = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
        file_idx = max(indices) + 1 if indices else 0
    else:
        file_idx = 0
    
    config = make_env_config(args)
    env = MetaDriveEnv(config)
    
    # Generate map with MetaDrive - it will use the num_scenarios setting internally
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
    """Run multiple iterations to benchmark MetaDrive map generation performance"""
    base_output_dir = Path(output_dir)
    
    # Create a specific directory for this benchmark
    benchmark_dir = base_output_dir / f"benchmark_blocks{args.map}_seed{args.seed}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Track metrics across all iterations
    all_metrics = []
    generation_times = []
    
    print(f"Running benchmark with {iterations} iterations...")
    print(f"Map blocks: {args.map}, Seed base: {args.seed}")
    
    for i in range(iterations):
        current_seed = args.seed + i
        file_idx = i
        
        args.seed = current_seed
        args.num_scenarios = 1

        config = make_env_config(args)

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
            print(f"Error in iteration {i}: {e}")
        
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
            "base_seed": args.seed,
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
    summary_path = benchmark_dir / "benchmark_summary.json"
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
    output_dir = Path(output_dir)
    
    # Start timing
    t0 = time.time()
    
    # Reset the environment with the new seed
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
        print(f"Warning: Failed to save map outputs: {e}")
    
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


def save_map_outputs(
    env,
    output_dir,
    file_idx,
    output_type="png"
):
    """Save map visualization to the specified output format"""
    output_dir = Path(output_dir)
    
    if output_type.lower() == "png":
        map_path = output_dir / f"map_{file_idx:04d}.png"
        img = draw_top_down_map(env.current_map)
        plt.imsave(str(map_path), img)
        plt.close()
    elif output_type.lower() == "gif":
        # Create GIF of the simulated episode
        gif_path = output_dir / f"map_{file_idx:04d}.gif"
        env.render_to_gif(str(gif_path), audio=False, fps=10)
    
    return True

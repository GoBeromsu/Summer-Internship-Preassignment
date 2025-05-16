import time
import json
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt


def make_env_config(args):
    """Create MetaDrive configuration using MetaDrive's built-in config system"""
    config = {
        # Map generation parameters
        "map": args.map,  # Number of blocks in the map
        "num_scenarios": args.num_scenarios,  # Number of different scenarios to generate
        "start_seed": args.seed,  # Starting seed for random generation
        
        # Rendering options
        "use_render": False,  # Disable screen rendering in headless environment

        # Logging options
        "log_level": 50,  # Minimal logging
        
    }
    return config


def generate_maps_batch(
    args,
    output_dir="outputs",
    output_type="png",
    iterations=1
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the next available index
    existing_files = list(output_dir.glob("metrics_*.json"))
    if existing_files:
        # Extract indices from filenames and find max
        indices = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
        start_index = max(indices) + 1 if indices else 0
    else:
        start_index = 0
    
    # Create environment just once
    config = make_env_config(args)
    env = MetaDriveEnv(config)
    
    # Generate maps for each iteration
    for i in range(iterations):
        current_seed = args.seed + i
        file_idx = start_index + i
        
        # Generate a single map with the current seed
        generate_random_map(
            args,
            env=env,
            seed=current_seed,
            file_idx=file_idx,
            map_blocks=args.map,
            output_dir=output_dir,
            output_type=output_type
        )

def generate_random_map(
    args,
    env,
    seed,
    file_idx,
    map_blocks,
    output_dir="outputs",
    output_type="png"
):
    output_dir = Path(output_dir)
    
    # Reset the environment with the new seed instead of creating a new one
    env.reset(seed=seed)
    
    # Initialize metrics
    metrics = {
        "index": file_idx,
        "seed": seed,
        "map_blocks": map_blocks
    }
    
    t0 = time.time()
    t1 = time.time()
    
    # Add summary metrics
    metrics.update({
        "time_elapsed": round(t1 - t0, 3)
    })

    # Save JSON metrics
    metrics_path = output_dir / f"metrics_{file_idx:04d}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save visual output based on selected type
    if output_type.lower() == "png":
        map_path = output_dir / f"map_{file_idx:04d}.png"
        try:
            img = draw_top_down_map(env.current_map)
    
            plt.imsave(str(map_path), img)
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save map image: {e}")
    elif output_type.lower() == "gif":
        try:
            # Create GIF of the simulated episode
            gif_path = output_dir / f"map_{file_idx:04d}.gif"
            env.render_to_gif(str(gif_path), audio=False, fps=10)
        except Exception as e:
            print(f"Warning: Failed to save GIF: {e}")

    print(f"[✓] Generated map {file_idx}: seed={seed}, blocks={map_blocks}, time={metrics['time_elapsed']}s")
    
    return metrics

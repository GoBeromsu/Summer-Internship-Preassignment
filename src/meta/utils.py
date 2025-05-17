from pathlib import Path
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt
import json
import datetime


def create_timestamp():
    """Create a timestamp string in format YYYYMMDD_HHMMSS for unique folder names"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir_exists(directory):
    """Ensure directory exists, create if not
    
    Args:
        directory: Directory path to check/create
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def make_env_config(args, seed=None):
    """Create MetaDrive configuration using MetaDrive's built-in config system
    
    Args:
        args: Command line arguments
        seed: Optional override for the seed value (takes precedence over args.seed)
    """
    config = {
        # Map generation parameters
        "map": args.map,  # Number of blocks in the map
        "num_scenarios": args.num_scenarios,
        "start_seed": seed if seed is not None else args.seed,  # Starting seed for random generation
        
        # Rendering options
        "use_render": False,  # Disable screen rendering in headless environment

        # Logging options
        "log_level": 50,  # Minimal logging
        
    }
    return config


def save_map(
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


def save_json(filepath, content, indent=2, overwrite=True):
    path = Path(filepath)
    path = path.with_suffix('.json')
    
    if not overwrite and path.exists():
        print(f"[!] Skipping existing file: {path}")
        return path
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w") as f:
            json.dump(content, f, indent=indent)
        print(f"[✓] JSON saved: {path}")
    except Exception as e:
        print(f"[!] Error writing JSON to {path}: {e}")
        raise
    
    return path


def find_next_index(directory, pattern="metrics_*.json"):
    """Find the next available index for numbered files in a directory
    
    Args:
        directory (str or Path): Directory to search in
        pattern (str): Glob pattern for files with index in the second part of stem
                       (e.g., "metrics_0001.json" where "0001" is the index)
    
    Returns:
        int: Next available index (0 if no existing files found)
    """
    directory = Path(directory)
    existing_files = list(directory.glob(pattern))
    
    if not existing_files:
        return 0
    
    # Extract indices from filenames and find max
    indices = [
        int(f.stem.split('_')[1]) 
        for f in existing_files 
        if len(f.stem.split('_')) > 1 and f.stem.split('_')[1].isdigit()
    ]
    
    return max(indices) + 1 if indices else 0 
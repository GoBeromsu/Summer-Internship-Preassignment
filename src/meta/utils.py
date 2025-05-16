from pathlib import Path
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt


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
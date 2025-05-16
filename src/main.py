import argparse
import sys
from pathlib import Path
from src.meta import generate_single_map, generate_maps_benchmark
import subprocess

def setup_metadrive():
    metadrive_path = Path("metadrive")
    if not metadrive_path.exists():
        print("MetaDrive not found. Installing...")
        subprocess.run(
            ["git", "clone", "https://github.com/metadriverse/metadrive.git", "--single-branch"],
            check=True
        )
        subprocess.run(["pip", "install", "-e", "./metadrive"], check=True)
        subprocess.run(["python", "-m", "metadrive.pull_asset"], check=True)
        print("MetaDrive installed successfully!")
    else:
        print("MetaDrive already exists.")


def parse_args():
    parser = argparse.ArgumentParser("MetaDrive Scenario Generator")
    
    # Required parameters
    parser.add_argument("--map", type=int, required=True, help="Number of blocks in each generated map")
    
    # Optional parameters
    parser.add_argument("--num-scenarios", type=int, default=1, 
                        help="Number of different scenarios to generate")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Starting seed for random generation")
    parser.add_argument("--output-type", type=str, default="png", choices=["png", "gif"], 
                        help="Visual output format (JSON metrics are always saved)")
    parser.add_argument("--output-dir", type=str, default="outputs", 
                        help="Directory to save all outputs")
    parser.add_argument("--render-mode", action="store_true", 
                        help="Render on screen (not recommended in Docker)")
    parser.add_argument("--benchmark", type=int, default=100,
                        help="Number of iterations to run in benchmark mode")
    
    return parser.parse_args()


def main():
    if not Path("metadrive").exists():
        setup_metadrive()

    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.benchmark:
        print(f"Running benchmark with {args.benchmark} iterations...")
        print(f"Map blocks: {args.map}, Starting seed: {args.seed}")
        print(f"Output format: {args.output_type} + JSON metrics")
        print(f"Output directory: {output_dir.absolute()}")
        
        # Run benchmark
        summary = generate_maps_benchmark(
            args=args,
            iterations=args.benchmark,
            output_dir=args.output_dir,
            output_type=args.output_type,
        )
        
        print(f"\nBenchmark complete. Average time: {summary['performance']['avg_generation_time']}s")
        
    else:
        print(f"Generating {args.num_scenarios} scenarios with {args.map} blocks each")
        print(f"Starting seed: {args.seed}")
        print(f"Output format: {args.output_type} + JSON metrics")
        print(f"Output directory: {output_dir.absolute()}")
        
        # Generate maps with the refactored function
        generate_single_map(
            args=args,
            output_dir=args.output_dir,
            output_type=args.output_type,
        )
        
        print(f"\nScenario generation complete. Results saved to {output_dir.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
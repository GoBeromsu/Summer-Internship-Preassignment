# My Metadrive Docker Commands

This document provides a reference for running the MetaDrive container with various command-line arguments.

## Basic Usage

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" metadrive [OPTIONS]
```

## Required Parameters

| Parameter        | Description                            |
| ---------------- | -------------------------------------- |
| `--map <number>` | Number of blocks in each generated map |

## Optional Parameters

| Parameter                  | Description                                          | Default         |
| -------------------------- | ---------------------------------------------------- | --------------- |
| `--num-scenarios <number>` | Number of different scenarios to generate            | 1               |
| `--seed <number>`          | Starting seed for random generation                  | 0               |
| `--output-type <png/gif>`  | Visual output format (JSON metrics are always saved) | png             |
| `--output-dir <path>`      | Base directory for outputs                           | outputs         |
| `--project-name <name>`    | Project name for organizing outputs                  | timestamp-based |
| `--render-mode`            | Render on screen (not recommended in Docker)         | false           |
| `--benchmark <number>`     | Number of iterations to run in benchmark mode        | 100             |

## Example Commands

### Generate a single map with 10 blocks

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" metadrive --map 10
```

### Run benchmark with 20 blocks and 3 iterations

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" metadrive --map 20 --benchmark 3 --project-name blockscale_benchmark
```

### Generate map with custom seed and output type

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" metadrive --map 15 --seed 42 --output-type gif
```

### Generate multiple scenarios

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" metadrive --map 10 --num-scenarios 5
```

## Notes

- The Docker container mounts the local `outputs` directory to `/app/outputs` inside the container
- Results will be organized in subdirectories based on the project name and block count
- For benchmark runs, performance metrics are saved in CSV and JSON formats

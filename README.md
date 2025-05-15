# Summer Internship pre-assignment
## Problem Framing
- **Purpose**: _Test-scenario generator prototype_
	- Prototype to generate scenario-based maps for ADS testing
- **Scope**: Generating maps
- **Perspective**: Framed as an autonomous vehicle testing task
	- A map is treated as a test scenario
	- Generating diverse maps helps evaluate system behaviour under varying driving conditions
- **Why use MetaDrive?**
	- MetaDrive is selected due to its stability and non-flaky nature, which aligns with both the pre-assignment and the anticipated internship tasks
	- MetaDrive enables deterministic map generation using seed control
	- “31.3% of benchmark test scenarios are potentially flaky due to nondeterministic simulations in CARLA, whereas MetaDrive does not yield any flaky tests” (Osikowicz et al., 2025)

#### What Does 'Map' Mean in ADS?
- According to Zhong et al. (2021), a map corresponds to the **L1 layer’s geometry**
	- Road shape, topology, and surface conditions must be varied to evaluate ADS performance under diverse scenarios

## About Map
### What Makes a Good Map?
- A well-designed map should generate **diverse, challenging, and reproducible** driving scenarios to **stress-test** the learning or control capacity of the ADS (Li et al., 2021)
- **Structural Complexity**
	- Measured through topological diversity and spatial challenge

### How Will the Map Be Generated?
- Procedurally generate maps using MetaDrive same as Osikowicz et al. (2025)
- Map MetaDrive configurations to L1 geometry metrics defined by Zhong et al. (2021)

### Evaluation Criteria:
- Generation time
- Topology-based structural complexity
  - Define a complexity metric
  - Use this as the basis for benchmarking
  - In block-based simulators like MetaDrive, **road connectivity** is the key factor affecting map complexity
  - Formally define map complexity metrics and empirically investigate their correlation with **scenario diversity** and **ADS failure exposure**

## Criteria
- MetaDrive metrics aligned with L1 geometry metrics defined by Zhong et al. (2021)

### Generation Runtime
- `generation_time`: Time taken to generate a single map
	- Measures the **scalability and stability** of the map generator
	- Shorter generation times are helpful but do not necessarily indicate the quality of a test scenario

### Metric-to-MetaDrive Mapping

| Metric                     | MetaDrive Extraction Method                                                               | Notes                                                                 |
|----------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `complexity_score`         | `env.engine.map.road_network.graph` → weighted by block count and type                    | Can parse `map_config["config"]` or iterate over `env.engine.block_sequence` |
| `avg_num_spawn_points`     | `env.engine.spawn_manager.spawn_points` → `len()` and statistical aggregation             | `spawn_points` is a `List[SpawnPoint]`                               |
| `min/max curvature`        | Extract `curvature` or `angle` from blocks of type `"r"` or `"Y"`                         | Requires custom loop over `env.engine.block_sequence` using `get_angle()` |
| `graph_diameter`           | Convert `road_network.graph` to NetworkX and compute diameter                             | MetaDrive represents road networks as edge-based implicit graphs      |
| `trigger object count`     | Filter `env.engine.object_manager.objects` for `TrafficCone`, `Barrier`, `CrashObject`    | Filter using `type(obj).__name__`                                    |

- MetaDrive’s block-based architecture allows easy extraction of road-level metrics
- Most metrics can be retrieved via `env.engine.block_sequence`, `spawn_manager`, `road_network.graph`, or `object_manager`

## References
- Zhong, Z., Tang, Y., Zhou, Y., Neves, V. de O., Liu, Y., & Ray, B. (2021). _A Survey on Scenario-Based Testing for Automated Driving Systems in High-Fidelity Simulation_ (No. arXiv:2112.00964). arXiv. [https://doi.org/10.48550/arXiv.2112.00964](https://doi.org/10.48550/arXiv.2112.00964)

- Osikowicz, O., McMinn, P., & Shin, D. (2025). _Empirically Evaluating Flaky Tests for Autonomous Driving Systems in Simulated Environments_.

- Li, Q., Peng, Z., Feng, L., Zhang, Q., Xue, Z., & Zhou, B. (2021). _MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning_ (Version 3). arXiv. [https://doi.org/10.48550/ARXIV.2109.12674](https://doi.org/10.48550/ARXIV.2109.12674)

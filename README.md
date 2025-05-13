# MetaDrive Simulation

A simulation tool for generating and analyzing road networks.

## Dependencies

- **numpy**: Core numerical computations and array operations for road generation
- **matplotlib**: Visualization of road networks and generation of map images
- **pillow**: Image processing support for saving visualizations
- **pytest**: Testing framework for unit and integration tests

### MetaDrive
- I used MetaDrive's docker image to run the simulation.
- [Installing MetaDrive — MetaDrive 0.1.1 documentation](https://metadrive-simulator.readthedocs.io/en/latest/install.html) in this documentation, they do not recommend installing MetaDrive with pip install metadrive-simulator, so I did not use it.
## Setup

```bash
# install project dependencies
pip install -e .
```
## Test

```bash
pytest -s tests/test_metadrive.py
```



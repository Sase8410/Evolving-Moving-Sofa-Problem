# AI Moving Sofa Problem

An evolutionary optimization project that explores how complex sofa shapes can move through an L-shaped hallway without colliding with the walls.

The project uses polygon geometry, collision detection, and a genetic algorithm to optimize sofa dimensions and maximize area while following a turning route.

## Technologies

- Python
- NumPy
- Shapely
- Matplotlib

## Features

- Interactive hallway and sofa simulation
- Rectangle, notched, and curved sofa shapes
- Waypoint-based movement
- Collision and containment testing
- Area-based fitness function
- Genetic algorithm optimization
- Interactive visualization and animation

## Run

```bash
pip install numpy shapely matplotlib
python sofa_genetic_algorithm.py
```

The current implementation uses sampled poses and fixed movement routes. Future improvements will include joint optimization of the sofa shape and movement path.

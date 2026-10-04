# AI Moving Sofa Problem — Step 1: Hallway Simulation
**File:** `sofa_simulation.py`

## Objective

Build an interactive 2D simulation that determines whether a rigid sofa fits at a given position and orientation inside an L-shaped hallway.

## Technologies

| Package | Purpose |
| --- | --- |
| NumPy | Store vertex coordinates and perform rotation calculations |
| Shapely | Construct polygons, combine hallway sections, and check containment |
| Matplotlib | Display the simulation and provide interactive controls |

## Implementation

### 1. Hallway geometry

Created an L-shaped hallway by combining two rectangular regions:

- Horizontal corridor: x from −5 to 1, y from 0 to 1.
- Vertical corridor: x from 0 to 1, y from 0 to 5.

Both corridors are one unit wide. The finite ends also act as boundaries.

### 2. Sofa representation

Represented the sofa as a rectangle with four vertices defined relative to its center.

- Width: 1.5 units.
- Height: 0.6 units.
- Area: 0.9 square units.

The polygon representation provides a foundation for introducing more complex shapes later.

### 3. Movement and rotation

Implemented `transform_sofa(x, y, angle_degrees)` to:

1. Convert the rotation angle from degrees to radians.
2. Rotate the vertices around the sofa center using a 2D rotation matrix.
3. Translate the rotated vertices to the requested position.

These transformations preserve the sofa's shape, dimensions, and area.

### 4. Collision detection

Implemented `sofa_fits(sofa)` using:

```python
hallway.covers(sofa)
```

A position is valid when the entire sofa lies inside or on the boundary of the hallway.

The area outside the hallway is calculated using:

```python
sofa.difference(hallway).area
```

### 5. Interactive visualization

Added:

- Sliders for horizontal position, vertical position, and rotation.
- Green coloring for valid positions.
- Red coloring for collisions.
- A marker showing the sofa center.
- A display of sofa area and area outside the hallway.
- A Reset button that restores the starting pose.

## Installation and Execution

Install the dependencies:

```bash
python -m pip install numpy matplotlib shapely
```

Run the simulation:

```bash
python sofa_simulation.py
```

## Suggested Manual Checks

| X position | Y position | Rotation | Expected result |
| --- | --- | --- | --- |
| −2.5 | 0.5 | 0° | Green: inside the horizontal corridor |
| 0.5 | 2.5 | 90° | Green: inside the vertical corridor |
| 0.5 | 2.5 | 0° | Red: extends outside the vertical corridor |

At acceptance, execution and manual-check results had not been reported or independently verified.

## Limitations

- Collision detection evaluates individual poses.
- Moving between two valid poses does not establish that the movement between them is collision-free.
- Sliders allow movement through invalid positions.
- The hallway has finite ends.
- This step does not include AI, shape optimization, or automated path planning.

## Next Step

Define movement routes as sequences of position and rotation waypoints, animate the sofa along those routes, and check intermediate poses for collisions.
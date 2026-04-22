#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_project3_map(clearance_cm: int, map_scale: float = 4.0):
    """
    Builds a 2D occupancy grid for the project 3 arena.

    The real arena is 2m x 4m. We scale it up by map_scale so
    the planner has more room to work with (e.g., 4x -> 8m x 16m).
    Everything—obstacles, walls, blocks—gets scaled the same way.

    Returns:
        grid          – 2D numpy array (0 = free, 1 = obstacle/inflated)
        map_info      – dict with bounds, dimensions, scale factor
        obstacle_shapes – list of obstacle dicts for plotting later
    """

    # Original arena dimensions (meters)
    base_x_max = 2.0
    base_y_max = 4.0

    # After scaling, these are the actual map bounds we'll use
    x_min_m, x_max_m = 0.0, base_x_max * map_scale
    y_min_m, y_max_m = 0.0, base_y_max * map_scale

    # Convert to centimeters for the grid (1 cell = 1 cm)
    width_cm = int(round((x_max_m - x_min_m) * 100.0))
    height_cm = int(round((y_max_m - y_min_m) * 100.0))

    # The occupancy grid — starts all free (zeros)
    grid = np.zeros((height_cm, width_cm), dtype=np.uint8)

    # Shorthand for the clearance we'll inflate obstacles by
    c = clearance_cm

    # We'll collect obstacle shapes here so we can draw them later
    obstacle_shapes = []

    # --- Helper functions ---

    def S(v):
        """Scale a value from base coords to map coords."""
        return v * map_scale

    def world_to_grid(x, y):
        """Convert world coordinates (meters) to grid indices (cm)."""
        return int(round(x * 100.0)), int(round(y * 100.0))

    def add_rect_raw(x1, y1, x2, y2):
        """
        Mark a rectangular region as occupied on the grid.
        This works in already-scaled world coords (meters).
        Inflates the rectangle by the clearance on all sides.
        """
        gx1, gy1 = world_to_grid(x1, y1)
        gx2, gy2 = world_to_grid(x2, y2)

        # Sort so min/max are correct regardless of corner order
        xmin, xmax = sorted([gx1, gx2])
        ymin, ymax = sorted([gy1, gy2])

        # Fill the grid, padded by clearance c on each side
        # Clamp to grid bounds so we don't go out of range
        grid[max(0, ymin - c):min(height_cm, ymax + c),
             max(0, xmin - c):min(width_cm, xmax + c)] = 1

    def add_rect(x1, y1, x2, y2, label=None):
        """
        Add a rectangle in BASE (unscaled) coordinates.
        Scales it up, marks the grid, and saves it for plotting.
        """
        x1s, y1s, x2s, y2s = S(x1), S(y1), S(x2), S(y2)
        add_rect_raw(x1s, y1s, x2s, y2s)

        obstacle_shapes.append({
            "type": "rect",
            "x_min": min(x1s, x2s),
            "x_max": max(x1s, x2s),
            "y_min": min(y1s, y2s),
            "y_max": max(y1s, y2s),
            "label": label,
        })

    def add_thick_line(x1, y1, length, angle_deg, thickness=0.05, label=None):
        """
        Draw an angled wall as a thick line.

        We approximate the line by stamping small rectangles along it
        (like dragging a square brush). Not the most elegant approach,
        but it handles arbitrary angles without needing polygon rasterization.
        """
        # Scale everything to map coords
        x1s = S(x1)
        y1s = S(y1)
        length_s = S(length)
        thickness_s = S(thickness)

        # Compute the endpoint from start + angle + length
        theta = math.radians(angle_deg)
        x2s = x1s + length_s * math.cos(theta)
        y2s = y1s + length_s * math.sin(theta)

        # Walk along the line and stamp small squares at each step
        steps = max(120, int(120 * map_scale))
        for i in range(steps + 1):
            t = i / steps
            x = x1s + t * (x2s - x1s)
            y = y1s + t * (y2s - y1s)
            add_rect_raw(
                x - thickness_s / 2.0,
                y - thickness_s / 2.0,
                x + thickness_s / 2.0,
                y + thickness_s / 2.0
            )

        # Save for plotting
        obstacle_shapes.append({
            "type": "line",
            "x1": x1s,
            "y1": y1s,
            "x2": x2s,
            "y2": y2s,
            "thickness": thickness_s,
            "label": label,
        })

    # =========================================================
    # OUTER WALLS — thin rectangles along the arena boundary
    # =========================================================
    add_rect(0.0, 0.0, 2.0, 0.05, "bottom wall")
    add_rect(0.0, 3.95, 2.0, 4.0, "top wall")
    add_rect(0.0, 0.0, 0.05, 4.0, "left wall")
    add_rect(1.95, 0.0, 2.0, 4.0, "right wall")

    # =========================================================
    # 3 SQUARE BLOCKS — centered at specific positions
    # Each block is 0.30 m x 0.30 m (before scaling)
    # =========================================================
    block_size = 0.30
    half = block_size / 2.0

    add_rect(1.55 - half, 0.42 - half, 1.55 + half, 0.42 + half, "block_1")
    add_rect(0.50 - half, 1.50 - half, 0.50 + half, 1.50 + half, "block_2")
    add_rect(0.30 - half, 2.70 - half, 0.30 + half, 2.70 + half, "block_3")

    # =========================================================
    # INTERNAL WALLS — angled lines inside the arena
    # These partition the space and force the planner to find gaps
    # =========================================================

    # Diagonal wall going up-right at 30 degrees from lower-left area
    add_thick_line(
        x1=0.10, y1=0.40,
        length=1.41, angle_deg=30,
        thickness=0.05, label="30 deg wall"
    )

    # Wall angled at 155 degrees, starting from the right side mid-arena
    add_thick_line(
        x1=1.9, y1=1.40,
        length=1.25, angle_deg=155,
        thickness=0.05, label="155 deg wall"
    )

    # Horizontal wall near the top (0 degrees = straight right)
    add_thick_line(
        x1=0.1, y1=2.92,
        length=1.45, angle_deg=0,
        thickness=0.05, label="0 deg wall"
    )

    # Pack up map metadata for use by the planner and visualization
    map_info = {
        "x_min_m": x_min_m,
        "x_max_m": x_max_m,
        "y_min_m": y_min_m,
        "y_max_m": y_max_m,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "map_scale": map_scale,
    }

    return grid, map_info, obstacle_shapes


def world_m_to_grid_cm(x_m: float, y_m: float, map_info: dict):
    """Convert a world position (meters) to grid position (cm)."""
    x_cm = (x_m - map_info["x_min_m"]) * 100.0
    y_cm = (y_m - map_info["y_min_m"]) * 100.0
    return x_cm, y_cm


def is_free(grid, x_cm, y_cm):
    """
    Check if a point (in cm coords) is free on the grid.
    Returns False if out of bounds or inside an obstacle.
    """
    xi = int(round(x_cm))
    yi = int(round(y_cm))
    if xi < 0 or yi < 0 or yi >= grid.shape[0] or xi >= grid.shape[1]:
        return False
    return grid[yi, xi] == 0


def print_obstacle_coordinates(obstacle_shapes):
    """Dump all obstacle positions to the console — handy for debugging."""
    print("\nScaled obstacle coordinates:")
    for obs in obstacle_shapes:
        if obs["type"] == "rect":
            cx = 0.5 * (obs["x_min"] + obs["x_max"])
            cy = 0.5 * (obs["y_min"] + obs["y_max"])
            print(
                f"  {obs['label']}: "
                f"x=[{obs['x_min']:.3f}, {obs['x_max']:.3f}], "
                f"y=[{obs['y_min']:.3f}, {obs['y_max']:.3f}], "
                f"center=({cx:.3f}, {cy:.3f})"
            )
        elif obs["type"] == "line":
            print(
                f"  {obs['label']}: "
                f"start=({obs['x1']:.3f}, {obs['y1']:.3f}), "
                f"end=({obs['x2']:.3f}, {obs['y2']:.3f}), "
                f"thickness={obs['thickness']:.3f}"
            )


def plot_map(grid, map_info, obstacle_shapes, start=None, goal=None):
    """
    Render the occupancy grid with obstacle outlines, labels,
    and optional start/goal markers. Mostly for sanity-checking
    that the map looks right before running the planner.
    """
    fig, ax = plt.subplots(figsize=(8, 14))

    # Show the grid as a grayscale image (dark = obstacle)
    extent = [
        map_info["x_min_m"],
        map_info["x_max_m"],
        map_info["y_min_m"],
        map_info["y_max_m"],
    ]

    ax.imshow(
        grid,
        origin="lower",
        cmap="gray_r",
        extent=extent,
        interpolation="nearest",
        alpha=0.80
    )

    # Overlay obstacle outlines and labels so we can verify placement
    for obs in obstacle_shapes:
        if obs["type"] == "rect":
            w = obs["x_max"] - obs["x_min"]
            h = obs["y_max"] - obs["y_min"]

            # Cyan rectangle outline
            rect = Rectangle(
                (obs["x_min"], obs["y_min"]),
                w,
                h,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5
            )
            ax.add_patch(rect)

            # Label at the center with coordinates
            cx = obs["x_min"] + w / 2.0
            cy = obs["y_min"] + h / 2.0
            ax.text(
                cx, cy,
                f"{obs['label']}\n({cx:.2f}, {cy:.2f})",
                color="yellow",
                fontsize=8,
                ha="center",
                va="center"
            )

        elif obs["type"] == "line":
            # Magenta line for angled walls
            ax.plot(
                [obs["x1"], obs["x2"]],
                [obs["y1"], obs["y2"]],
                color="magenta",
                linewidth=3
            )

            # Label at midpoint showing start -> end
            mx = 0.5 * (obs["x1"] + obs["x2"])
            my = 0.5 * (obs["y1"] + obs["y2"])
            ax.text(
                mx, my,
                f"{obs['label']}\n({obs['x1']:.2f},{obs['y1']:.2f})→({obs['x2']:.2f},{obs['y2']:.2f})",
                color="yellow",
                fontsize=7,
                ha="center",
                va="center"
            )

    # Drop start and goal markers if provided
    if start is not None:
        ax.plot(start[0], start[1], 'go', markersize=8, label="Start")
    if goal is not None:
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Uniformly Scaled Map\n"
        f"Scale = {map_info['map_scale']:.2f}x   "
        f"Size = {map_info['x_max_m']:.1f}m x {map_info['y_max_m']:.1f}m"
    )
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # --- Robot physical dimensions (meters) ---
    wheel_radius = 0.30
    wheel_separation = 2.0 * 0.345   # 0.69 m total track width
    base_length = 1.05
    base_width = 0.65

    # Scale factor: original 2x4m arena becomes 8x16m
    map_scale = 4.0

    # For collision checking, we inflate obstacles by the robot's radius
    # plus a small safety buffer so we don't clip corners
    robot_radius = base_width / 2.0   # 0.325 m — treat the robot as a circle
    clearance = 0.02                  # 2 cm extra breathing room
    total_clearance_cm = int(round((robot_radius + clearance) * 100.0))

    # Build the map
    grid, map_info, obstacle_shapes = create_project3_map(
        total_clearance_cm,
        map_scale=map_scale
    )

    # Start and goal positions, also scaled to match the map
    start = (1.0 * map_scale, 0.30 * map_scale)   # bottom-center-ish
    goal = (1.0 * map_scale, 3.50 * map_scale)     # near the top

    # Convert to grid coords so we can check if they're in free space
    sx_cm, sy_cm = world_m_to_grid_cm(start[0], start[1], map_info)
    gx_cm, gy_cm = world_m_to_grid_cm(goal[0], goal[1], map_info)

    # Print everything so we can double-check before running the planner
    print(
        f"Scaled map bounds: x=[{map_info['x_min_m']:.2f}, {map_info['x_max_m']:.2f}], "
        f"y=[{map_info['y_min_m']:.2f}, {map_info['y_max_m']:.2f}]")
    print("Robot params:")
    print(f"  wheel_radius     = {wheel_radius:.3f} m")
    print(f"  wheel_separation = {wheel_separation:.3f} m")
    print(f"  base_length      = {base_length:.3f} m")
    print(f"  base_width       = {base_width:.3f} m")
    print("Scaling:")
    print(f"  map_scale        = {map_scale:.3f}")
    print("Planning footprint:")
    print(f"  robot_radius     = {robot_radius:.3f} m")
    print(f"  clearance        = {clearance:.3f} m")
    print(f"  total inflation  = {total_clearance_cm} cm")
    print(
        f"Start world={start} -> grid=({sx_cm:.1f}, {sy_cm:.1f}) "
        f"free={is_free(grid, sx_cm, sy_cm)}")
    print(
        f"Goal  world={goal} -> grid=({gx_cm:.1f}, {gy_cm:.1f}) "
        f"free={is_free(grid, gx_cm, gy_cm)}")

    print_obstacle_coordinates(obstacle_shapes)
    plot_map(grid, map_info, obstacle_shapes, start=start, goal=goal)


if __name__ == "__main__":
    main()